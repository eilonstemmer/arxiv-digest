"""arXiv API ingest: rate-limited fetch, Atom XML parsing, dedup-by-arxiv_id.

The 3-second-between-requests rule is a hard arXiv requirement. The
RateLimiter class implements a token bucket matching arXiv's stated terms
(bursts of 4, refill 1 token per 3s). Tests pass `refill_seconds=0` to
disable rate limiting so they run instantly.

`_http_get` is the only network egress point in the module. Tests
monkeypatch it to inject fixture XML; production code uses the real
urllib path.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import structlog
import yaml

log = structlog.get_logger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3.0
ARXIV_BURST_CAPACITY = 4
ARXIV_BATCH_SIZE = 100
DEFAULT_USER_AGENT = "arxiv-digest/0.1"

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
_NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS}


@dataclasses.dataclass(frozen=True)
class ParsedPaper:
    """Plain-data representation of one arXiv Atom entry."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    primary_category: str
    categories: list[str]
    published_at: str
    updated_at: str | None
    url_abs: str
    url_pdf: str


@dataclasses.dataclass(frozen=True)
class IngestResult:
    """Counters returned by `fetch_papers`."""

    fetched: int
    new: int
    duplicates: int
    errors: int


class RateLimiter:
    """Token bucket matching arXiv's stated policy.

    Default: capacity 4, refill 1 token per 3 seconds. `wait()` consumes
    one token; if the bucket is empty it blocks until enough has refilled
    for one. Single-process; not thread-safe (the ingest stage is sequential).

    Pass `refill_seconds=0` to disable rate limiting entirely (used in tests).
    """

    def __init__(
        self,
        capacity: int = ARXIV_BURST_CAPACITY,
        refill_seconds: float = ARXIV_RATE_LIMIT_SECONDS,
    ) -> None:
        self.capacity = capacity
        self.refill_seconds = refill_seconds
        self._tokens: float = float(capacity)
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        if self.refill_seconds <= 0:
            self._tokens = float(self.capacity)
            return
        now = time.monotonic()
        elapsed = now - self._last_refill
        added = elapsed / self.refill_seconds
        self._tokens = min(float(self.capacity), self._tokens + added)
        self._last_refill = now

    def wait(self) -> None:
        if self.refill_seconds <= 0:
            return
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return
        sleep_for = (1.0 - self._tokens) * self.refill_seconds
        log.debug("ratelimiter.sleep", seconds=round(sleep_for, 3))
        time.sleep(sleep_for)
        self._refill()
        self._tokens -= 1.0


def _http_get(
    url: str,
    *,
    timeout: float = 30.0,
    user_agent: str = DEFAULT_USER_AGENT,
) -> bytes:
    """Single network egress point. Tests monkeypatch this to inject fixtures."""
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body: bytes = resp.read()
        return body


def load_category_codes(yaml_path: str | Path) -> list[str]:
    """Parse config/categories.yaml into a flat, de-duplicated list of codes.

    Commented-out groups are absent from the parsed YAML; only currently
    active groups contribute. Empty / unparseable groups are skipped with
    a warning rather than raising, so a partial config still ingests
    something useful.
    """
    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: top-level must be a YAML mapping")
    groups = raw.get("groups")
    if groups is None:
        return []
    if not isinstance(groups, dict):
        raise ValueError(f"{yaml_path}: 'groups' must be a YAML mapping")

    seen: dict[str, None] = {}  # insertion-ordered dedup
    for group_name, codes in groups.items():
        if not isinstance(codes, list):
            log.warning("ingest.config.skip_invalid_group", group=str(group_name))
            continue
        for code in codes:
            if isinstance(code, str) and code:
                seen[code] = None
    return list(seen.keys())


def build_query_url(
    categories: Iterable[str],
    *,
    start: int = 0,
    max_results: int = ARXIV_BATCH_SIZE,
    base_url: str = ARXIV_API_URL,
) -> str:
    """Build an arXiv API URL for a multi-category OR search, ordered by
    submission date descending.

    arXiv expects unescaped `+` and `:` inside `search_query`, so we
    build the query string manually rather than letting urlencode escape
    them.
    """
    cat_list = list(categories)
    if not cat_list:
        raise ValueError("at least one category code is required")
    search_query = "+OR+".join(f"cat:{c}" for c in cat_list)
    params = {
        "search_query": search_query,
        "start": str(start),
        "max_results": str(max_results),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    qs = "&".join(
        f"{k}={urllib.parse.quote(v, safe='+:')}" for k, v in params.items()
    )
    return f"{base_url}?{qs}"


def _arxiv_id_from_entry_id(entry_id: str) -> str:
    """Extract the base arXiv ID from an Atom entry's `<id>` URL.

    arXiv entry IDs look like 'http://arxiv.org/abs/2401.00001v2'. We
    strip the URL prefix and the trailing version suffix so the v1 and
    v2 of the same paper share a primary key.
    """
    tail = entry_id.rstrip("/").rsplit("/", 1)[-1]
    base, sep, version = tail.rpartition("v")
    if sep == "v" and version.isdigit():
        return base
    return tail


def _normalize_ws(s: str) -> str:
    """Collapse all whitespace runs to single spaces. arXiv wraps long
    titles across lines; we want flat text."""
    return " ".join(s.split())


def _text(el: ET.Element | None) -> str:
    if el is None or el.text is None:
        raise ValueError("missing required text element")
    return el.text


def _parse_entry(entry: ET.Element) -> ParsedPaper:
    entry_id = _text(entry.find("atom:id", _NS))
    arxiv_id = _arxiv_id_from_entry_id(entry_id)
    title = _normalize_ws(_text(entry.find("atom:title", _NS)))
    abstract = _normalize_ws(_text(entry.find("atom:summary", _NS)))
    published_at = _text(entry.find("atom:published", _NS)).strip()
    updated_el = entry.find("atom:updated", _NS)
    updated_at = (
        updated_el.text.strip()
        if updated_el is not None and updated_el.text is not None
        else None
    )

    authors: list[str] = []
    for author in entry.findall("atom:author", _NS):
        name_el = author.find("atom:name", _NS)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())
    if not authors:
        raise ValueError(f"no authors in entry {arxiv_id}")

    primary_cat_el = entry.find("arxiv:primary_category", _NS)
    if primary_cat_el is None:
        raise ValueError(f"no primary_category in entry {arxiv_id}")
    primary_category = primary_cat_el.attrib["term"]

    categories: list[str] = []
    for cat in entry.findall("atom:category", _NS):
        term = cat.attrib.get("term")
        if term and term not in categories:
            categories.append(term)
    if primary_category not in categories:
        categories.insert(0, primary_category)

    url_abs = ""
    url_pdf = ""
    for link in entry.findall("atom:link", _NS):
        rel = link.attrib.get("rel", "")
        href = link.attrib.get("href", "")
        link_type = link.attrib.get("type", "")
        if rel == "alternate" and link_type == "text/html":
            url_abs = href
        elif link_type == "application/pdf":
            url_pdf = href
    if not url_abs:
        url_abs = f"https://arxiv.org/abs/{arxiv_id}"
    if not url_pdf:
        url_pdf = f"https://arxiv.org/pdf/{arxiv_id}"

    return ParsedPaper(
        arxiv_id=arxiv_id,
        title=title,
        authors=authors,
        abstract=abstract,
        primary_category=primary_category,
        categories=categories,
        published_at=published_at,
        updated_at=updated_at,
        url_abs=url_abs,
        url_pdf=url_pdf,
    )


def parse_xml(xml_bytes: bytes) -> list[ParsedPaper]:
    """Parse an arXiv Atom XML response. Skips and logs malformed entries.

    A top-level ET.ParseError propagates -- the caller (fetch_papers)
    catches it and bumps the errors counter.
    """
    root = ET.fromstring(xml_bytes)
    out: list[ParsedPaper] = []
    for entry in root.findall("atom:entry", _NS):
        try:
            paper = _parse_entry(entry)
        except (ValueError, KeyError) as exc:
            log.error("ingest.parse.entry_failed", error=str(exc))
            continue
        out.append(paper)
    return out


def insert_paper(conn: sqlite3.Connection, paper: ParsedPaper) -> bool:
    """Insert a paper if its arxiv_id is new. Returns True if inserted,
    False if the row already exists.

    Stores `authors` and `categories` as JSON arrays. `fetched_at` is
    now-UTC at insert time.
    """
    now = datetime.now(UTC).isoformat(timespec="seconds")
    try:
        conn.execute(
            """
            INSERT INTO papers(arxiv_id, title, authors, abstract,
                               primary_category, categories, published_at,
                               updated_at, fetched_at, url_abs, url_pdf)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                paper.arxiv_id,
                paper.title,
                json.dumps(paper.authors, ensure_ascii=False),
                paper.abstract,
                paper.primary_category,
                json.dumps(paper.categories, ensure_ascii=False),
                paper.published_at,
                paper.updated_at,
                now,
                paper.url_abs,
                paper.url_pdf,
            ),
        )
        return True
    except sqlite3.IntegrityError:
        return False


def fetch_papers(
    conn: sqlite3.Connection,
    categories: list[str],
    *,
    max_results: int = 2000,
    batch_size: int = ARXIV_BATCH_SIZE,
    rate_limiter: RateLimiter | None = None,
) -> IngestResult:
    """Fetch recent papers across `categories`, paginating in batches of
    `batch_size` up to `max_results`. Dedup against the papers table.

    Categories are queried as a single multi-category OR search ordered
    by submission date descending. The pagination loop stops when arXiv
    returns fewer entries than requested (end of feed) or when
    max_results is reached.
    """
    if not categories:
        raise ValueError("at least one category code is required")
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    fetched = 0
    new = 0
    duplicates = 0
    errors = 0

    start = 0
    while start < max_results:
        batch = min(batch_size, max_results - start)
        url = build_query_url(categories, start=start, max_results=batch)
        rate_limiter.wait()
        log.info("ingest.fetch.batch", start=start, batch=batch)
        try:
            xml_bytes = _http_get(url)
        except urllib.error.URLError as exc:
            log.error("ingest.fetch.http_error", error=str(exc), start=start)
            errors += 1
            break
        try:
            papers = parse_xml(xml_bytes)
        except ET.ParseError as exc:
            log.error("ingest.parse.xml_error", error=str(exc), start=start)
            errors += 1
            break

        fetched += len(papers)
        if not papers:
            break

        for paper in papers:
            if insert_paper(conn, paper):
                new += 1
            else:
                duplicates += 1

        if len(papers) < batch:
            # Fewer than requested -> end of feed reached.
            break
        start += batch

    log.info(
        "ingest.complete",
        categories=len(categories),
        fetched=fetched,
        new=new,
        duplicates=duplicates,
        errors=errors,
    )
    return IngestResult(
        fetched=fetched, new=new, duplicates=duplicates, errors=errors
    )
