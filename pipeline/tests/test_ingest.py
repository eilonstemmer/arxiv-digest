"""Tests for src.ingest: rate limiter, URL build, XML parse, dedup, fetch loop."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import db, ingest

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHETIC_XML = (FIXTURES / "synthetic_papers.xml").read_bytes()
EMPTY_XML = (FIXTURES / "empty_feed.xml").read_bytes()


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


# ---- RateLimiter ----------------------------------------------------------


def test_rate_limiter_zero_disables() -> None:
    rl = ingest.RateLimiter(capacity=1, refill_seconds=0.0)
    start = time.monotonic()
    for _ in range(20):
        rl.wait()
    assert time.monotonic() - start < 0.05


def test_rate_limiter_first_burst_no_wait() -> None:
    rl = ingest.RateLimiter(capacity=4, refill_seconds=0.1)
    start = time.monotonic()
    for _ in range(4):
        rl.wait()
    assert time.monotonic() - start < 0.05


def test_rate_limiter_enforces_after_burst() -> None:
    rl = ingest.RateLimiter(capacity=4, refill_seconds=0.1)
    for _ in range(4):
        rl.wait()
    start = time.monotonic()
    rl.wait()
    elapsed = time.monotonic() - start
    # Approximately one refill_seconds, with generous bounds for timer jitter.
    assert 0.07 <= elapsed <= 0.25


def test_rate_limiter_refills_over_time() -> None:
    rl = ingest.RateLimiter(capacity=4, refill_seconds=0.05)
    for _ in range(4):
        rl.wait()
    # Sleep long enough to accrue several tokens.
    time.sleep(0.12)
    start = time.monotonic()
    rl.wait()
    assert time.monotonic() - start < 0.02


# ---- URL building ---------------------------------------------------------


def test_build_query_url_single_category() -> None:
    url = ingest.build_query_url(["cs.AI"], start=0, max_results=10)
    assert "search_query=cat:cs.AI" in url
    assert "start=0" in url
    assert "max_results=10" in url
    assert "sortBy=submittedDate" in url
    assert "sortOrder=descending" in url


def test_build_query_url_multi_category_uses_or_join() -> None:
    url = ingest.build_query_url(["cs.AI", "cs.LG", "cs.CL"])
    assert "cat:cs.AI" in url
    assert "cat:cs.LG" in url
    assert "cat:cs.CL" in url
    assert "+OR+" in url
    # Three categories -> two OR joiners
    assert url.count("+OR+") == 2


def test_build_query_url_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ingest.build_query_url([])


def test_build_query_url_respects_base_url() -> None:
    url = ingest.build_query_url(
        ["cs.AI"], base_url="https://example.com/api/query"
    )
    assert url.startswith("https://example.com/api/query?")


# ---- load_category_codes --------------------------------------------------


def test_load_category_codes_two_groups(tmp_path: Path) -> None:
    p = tmp_path / "categories.yaml"
    p.write_text(
        """
groups:
  ai_ml:
    - cs.AI
    - cs.LG
  robotics_systems:
    - cs.RO
""",
        encoding="utf-8",
    )
    assert ingest.load_category_codes(p) == ["cs.AI", "cs.LG", "cs.RO"]


def test_load_category_codes_dedups(tmp_path: Path) -> None:
    p = tmp_path / "categories.yaml"
    p.write_text(
        """
groups:
  a:
    - cs.AI
    - cs.LG
  b:
    - cs.LG
    - cs.RO
""",
        encoding="utf-8",
    )
    # cs.LG appears in both groups; only one entry survives, in first-seen order.
    assert ingest.load_category_codes(p) == ["cs.AI", "cs.LG", "cs.RO"]


def test_load_category_codes_no_groups(tmp_path: Path) -> None:
    p = tmp_path / "categories.yaml"
    p.write_text("groups: {}\n", encoding="utf-8")
    assert ingest.load_category_codes(p) == []


def test_load_category_codes_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "categories.yaml"
    p.write_text("", encoding="utf-8")
    assert ingest.load_category_codes(p) == []


def test_load_category_codes_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "categories.yaml"
    p.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        ingest.load_category_codes(p)


def test_load_category_codes_real_default_config() -> None:
    """The shipped config/categories.yaml has ai_ml + robotics_systems uncommented."""
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / "config" / "categories.yaml"
    codes = ingest.load_category_codes(path)
    # ai_ml + robotics_systems uncommented per default.
    assert "cs.AI" in codes
    assert "cs.LG" in codes
    assert "cs.RO" in codes
    # finance_quant, biology, etc. are commented out.
    assert "q-fin.CP" not in codes
    assert "cs.CR" not in codes


# ---- arxiv_id parsing -----------------------------------------------------


@pytest.mark.parametrize(
    ("entry_id", "expected"),
    [
        ("http://arxiv.org/abs/2401.00001v1", "2401.00001"),
        ("http://arxiv.org/abs/2401.00001v2", "2401.00001"),
        ("http://arxiv.org/abs/2401.00001", "2401.00001"),
        ("http://arxiv.org/abs/2401.12345v10", "2401.12345"),
        ("http://arxiv.org/abs/2401.00001v1/", "2401.00001"),
    ],
)
def test_arxiv_id_from_entry_id(entry_id: str, expected: str) -> None:
    assert ingest._arxiv_id_from_entry_id(entry_id) == expected


# ---- XML parsing ----------------------------------------------------------


def test_parse_xml_synthetic_fixture_yields_three_papers() -> None:
    papers = ingest.parse_xml(SYNTHETIC_XML)
    assert len(papers) == 3


def test_parse_xml_first_entry_full_shape() -> None:
    papers = ingest.parse_xml(SYNTHETIC_XML)
    p = papers[0]
    assert p.arxiv_id == "2401.00001"
    assert (
        p.title
        == "Adaptive Tool-Use Memory for Long-Horizon Agents in Cluttered Environments"
    )
    assert p.authors == ["Alice Researcher"]
    assert "agent memory system" in p.abstract
    assert "\n" not in p.abstract  # whitespace collapsed
    assert p.primary_category == "cs.AI"
    assert p.categories == ["cs.AI", "cs.LG"]
    assert p.published_at == "2024-01-01T08:00:00Z"
    assert p.updated_at == "2024-01-02T08:00:00Z"
    assert p.url_abs == "http://arxiv.org/abs/2401.00001v1"
    assert p.url_pdf == "http://arxiv.org/pdf/2401.00001v1"


def test_parse_xml_strips_version_suffix_from_id() -> None:
    papers = ingest.parse_xml(SYNTHETIC_XML)
    ids = [p.arxiv_id for p in papers]
    # v2 entry should still be stored as base id
    assert "2401.00002" in ids
    assert not any("v" in i.split(".")[-1] for i in ids if "v" in i)


def test_parse_xml_handles_multiple_authors_and_unicode() -> None:
    papers = ingest.parse_xml(SYNTHETIC_XML)
    p = papers[1]
    assert p.authors == ["Sébastien Le Berre", "Carol Engineer", "Dave Postdoc"]


def test_parse_xml_collapses_title_whitespace() -> None:
    papers = ingest.parse_xml(SYNTHETIC_XML)
    # The fixture's first title spans two lines with indentation; the parser
    # must collapse to a single-line title without multiple spaces.
    assert "  " not in papers[0].title


def test_parse_xml_empty_feed() -> None:
    assert ingest.parse_xml(EMPTY_XML) == []


def test_parse_xml_skips_entries_missing_required_fields() -> None:
    # Build a feed where one entry lacks the primary_category element.
    broken = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.99999v1</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>Broken Entry</title>
    <summary>missing primary_category</summary>
    <author><name>A</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2401.00001v1</id>
    <published>2024-01-01T00:00:00Z</published>
    <title>Good Entry</title>
    <summary>ok</summary>
    <author><name>B</name></author>
    <arxiv:primary_category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""
    papers = ingest.parse_xml(broken)
    assert len(papers) == 1
    assert papers[0].arxiv_id == "2401.00001"


# ---- insert_paper + dedup -------------------------------------------------


def _sample_paper(arxiv_id: str = "2401.00001") -> ingest.ParsedPaper:
    return ingest.ParsedPaper(
        arxiv_id=arxiv_id,
        title="Test Title",
        authors=["A", "B"],
        abstract="Test abstract content.",
        primary_category="cs.AI",
        categories=["cs.AI", "cs.LG"],
        published_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-02T00:00:00Z",
        url_abs=f"https://arxiv.org/abs/{arxiv_id}",
        url_pdf=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def test_insert_paper_new_returns_true(conn: sqlite3.Connection) -> None:
    assert ingest.insert_paper(conn, _sample_paper()) is True
    row = conn.execute(
        "SELECT title, primary_category FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert row["title"] == "Test Title"
    assert row["primary_category"] == "cs.AI"


def test_insert_paper_duplicate_returns_false(conn: sqlite3.Connection) -> None:
    paper = _sample_paper()
    assert ingest.insert_paper(conn, paper) is True
    assert ingest.insert_paper(conn, paper) is False


def test_insert_paper_persists_json_columns(conn: sqlite3.Connection) -> None:
    import json

    ingest.insert_paper(conn, _sample_paper())
    row = conn.execute(
        "SELECT authors, categories FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert json.loads(row["authors"]) == ["A", "B"]
    assert json.loads(row["categories"]) == ["cs.AI", "cs.LG"]


# ---- fetch_papers end-to-end ----------------------------------------------


def test_fetch_papers_inserts_from_fixture(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "_http_get", lambda url, **kw: SYNTHETIC_XML)
    result = ingest.fetch_papers(
        conn,
        ["cs.AI"],
        max_results=3,
        rate_limiter=ingest.RateLimiter(refill_seconds=0),
    )
    assert result.fetched == 3
    assert result.new == 3
    assert result.duplicates == 0
    assert result.errors == 0

    row_count = conn.execute("SELECT COUNT(*) AS c FROM papers").fetchone()["c"]
    assert int(row_count) == 3


def test_fetch_papers_is_idempotent(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "_http_get", lambda url, **kw: SYNTHETIC_XML)
    rl = ingest.RateLimiter(refill_seconds=0)
    first = ingest.fetch_papers(conn, ["cs.AI"], max_results=3, rate_limiter=rl)
    second = ingest.fetch_papers(conn, ["cs.AI"], max_results=3, rate_limiter=rl)
    assert first.new == 3
    assert second.new == 0
    assert second.duplicates == 3


def test_fetch_papers_stops_at_short_batch(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    # batch_size=5, fixture has 3 papers -> first call returns fewer than
    # batch_size, so the loop breaks after one HTTP call.
    call_count = 0

    def fake_get(url: str, **_: object) -> bytes:
        nonlocal call_count
        call_count += 1
        return SYNTHETIC_XML

    monkeypatch.setattr(ingest, "_http_get", fake_get)
    result = ingest.fetch_papers(
        conn,
        ["cs.AI"],
        max_results=20,
        batch_size=5,
        rate_limiter=ingest.RateLimiter(refill_seconds=0),
    )
    assert call_count == 1
    assert result.fetched == 3


def test_fetch_papers_paginates_when_full_batch_returned(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Trick the loop into thinking a "full batch" came back, so it paginates.
    # First call returns SYNTHETIC (3 papers) with batch=3 (full).
    # Second call returns EMPTY (0 papers) -> loop exits via 'if not papers'.
    responses = [SYNTHETIC_XML, EMPTY_XML]
    call_count = 0

    def fake_get(url: str, **_: object) -> bytes:
        nonlocal call_count
        resp = responses[call_count]
        call_count += 1
        return resp

    monkeypatch.setattr(ingest, "_http_get", fake_get)
    result = ingest.fetch_papers(
        conn,
        ["cs.AI"],
        max_results=10,
        batch_size=3,
        rate_limiter=ingest.RateLimiter(refill_seconds=0),
    )
    assert call_count == 2
    assert result.fetched == 3
    assert result.new == 3


def test_fetch_papers_empty_categories_raises(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="at least one"):
        ingest.fetch_papers(
            conn, [], rate_limiter=ingest.RateLimiter(refill_seconds=0)
        )


def test_fetch_papers_http_error_breaks_loop(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    import urllib.error

    def fake_get(url: str, **_: object) -> bytes:
        raise urllib.error.URLError("simulated network failure")

    monkeypatch.setattr(ingest, "_http_get", fake_get)
    result = ingest.fetch_papers(
        conn,
        ["cs.AI"],
        max_results=100,
        rate_limiter=ingest.RateLimiter(refill_seconds=0),
    )
    assert result.fetched == 0
    assert result.errors == 1


def test_fetch_papers_xml_parse_error_breaks_loop(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ingest, "_http_get", lambda url, **kw: b"not valid xml")
    result = ingest.fetch_papers(
        conn,
        ["cs.AI"],
        max_results=100,
        rate_limiter=ingest.RateLimiter(refill_seconds=0),
    )
    assert result.errors == 1
    assert result.fetched == 0
