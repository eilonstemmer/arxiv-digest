"""Summarize stage: Sonnet Batch summaries with cross-domain candidate selection.

Pipeline flow (PROJECT_BRIEF.md section 6, step 8):

  1. SELECT papers WHERE relevance_score > 0 AND summarized = 0 AND
     embedding IS NOT NULL ORDER BY relevance_score DESC LIMIT digest_top_n.

  2. Build an in-memory FAISS IndexFlatIP from L2-normalised embeddings of
     ALL papers with an embedding published within the prior 90 days
     (relative to the injected `now`, defaulting to datetime.now(UTC)).

  3. For each eligible paper, search the FAISS index for the 20 nearest
     neighbours whose primary_category differs from the paper's own.
     Keep the top 3 with cosine similarity > 0.65.

  4. Build a BatchRequest per paper, filling prompts/summarize.md with the
     simple placeholders AND rendering the {{#each CANDIDATES}} loop
     (or the {{#if NO_CANDIDATES}} block when empty).

  5. Fill the system prompt {{CONDENSED_PROFILE}} with the full profile.md.

  6. llm.run_batch with summarize_model (default "claude-sonnet-4-6").

  7. Writeback: INSERT INTO summaries, UPDATE papers SET summarized = 1.
     On failure: leave summarized = 0, log warning.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt
import structlog

from . import db, embed, llm, settings

log = structlog.get_logger(__name__)

DEFAULT_PROFILE_PATH = Path(os.environ.get("PROFILE_PATH", "/config/profile.md"))
FAISS_WINDOW_DAYS = 90
FAISS_TOP_K = 20
CROSS_DOMAIN_MAX = 3
CROSS_DOMAIN_MIN_SIM = 0.65
NO_CANDIDATES_TEXT = (
    "(No candidate cross-domain papers met the similarity threshold this week.\n"
    "Return `cross_domain_hooks: []`.)"
)

@dataclasses.dataclass(frozen=True)
class SummarizeResult:
    """Outcome counters for one summarize stage run."""

    papers_eligible: int
    papers_summarized_succeeded: int
    papers_summarized_failed: int
    total_cross_domain_hooks: int
    cost_usd: float


# ---------------------------------------------------------------------------
# Candidate data container
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Candidate:
    arxiv_id: str
    title: str
    abstract: str
    primary_category: str
    similarity: float


# ---------------------------------------------------------------------------
# Eligibility query
# ---------------------------------------------------------------------------


def _eligible_papers(
    conn: sqlite3.Connection, *, top_n: int
) -> list[sqlite3.Row]:
    """Papers with relevance_score > 0, not yet summarized, have embeddings.

    Ordered by relevance_score descending, limited to top_n.
    """
    return list(
        conn.execute(
            """
            SELECT arxiv_id, title, authors, abstract, primary_category,
                   categories, embedding
            FROM papers
            WHERE relevance_score > 0
              AND summarized = 0
              AND embedding IS NOT NULL
            ORDER BY relevance_score DESC
            LIMIT ?
            """,
            (top_n,),
        )
    )


# ---------------------------------------------------------------------------
# FAISS index over 90-day corpus
# ---------------------------------------------------------------------------


def _build_faiss_index(
    conn: sqlite3.Connection,
    *,
    now: datetime,
) -> tuple[faiss.IndexFlatIP, list[sqlite3.Row]]:
    """Build an IndexFlatIP from all papers with embeddings published in the
    prior 90 days (relative to `now`). Returns (index, corpus_rows)."""
    cutoff = (now - timedelta(days=FAISS_WINDOW_DAYS)).isoformat(
        timespec="seconds"
    )
    rows = list(
        conn.execute(
            """
            SELECT arxiv_id, title, abstract, primary_category, embedding
            FROM papers
            WHERE embedding IS NOT NULL
              AND published_at >= ?
            """,
            (cutoff,),
        )
    )

    index = faiss.IndexFlatIP(embed.EMBEDDING_DIM)
    if not rows:
        return index, rows

    vecs = np.stack([embed.deserialize(bytes(r["embedding"])) for r in rows])
    # L2-normalise before adding so inner product == cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    normalised: npt.NDArray[np.float32] = (vecs / norms).astype(np.float32)
    index.add(normalised)

    log.info(
        "summarize.faiss.built",
        corpus_size=len(rows),
        cutoff=cutoff,
    )
    return index, rows


# ---------------------------------------------------------------------------
# Cross-domain candidate selection
# ---------------------------------------------------------------------------


def _find_cross_domain_candidates(
    *,
    paper_vec: npt.NDArray[np.float32],
    paper_category: str,
    index: faiss.IndexFlatIP,
    corpus_rows: list[sqlite3.Row],
) -> list[_Candidate]:
    """Return up to CROSS_DOMAIN_MAX candidates with different primary_category
    and cosine similarity > CROSS_DOMAIN_MIN_SIM, ordered by similarity desc.
    """
    if index.ntotal == 0:
        return []

    # L2-normalise the query
    norm = float(np.linalg.norm(paper_vec))
    if norm == 0.0:
        return []
    query_norm: npt.NDArray[np.float32] = (paper_vec / norm).astype(np.float32)

    k = min(FAISS_TOP_K, index.ntotal)
    dists, idxs = index.search(query_norm.reshape(1, -1), k)
    # dists[0]: inner products; for unit vectors == cosine similarity

    candidates: list[_Candidate] = []
    for dist_val, idx_val in zip(dists[0], idxs[0], strict=False):
        sim = float(dist_val)
        idx = int(idx_val)
        if idx < 0:
            continue
        row = corpus_rows[idx]
        if str(row["primary_category"]) == paper_category:
            continue
        if sim <= CROSS_DOMAIN_MIN_SIM:
            continue
        candidates.append(
            _Candidate(
                arxiv_id=str(row["arxiv_id"]),
                title=str(row["title"]),
                abstract=str(row["abstract"]),
                primary_category=str(row["primary_category"]),
                similarity=sim,
            )
        )

    # Sort descending by similarity, take top 3
    candidates.sort(key=lambda c: c.similarity, reverse=True)
    return candidates[:CROSS_DOMAIN_MAX]


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _render_candidates_block(template: str, candidates: list[_Candidate]) -> str:
    """Replace the {{#each CANDIDATES}}...{{/each}} block with rendered text
    and remove (or replace) the {{#if NO_CANDIDATES}}...{{/if}} block.

    Uses a lambda replacement so backslashes in titles/abstracts are safe.
    """
    each_pattern = re.compile(
        r"\{\{#each CANDIDATES\}\}.*?\{\{/each\}\}", re.DOTALL
    )
    if_pattern = re.compile(
        r"\{\{#if NO_CANDIDATES\}\}.*?\{\{/if\}\}", re.DOTALL
    )

    if candidates:
        parts: list[str] = []
        for c in candidates:
            line = (
                f"- **{c.arxiv_id}** "
                f"(sim={c.similarity:.2f}, category={c.primary_category})\n"
                f"  Title: {c.title}\n"
                f"  Abstract: {c.abstract}"
            )
            parts.append(line)
        rendered = "\n".join(parts)
        out = each_pattern.sub(lambda _: rendered, template, count=1)
        out = if_pattern.sub("", out)
    else:
        out = each_pattern.sub("", template, count=1)
        out = if_pattern.sub(lambda _: NO_CANDIDATES_TEXT, out)

    return out


def _build_user_message(
    user_template: str,
    row: sqlite3.Row,
    candidates: list[_Candidate],
) -> str:
    """Fill the user template with per-paper fields and rendered candidates."""
    authors = json.loads(str(row["authors"]))
    categories = json.loads(str(row["categories"]))

    # First, fill scalar placeholders
    filled = llm.fill_template(
        user_template,
        {
            "ARXIV_ID": str(row["arxiv_id"]),
            "TITLE": str(row["title"]),
            "AUTHORS_COMMA_SEPARATED": ", ".join(str(a) for a in authors),
            "PRIMARY_CATEGORY": str(row["primary_category"]),
            "CATEGORIES_COMMA_SEPARATED": ", ".join(str(c) for c in categories),
            "ABSTRACT": str(row["abstract"]),
        },
    )

    # Then render the Handlebars-style loop
    return _render_candidates_block(filled, candidates)


# ---------------------------------------------------------------------------
# Summarize stage result writeback
# ---------------------------------------------------------------------------


def _write_success(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    data: dict[str, Any],
    model: str,
    now_iso: str,
) -> None:
    """INSERT summary row and mark the paper as summarized."""
    with db.transaction(conn):
        conn.execute(
            """
            INSERT INTO summaries(
                arxiv_id, problem, method, key_result, why_it_matters,
                cross_domain_hooks, novelty_signal, tags, generated_at, model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                arxiv_id,
                str(data["problem"]),
                str(data["method"]),
                str(data["key_result"]),
                str(data["why_it_matters"]),
                json.dumps(data["cross_domain_hooks"]),
                str(data["novelty_signal"]),
                json.dumps(data["tags"]),
                now_iso,
                model,
            ),
        )
        conn.execute(
            "UPDATE papers SET summarized = 1 WHERE arxiv_id = ?",
            (arxiv_id,),
        )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def summarize(
    conn: sqlite3.Connection,
    *,
    profile_path: Path | str | None = None,
    prompts_dir: Path | str | None = None,
    now: datetime | None = None,
) -> SummarizeResult:
    """Run the full summarize stage.

    Reads `digest_top_n`, `summarize_model`, and `anthropic_api_key` from
    settings (with env-var fallback per the standard chain). Raises
    MissingSettingError if the API key is not set anywhere.

    Parameters
    ----------
    conn:
        Open SQLite connection with schema initialised.
    profile_path:
        Path to profile.md. Defaults to PROFILE_PATH env var or
        /config/profile.md.
    prompts_dir:
        Directory containing summarize.md and summarize.schema.json.
        Defaults to PROMPTS_DIR env var or /prompts.
    now:
        Override the current timestamp used for the 90-day FAISS window.
        Defaults to datetime.now(UTC).
    """
    profile_path = (
        Path(profile_path) if profile_path is not None else DEFAULT_PROFILE_PATH
    )
    if now is None:
        now = datetime.now(UTC)

    top_n = settings.digest_top_n(conn)
    rows = _eligible_papers(conn, top_n=top_n)
    eligible = len(rows)

    if eligible == 0:
        log.info("summarize.no_eligible_papers")
        return SummarizeResult(
            papers_eligible=0,
            papers_summarized_succeeded=0,
            papers_summarized_failed=0,
            total_cross_domain_hooks=0,
            cost_usd=0.0,
        )

    api_key = settings.anthropic_api_key(conn)
    model = settings.summarize_model(conn)
    prompt = llm.load_prompt("summarize", prompts_dir=prompts_dir)
    user_template = prompt.user_template

    profile_text = profile_path.read_text(encoding="utf-8")
    system = llm.fill_template(
        prompt.system_template, {"CONDENSED_PROFILE": profile_text}
    )

    # Build FAISS index over the prior 90 days
    index, corpus_rows = _build_faiss_index(conn, now=now)

    # Build one BatchRequest per eligible paper
    requests: list[llm.BatchRequest] = []
    candidates_by_id: dict[str, list[_Candidate]] = {}

    for row in rows:
        arxiv_id = str(row["arxiv_id"])
        paper_vec = embed.deserialize(bytes(row["embedding"]))
        paper_category = str(row["primary_category"])

        candidates = _find_cross_domain_candidates(
            paper_vec=paper_vec,
            paper_category=paper_category,
            index=index,
            corpus_rows=corpus_rows,
        )
        candidates_by_id[arxiv_id] = candidates

        user_msg = _build_user_message(user_template, row, candidates)
        requests.append(
            llm.BatchRequest(custom_id=arxiv_id, user_message=user_msg)
        )

    log.info(
        "summarize.batch.submitting",
        count=len(requests),
        model=model,
    )

    results = llm.run_batch(
        model=model,
        system=system,
        requests=requests,
        schema=prompt.schema,
        api_key=api_key,
    )

    succeeded = 0
    failed = 0
    total_cost = 0.0
    total_hooks = 0
    now_iso = now.isoformat(timespec="seconds")

    for result in results:
        total_cost += result.cost_usd
        data = result.data
        if result.status == "success" and data is not None:
            hooks = data.get("cross_domain_hooks", [])
            hook_count = len(hooks) if isinstance(hooks, list) else 0
            total_hooks += hook_count
            try:
                _write_success(
                    conn,
                    arxiv_id=result.custom_id,
                    data=data,
                    model=model,
                    now_iso=now_iso,
                )
                succeeded += 1
            except Exception as exc:
                log.warning(
                    "summarize.writeback.error",
                    arxiv_id=result.custom_id,
                    error=str(exc),
                )
                failed += 1
        else:
            err = (result.error or "unknown error")[:480]
            log.warning(
                "summarize.result.failed",
                arxiv_id=result.custom_id,
                status=result.status,
                error=err,
            )
            failed += 1

    log.info(
        "summarize.complete",
        eligible=eligible,
        succeeded=succeeded,
        failed=failed,
        total_cross_domain_hooks=total_hooks,
        cost_usd=round(total_cost, 4),
    )
    return SummarizeResult(
        papers_eligible=eligible,
        papers_summarized_succeeded=succeeded,
        papers_summarized_failed=failed,
        total_cross_domain_hooks=total_hooks,
        cost_usd=total_cost,
    )


# Keep the corpus index available to helpers (unused externally, but ensures
# the `corpus_id_to_idx` variable is referenced in a type-correct way).
__all__ = ["SummarizeResult", "summarize"]
