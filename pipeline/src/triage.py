"""Triage stage: profile-similarity prefilter + Haiku 4.5 Batch scoring.

Pipeline flow (PROJECT_BRIEF.md section 6, step 6):

  a. Compute the profile embedding once per run.
     `config/profile.md` is hashed (SHA-256); if the hash matches the
     cached `profile_embedding_hash` setting, we decode the cached
     base64 vector from `profile_embedding_b64`. Otherwise we embed
     fresh with MiniLM and update both settings.

  b. Score every untriaged paper (embedding IS NOT NULL,
     triaged_at IS NULL) by cosine similarity to the profile embedding.
     Write `papers.prefilter_score` for all of them.

  c. Keep the top `triage_prefilter_keep_fraction` (default 0.4) by
     prefilter_score. The remainder get `relevance_score = 0.0`,
     `relevance_reason = "below profile-similarity prefilter"`,
     `triaged_at = now`. They stay in the DB and remain searchable;
     they just don't burn a Haiku call.

  d. Send the survivors to Claude Haiku 4.5 via Batch API using
     `prompts/triage.md` + `triage.schema.json`. Persist the parsed
     response back to the papers table.

If `triage_prefilter_enabled` is False, steps a-c are skipped and
every eligible paper goes straight to Haiku.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import structlog

from . import db, embed, llm, settings

log = structlog.get_logger(__name__)

PREFILTER_DROPPED_REASON = "below profile-similarity prefilter"
DEFAULT_PROFILE_PATH = Path(os.environ.get("PROFILE_PATH", "/config/profile.md"))


@dataclasses.dataclass(frozen=True)
class TriageResult:
    """Outcome counters for one triage stage run.

    `papers_prefiltered + papers_haiku_requested == papers_eligible`,
    and `papers_haiku_succeeded + papers_haiku_failed == papers_haiku_requested`.
    """

    papers_eligible: int
    papers_prefiltered: int
    papers_haiku_requested: int
    papers_haiku_succeeded: int
    papers_haiku_failed: int
    cost_usd: float


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Profile embedding cache
# ---------------------------------------------------------------------------


def profile_embedding(
    conn: sqlite3.Connection,
    *,
    profile_path: Path | str,
) -> npt.NDArray[np.float32]:
    """Return the profile embedding, computing fresh on first use or when
    `profile.md` content changes (detected via SHA-256).

    The cached vector lives in two pipeline-managed settings keys:
    `profile_embedding_hash` and `profile_embedding_b64`. The webapp's
    Settings page intentionally does not expose them.
    """
    text = Path(profile_path).read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cached_hash = settings.profile_embedding_hash(conn)
    cached_b64 = settings.profile_embedding_b64(conn)

    if cached_hash == sha and cached_b64:
        log.info("triage.profile_embedding.cached", hash=sha[:12])
        return embed.deserialize(base64.b64decode(cached_b64))

    log.info(
        "triage.profile_embedding.computing",
        hash=sha[:12],
        prev_hash=cached_hash[:12] if cached_hash else None,
    )
    vec = embed.embed_text(text)
    b64 = base64.b64encode(embed.serialize(vec)).decode("ascii")
    db.set_setting(conn, "profile_embedding_hash", sha)
    db.set_setting(conn, "profile_embedding_b64", b64)
    return vec


# ---------------------------------------------------------------------------
# Eligibility + prefilter
# ---------------------------------------------------------------------------


def _eligible_papers(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Untriaged papers that have an embedding. Ordered newest first."""
    return list(
        conn.execute(
            """
            SELECT arxiv_id, title, authors, abstract, primary_category,
                   categories, embedding
            FROM papers
            WHERE triaged_at IS NULL AND embedding IS NOT NULL
            ORDER BY published_at DESC
            """
        )
    )


def _run_prefilter(
    conn: sqlite3.Connection,
    *,
    rows: list[sqlite3.Row],
    keep_fraction: float,
    profile_vec: npt.NDArray[np.float32],
) -> list[sqlite3.Row]:
    """Score all rows, write `prefilter_score` for each, mark drops as
    triaged with relevance_score=0. Returns the survivor rows."""
    if not rows:
        return []

    corpus = np.stack(
        [embed.deserialize(bytes(r["embedding"])) for r in rows]
    )
    sims = embed.cosine_similarity_batch(profile_vec, corpus)

    n = len(rows)
    keep_count = max(1, round(n * keep_fraction))
    keep_count = min(keep_count, n)

    # argsort ascending; reverse via [::-1] for desc order
    order = np.argsort(sims)[::-1]
    survivor_indices = set(int(i) for i in order[:keep_count])

    survivors: list[sqlite3.Row] = []
    now = _now_iso()
    with db.transaction(conn):
        for i, row in enumerate(rows):
            arxiv_id = str(row["arxiv_id"])
            score = float(sims[i])
            if i in survivor_indices:
                conn.execute(
                    "UPDATE papers SET prefilter_score = ? WHERE arxiv_id = ?",
                    (score, arxiv_id),
                )
                survivors.append(row)
            else:
                conn.execute(
                    """
                    UPDATE papers
                       SET prefilter_score = ?,
                           relevance_score = 0.0,
                           relevance_reason = ?,
                           triaged_at = ?
                     WHERE arxiv_id = ?
                    """,
                    (score, PREFILTER_DROPPED_REASON, now, arxiv_id),
                )

    log.info(
        "triage.prefilter.complete",
        total=n,
        kept=len(survivors),
        dropped=n - len(survivors),
        keep_fraction=keep_fraction,
    )
    return survivors


# ---------------------------------------------------------------------------
# Haiku batch
# ---------------------------------------------------------------------------


def _build_user_message(prompt: llm.Prompt, row: sqlite3.Row) -> str:
    authors = json.loads(str(row["authors"]))
    categories = json.loads(str(row["categories"]))
    return llm.fill_template(
        prompt.user_template,
        {
            "TITLE": str(row["title"]),
            "AUTHORS_COMMA_SEPARATED": ", ".join(str(a) for a in authors),
            "PRIMARY_CATEGORY": str(row["primary_category"]),
            "CATEGORIES_COMMA_SEPARATED": ", ".join(
                str(c) for c in categories
            ),
            "ABSTRACT": str(row["abstract"]),
        },
    )


def _run_haiku_stage(
    conn: sqlite3.Connection,
    *,
    survivors: list[sqlite3.Row],
    profile_text: str,
    api_key: str,
    model: str,
    prompts_dir: Path | str | None,
) -> tuple[int, int, float]:
    """Run Haiku Batch on `survivors`. Writes results to the papers table.
    Returns (succeeded, failed, total_cost_usd)."""
    if not survivors:
        return 0, 0, 0.0

    prompt = llm.load_prompt("triage", prompts_dir=prompts_dir)
    system = llm.fill_template(
        prompt.system_template, {"PROFILE_MARKDOWN": profile_text}
    )
    requests = [
        llm.BatchRequest(
            custom_id=str(row["arxiv_id"]),
            user_message=_build_user_message(prompt, row),
        )
        for row in survivors
    ]
    log.info("triage.haiku.submitting", count=len(requests), model=model)
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
    now = _now_iso()
    with db.transaction(conn):
        for result in results:
            total_cost += result.cost_usd
            data = result.data
            if result.status == "success" and data is not None:
                score = float(data["relevance_score"])
                reason = str(data["reason"])[:500]
                conn.execute(
                    """
                    UPDATE papers
                       SET relevance_score = ?,
                           relevance_reason = ?,
                           triaged_at = ?
                     WHERE arxiv_id = ?
                    """,
                    (score, reason, now, result.custom_id),
                )
                succeeded += 1
            else:
                err = (result.error or "unknown error")[:480]
                conn.execute(
                    """
                    UPDATE papers
                       SET relevance_score = 0.0,
                           relevance_reason = ?,
                           triaged_at = ?
                     WHERE arxiv_id = ?
                    """,
                    (f"triage failed: {err}", now, result.custom_id),
                )
                failed += 1

    log.info(
        "triage.haiku.complete",
        succeeded=succeeded,
        failed=failed,
        cost_usd=round(total_cost, 4),
    )
    return succeeded, failed, total_cost


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def triage(
    conn: sqlite3.Connection,
    *,
    profile_path: Path | str | None = None,
    prompts_dir: Path | str | None = None,
) -> TriageResult:
    """Run the full triage stage.

    Reads `triage_prefilter_enabled`, `triage_prefilter_keep_fraction`,
    `triage_model`, and `anthropic_api_key` from settings (with env-var
    fallback per the standard chain). Raises MissingSettingError if the
    API key is not set anywhere.
    """
    profile_path = (
        Path(profile_path) if profile_path is not None else DEFAULT_PROFILE_PATH
    )

    rows = _eligible_papers(conn)
    eligible = len(rows)
    if eligible == 0:
        log.info("triage.no_eligible_papers")
        return TriageResult(
            papers_eligible=0,
            papers_prefiltered=0,
            papers_haiku_requested=0,
            papers_haiku_succeeded=0,
            papers_haiku_failed=0,
            cost_usd=0.0,
        )

    api_key = settings.anthropic_api_key(conn)
    model = settings.triage_model(conn)
    prefilter_enabled = settings.triage_prefilter_enabled(conn)
    keep_fraction = settings.triage_prefilter_keep_fraction(conn)

    if prefilter_enabled:
        profile_vec = profile_embedding(conn, profile_path=profile_path)
        survivors = _run_prefilter(
            conn,
            rows=rows,
            keep_fraction=keep_fraction,
            profile_vec=profile_vec,
        )
        prefiltered = eligible - len(survivors)
    else:
        survivors = rows
        prefiltered = 0
        log.info("triage.prefilter_disabled", eligible=eligible)

    profile_text = Path(profile_path).read_text(encoding="utf-8")
    succeeded, failed, cost = _run_haiku_stage(
        conn,
        survivors=survivors,
        profile_text=profile_text,
        api_key=api_key,
        model=model,
        prompts_dir=prompts_dir,
    )

    return TriageResult(
        papers_eligible=eligible,
        papers_prefiltered=prefiltered,
        papers_haiku_requested=len(survivors),
        papers_haiku_succeeded=succeeded,
        papers_haiku_failed=failed,
        cost_usd=cost,
    )
