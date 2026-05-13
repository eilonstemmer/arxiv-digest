"""Trend stage: SQL rollups (W-1/W-4/W-12/W-52) + Sonnet trend narrative.

For each cluster in the current week, computes paper-count deltas relative
to the same cluster label in prior weeks, classifies velocity, persists to
`cluster_trends`, then calls Sonnet once to produce a human-readable trend
narrative.

Re-running for the same week is idempotent: existing `cluster_trends` rows
for that week are deleted and re-inserted.

On LLM failure (API or schema), the function logs a warning and returns a
`TrendResult` with `trend_payload=None`. It does NOT raise.
"""

from __future__ import annotations

import dataclasses
import datetime
import sqlite3
from pathlib import Path
from typing import Any

import structlog

from . import db, llm, settings

log = structlog.get_logger(__name__)

TOP_PAPERS_PER_CLUSTER = 5


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TrendResult:
    """Outcome of one trend-stage run."""

    week: str
    clusters_processed: int
    trend_payload: dict[str, Any] | None  # validated JSON or None on failure
    cost_usd: float


# ---------------------------------------------------------------------------
# ISO week arithmetic
# ---------------------------------------------------------------------------


def _prior_week(week_str: str, n_weeks_ago: int) -> str:
    """Return the ISO week label that is `n_weeks_ago` weeks before `week_str`.

    Input/output format: ``"YYYY-WNN"`` (e.g. ``"2026-W20"``).
    Handles year rollovers and ISO-week-53 years correctly.
    """
    year_s, w_s = week_str.split("-W")
    monday = datetime.date.fromisocalendar(int(year_s), int(w_s), 1)
    prior = monday - datetime.timedelta(weeks=n_weeks_ago)
    iso = prior.isocalendar()
    return f"{iso.year:04d}-W{iso.week:02d}"


# ---------------------------------------------------------------------------
# Delta + classification helpers
# ---------------------------------------------------------------------------


def _compute_delta(current: int, prior: int | None) -> float | None:
    """Relative change (current - prior) / prior, or None when prior is 0 / absent."""
    if prior is None or prior == 0:
        return None
    return (current - prior) / prior


def _classify_velocity(
    delta_vs_w4: float | None,
    delta_vs_w12: float | None,
) -> str:
    """Return velocity class string for one cluster."""
    if delta_vs_w4 is None:
        return "flat"
    if delta_vs_w4 > 0.3 and delta_vs_w12 is not None and delta_vs_w12 > 0.2:
        return "accelerating"
    if delta_vs_w4 > 0.1:
        return "growing"
    if delta_vs_w4 < -0.1:
        return "shrinking"
    return "flat"


def _is_new(
    conn: sqlite3.Connection,
    label: str,
    week: str,
) -> int:
    """Return 1 if `label` did not appear in `clusters` for the 12 weeks prior to `week`."""
    prior_weeks = [_prior_week(week, n) for n in range(1, 13)]
    placeholders = ",".join("?" * len(prior_weeks))
    row = conn.execute(
        f"SELECT COUNT(*) AS c FROM clusters WHERE label = ? AND week IN ({placeholders})",
        (label, *prior_weeks),
    ).fetchone()
    return 1 if int(row["c"]) == 0 else 0


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

_OPEN_CLUSTERS = "{{#each CLUSTERS}}"
_OPEN_PAPERS = "{{#each this.top_papers}}"
_CLOSE_EACH = "{{/each}}"


def _fmt_value(v: Any) -> str:
    """Format a cluster field value for template substitution."""
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _render_cluster_block(
    item_template: str,
    cluster_rows: list[dict[str, Any]],
) -> str:
    """Render the ``{{#each CLUSTERS}}`` loop body for every cluster row.

    The item_template may contain a nested ``{{#each this.top_papers}}``
    block. We resolve the inner loop first using string splitting so that
    the outer ``{{/each}}`` marker is never confused with the inner one.
    """
    parts: list[str] = []
    for cdata in cluster_rows:
        block = item_template

        # Render nested {{#each this.top_papers}} ... {{/each}} block.
        open_pos = block.find(_OPEN_PAPERS)
        if open_pos != -1:
            close_pos = block.find(_CLOSE_EACH, open_pos + len(_OPEN_PAPERS))
            if close_pos != -1:
                paper_template = block[open_pos + len(_OPEN_PAPERS) : close_pos]
                paper_lines: list[str] = []
                for p in cdata.get("top_papers", []):
                    paper_line = paper_template.replace(
                        "{{this.arxiv_id}}", str(p["arxiv_id"])
                    ).replace("{{this.title}}", str(p["title"]))
                    paper_lines.append(paper_line)
                papers_rendered = "".join(paper_lines)
                block = (
                    block[:open_pos]
                    + papers_rendered
                    + block[close_pos + len(_CLOSE_EACH) :]
                )

        # Replace per-cluster placeholders.
        for key in (
            "cluster_id",
            "label",
            "description",
            "paper_count",
            "delta_vs_w1",
            "delta_vs_w4",
            "delta_vs_w12",
            "delta_vs_w52",
            "is_new",
            "velocity_class",
        ):
            block = block.replace("{{this." + key + "}}", _fmt_value(cdata.get(key)))

        parts.append(block)

    return "".join(parts)


def _render_user_message(
    template: str,
    condensed_profile: str,
    cluster_rows: list[dict[str, Any]],
) -> str:
    """Fill all placeholders in the trend-narrative user template."""
    # 1. Fill CONDENSED_PROFILE.
    msg = llm.fill_template(template, {"CONDENSED_PROFILE": condensed_profile})

    # 2. Render the {{#each CLUSTERS}} ... {{/each}} block.
    # The template has nested {{#each}} blocks, so we locate the outer
    # open tag and take everything up to the LAST {{/each}} in the template
    # as the outer close (since the nested inner loop closes first).
    open_pos = msg.find(_OPEN_CLUSTERS)
    if open_pos != -1:
        close_pos = msg.rfind(_CLOSE_EACH)
        if close_pos != -1 and close_pos > open_pos:
            item_template = msg[open_pos + len(_OPEN_CLUSTERS) : close_pos]
            rendered = _render_cluster_block(item_template, cluster_rows)
            msg = msg[:open_pos] + rendered + msg[close_pos + len(_CLOSE_EACH) :]

    return msg


# ---------------------------------------------------------------------------
# Top-N paper selection
# ---------------------------------------------------------------------------


def _top_papers(
    conn: sqlite3.Connection,
    week: str,
    cluster_id: int,
    top_n: int = TOP_PAPERS_PER_CLUSTER,
) -> list[dict[str, Any]]:
    """Return the top-N papers for a cluster ordered by relevance_score DESC."""
    rows = conn.execute(
        """
        SELECT p.arxiv_id, p.title, p.relevance_score
          FROM paper_clusters pc
          JOIN papers p ON p.arxiv_id = pc.arxiv_id
         WHERE pc.week = ? AND pc.cluster_id = ?
         ORDER BY p.relevance_score DESC
         LIMIT ?
        """,
        (week, cluster_id, top_n),
    ).fetchall()
    return [
        {"arxiv_id": str(r["arxiv_id"]), "title": str(r["title"])}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_trends(
    conn: sqlite3.Connection,
    *,
    week: str,
    profile_path: Path | str | None = None,
    prompts_dir: Path | str | None = None,
) -> TrendResult:
    """Compute W-1/W-4/W-12/W-52 deltas, persist cluster_trends, call Sonnet.

    Parameters
    ----------
    conn:
        Open SQLite connection (schema already initialised).
    week:
        ISO week label for the current run, e.g. ``"2026-W20"``.
    profile_path:
        Path to the operator's ``profile.md``. If None the profile section
        in the prompt is left blank (empty string).
    prompts_dir:
        Directory containing ``trend_narrative.md`` and
        ``trend_narrative.schema.json``. Defaults to ``llm.DEFAULT_PROMPTS_DIR``.

    Returns
    -------
    TrendResult
        ``trend_payload`` is None on any LLM failure; the function never raises
        due to LLM issues.
    """
    # ------------------------------------------------------------------
    # 1. Fetch clusters for this week.
    # ------------------------------------------------------------------
    cluster_rows_raw = conn.execute(
        "SELECT cluster_id, label, description, paper_count FROM clusters WHERE week = ?",
        (week,),
    ).fetchall()

    if not cluster_rows_raw:
        log.info("trend.no_clusters", week=week)
        return TrendResult(
            week=week,
            clusters_processed=0,
            trend_payload=None,
            cost_usd=0.0,
        )

    # ------------------------------------------------------------------
    # 2. Compute deltas and classify for every cluster.
    # ------------------------------------------------------------------
    prior_offsets: dict[str, int] = {
        "w1": 1,
        "w4": 4,
        "w12": 12,
        "w52": 52,
    }

    cluster_data: list[dict[str, Any]] = []

    for row in cluster_rows_raw:
        cluster_id = int(row["cluster_id"])
        label = str(row["label"])
        current_count = int(row["paper_count"])

        # Look up prior counts keyed by label.
        deltas: dict[str, float | None] = {}
        prior_counts: dict[str, int | None] = {}
        for key, offset in prior_offsets.items():
            prior_week_str = _prior_week(week, offset)
            prior_row = conn.execute(
                "SELECT paper_count FROM clusters WHERE label = ? AND week = ?",
                (label, prior_week_str),
            ).fetchone()
            prior_count = int(prior_row["paper_count"]) if prior_row else None
            prior_counts[key] = prior_count
            deltas[f"delta_vs_{key}"] = _compute_delta(current_count, prior_count)

        is_new_flag = _is_new(conn, label, week)
        vc = _classify_velocity(deltas["delta_vs_w4"], deltas["delta_vs_w12"])

        top = _top_papers(conn, week, cluster_id)

        cluster_data.append(
            {
                "cluster_id": cluster_id,
                "label": label,
                "description": str(row["description"]),
                "paper_count": current_count,
                "delta_vs_w1": deltas["delta_vs_w1"],
                "delta_vs_w4": deltas["delta_vs_w4"],
                "delta_vs_w12": deltas["delta_vs_w12"],
                "delta_vs_w52": deltas["delta_vs_w52"],
                "is_new": is_new_flag,
                "velocity_class": vc,
                "top_papers": top,
            }
        )

    # ------------------------------------------------------------------
    # 3. Persist to cluster_trends (idempotent: DELETE then INSERT).
    # ------------------------------------------------------------------
    with db.transaction(conn):
        conn.execute("DELETE FROM cluster_trends WHERE week = ?", (week,))
        for cdata in cluster_data:
            conn.execute(
                """
                INSERT INTO cluster_trends(
                    week, cluster_id,
                    delta_vs_w1, delta_vs_w4, delta_vs_w12, delta_vs_w52,
                    is_new, velocity_class
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    week,
                    cdata["cluster_id"],
                    cdata["delta_vs_w1"],
                    cdata["delta_vs_w4"],
                    cdata["delta_vs_w12"],
                    cdata["delta_vs_w52"],
                    cdata["is_new"],
                    cdata["velocity_class"],
                ),
            )

    log.info(
        "trend.persisted",
        week=week,
        clusters=len(cluster_data),
    )

    # ------------------------------------------------------------------
    # 4. Generate trend narrative.
    # ------------------------------------------------------------------
    # Load profile content (condensation deferred; use full text).
    condensed_profile = ""
    if profile_path is not None:
        p = Path(profile_path)
        if p.exists():
            condensed_profile = p.read_text(encoding="utf-8").strip()

    # Load prompt.
    prompt = llm.load_prompt("trend_narrative", prompts_dir=prompts_dir)

    # Render user message.
    user_msg = _render_user_message(
        prompt.user_template,
        condensed_profile,
        cluster_data,
    )

    api_key = settings.anthropic_api_key(conn)
    model = settings.summarize_model(conn)

    result = llm.call_direct(
        model=model,
        system=prompt.system_template,
        user=user_msg,
        schema=prompt.schema,
        api_key=api_key,
    )

    if result.status not in ("success",):
        log.warning(
            "trend.narrative.failed",
            week=week,
            status=result.status,
            error=result.error,
        )
        return TrendResult(
            week=week,
            clusters_processed=len(cluster_data),
            trend_payload=None,
            cost_usd=result.cost_usd,
        )

    log.info(
        "trend.complete",
        week=week,
        clusters_processed=len(cluster_data),
        cost_usd=round(result.cost_usd, 4),
    )
    return TrendResult(
        week=week,
        clusters_processed=len(cluster_data),
        trend_payload=result.data,
        cost_usd=result.cost_usd,
    )
