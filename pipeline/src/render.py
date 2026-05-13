"""Render stage: Jinja2 HTML generation for weekly/monthly/yearly digests.

Produces fully self-contained HTML files with inline CSS and server-rendered
SVG bar charts.  Each render function is idempotent: re-running overwrites
both the HTML file and the digests row for that digest_id.

Templates live in pipeline/src/templates/ alongside this module.  The
templates_dir parameter exists primarily to allow tests to supply an
alternative directory; normal pipeline operation uses the bundled templates.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import db

log = structlog.get_logger(__name__)

# Where the bundled templates live when no override is given.
_DEFAULT_TEMPLATES_DIR = Path(__file__).parent / "templates"

# How many papers to show in monthly / yearly rollup pages.
_ROLLUP_TOP_N = 20
# How many clusters to show on rollup pages.
_ROLLUP_TOP_CLUSTERS = 10

# SVG layout constants.
_SVG_ROW_HEIGHT = 28
_SVG_BAR_MAX_WIDTH = 500
_SVG_LABEL_X = 8
_SVG_BAR_X = 200
_SVG_VALUE_PAD = 6
_SVG_PADDING_TOP = 10
_SVG_PADDING_BOTTOM = 10


@dataclasses.dataclass(frozen=True)
class RenderResult:
    """Outcome of one render call."""

    digest_id: str
    html_path: Path
    paper_count: int


# ── Jinja2 custom filters ─────────────────────────────────────────────────────


def _filter_format_score(score: float | None) -> str:
    """Format a relevance_score as one decimal place, e.g. '7.4'."""
    if score is None:
        return "—"
    return f"{float(score):.1f}"


def _filter_pct_delta(delta: float | None) -> str:
    """Format a percentage delta as '+12%', '-8%', or '—' for None."""
    if delta is None:
        return "—"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}%"


# ── Jinja2 environment factory ────────────────────────────────────────────────


def _make_env(templates_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["format_score"] = _filter_format_score
    env.filters["pct_delta"] = _filter_pct_delta
    return env


# ── SVG bar chart ─────────────────────────────────────────────────────────────


def _svg_bar_chart(
    rows: list[tuple[str, int]],
    *,
    max_bar_width: int = _SVG_BAR_MAX_WIDTH,
    row_height: int = _SVG_ROW_HEIGHT,
    bar_x: int = _SVG_BAR_X,
    label_x: int = _SVG_LABEL_X,
    value_pad: int = _SVG_VALUE_PAD,
    pad_top: int = _SVG_PADDING_TOP,
    pad_bottom: int = _SVG_PADDING_BOTTOM,
) -> str:
    """Return a self-contained SVG string: horizontal bar chart.

    ``rows`` is a list of ``(label, count)`` pairs ordered by desired display
    order (largest first is conventional).  Returns an empty-string SVG when
    the list is empty.
    """
    if not rows:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"></svg>'

    max_count = max(count for _, count in rows)
    if max_count == 0:
        max_count = 1  # avoid division by zero

    total_width = bar_x + max_bar_width + 60  # room for value labels
    total_height = pad_top + len(rows) * row_height + pad_bottom

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{total_width}" height="{total_height}" '
        f'role="img" aria-label="Cluster paper counts">'
    ]

    for i, (label, count) in enumerate(rows):
        y_center = pad_top + i * row_height + row_height // 2
        bar_width = int(count / max_count * max_bar_width)
        # Clip label to avoid SVG overflow into bar area.
        clipped = label[:28] + "…" if len(label) > 28 else label

        parts.append(
            f'  <text x="{label_x}" y="{y_center + 5}" '
            f'font-family="Georgia, serif" font-size="12" fill="#333">'
            f"{clipped}</text>"
        )
        parts.append(
            f'  <rect x="{bar_x}" y="{y_center - 9}" '
            f'width="{bar_width}" height="17" '
            f'fill="#2563eb" rx="2" />'
        )
        value_x = bar_x + bar_width + value_pad
        parts.append(
            f'  <text x="{value_x}" y="{y_center + 5}" '
            f'font-family="ui-monospace, monospace" font-size="11" fill="#555">'
            f"{count}</text>"
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ── DB query helpers ──────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _authors_display(authors_json: str, *, max_authors: int = 4) -> str:
    """Decode authors JSON array and return a short display string."""
    try:
        names: list[str] = json.loads(authors_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(authors_json)
    if len(names) <= max_authors:
        return ", ".join(names)
    shown = ", ".join(names[:max_authors])
    rest = len(names) - max_authors
    return f"{shown} +{rest} more"


def _load_weekly_data(
    conn: sqlite3.Connection,
    week: str,
) -> tuple[list[dict[str, Any]], int]:
    """Load clusters + papers for a single week.

    Returns ``(clusters, total_paper_count)`` where each cluster dict has:
      label, description, paper_count, cluster_id, trend (or None), papers [].
    Each paper dict has paper fields + optional summary sub-dict.
    """
    # Load clusters
    cluster_rows = conn.execute(
        """
        SELECT cluster_id, label, description, subtopics, paper_count
          FROM clusters
         WHERE week = ?
         ORDER BY paper_count DESC
        """,
        (week,),
    ).fetchall()

    # Load cluster trends
    trend_rows = conn.execute(
        """
        SELECT cluster_id, delta_vs_w1, delta_vs_w4, delta_vs_w12, delta_vs_w52,
               is_new, velocity_class
          FROM cluster_trends
         WHERE week = ?
        """,
        (week,),
    ).fetchall()
    trends_by_id: dict[int, sqlite3.Row] = {
        int(r["cluster_id"]): r for r in trend_rows
    }

    # Load paper_clusters
    pc_rows = conn.execute(
        """
        SELECT pc.arxiv_id, pc.cluster_id,
               p.title, p.authors, p.abstract, p.primary_category,
               p.relevance_score, p.url_abs, p.url_pdf
          FROM paper_clusters pc
          JOIN papers p ON p.arxiv_id = pc.arxiv_id
         WHERE pc.week = ?
        """,
        (week,),
    ).fetchall()

    # Load summaries for all papers in this week
    arxiv_ids_in_week = [str(r["arxiv_id"]) for r in pc_rows]
    summaries_by_id: dict[str, dict[str, Any]] = {}
    if arxiv_ids_in_week:
        placeholders = ",".join("?" * len(arxiv_ids_in_week))
        summ_rows = conn.execute(
            f"""
            SELECT arxiv_id, problem, method, key_result, why_it_matters,
                   cross_domain_hooks, novelty_signal, tags
              FROM summaries
             WHERE arxiv_id IN ({placeholders})
            """,
            arxiv_ids_in_week,
        ).fetchall()
        for s in summ_rows:
            try:
                hooks = json.loads(str(s["cross_domain_hooks"]))
            except (json.JSONDecodeError, TypeError):
                hooks = []
            try:
                tags = json.loads(str(s["tags"]))
            except (json.JSONDecodeError, TypeError):
                tags = []
            summaries_by_id[str(s["arxiv_id"])] = {
                "problem": s["problem"],
                "method": s["method"],
                "key_result": s["key_result"],
                "why_it_matters": s["why_it_matters"],
                "cross_domain_hooks": hooks,
                "novelty_signal": s["novelty_signal"],
                "tags": tags,
            }

    # Group papers by cluster_id, sorted by relevance_score desc
    papers_by_cluster: dict[int, list[dict[str, Any]]] = {}
    for r in pc_rows:
        cid = int(r["cluster_id"])
        papers_by_cluster.setdefault(cid, []).append(
            {
                "arxiv_id": str(r["arxiv_id"]),
                "title": str(r["title"]),
                "authors_display": _authors_display(str(r["authors"])),
                "primary_category": str(r["primary_category"]),
                "relevance_score": float(r["relevance_score"])
                if r["relevance_score"] is not None
                else 0.0,
                "url_abs": str(r["url_abs"]),
                "url_pdf": str(r["url_pdf"]),
                "summary": summaries_by_id.get(str(r["arxiv_id"])),
            }
        )

    # Sort each cluster's papers by relevance_score desc
    for papers in papers_by_cluster.values():
        papers.sort(key=lambda p: p["relevance_score"], reverse=True)

    # Assemble clusters
    clusters: list[dict[str, Any]] = []
    for cr in cluster_rows:
        cid = int(cr["cluster_id"])
        trend_row = trends_by_id.get(cid)
        trend: dict[str, Any] | None = None
        if trend_row is not None:
            trend = {
                "delta_vs_w1": trend_row["delta_vs_w1"],
                "delta_vs_w4": trend_row["delta_vs_w4"],
                "delta_vs_w12": trend_row["delta_vs_w12"],
                "delta_vs_w52": trend_row["delta_vs_w52"],
                "is_new": bool(trend_row["is_new"]),
                "velocity_class": trend_row["velocity_class"],
            }
        clusters.append(
            {
                "cluster_id": cid,
                "label": str(cr["label"]),
                "description": str(cr["description"]),
                "paper_count": int(cr["paper_count"]),
                "subtopics": json.loads(str(cr["subtopics"]))
                if cr["subtopics"]
                else [],
                "trend": trend,
                "papers": papers_by_cluster.get(cid, []),
            }
        )

    total_papers = sum(len(c["papers"]) for c in clusters)
    return clusters, total_papers


def _load_rollup_data(
    conn: sqlite3.Connection,
    weeks: list[str],
    *,
    top_clusters_n: int = _ROLLUP_TOP_CLUSTERS,
    top_papers_n: int = _ROLLUP_TOP_N,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Aggregate clusters and papers across a list of weeks.

    Returns ``(top_clusters, top_papers, total_paper_count)``.
    Clusters are merged by label (case-sensitive) and their paper_counts summed.
    Papers are pulled from paper_clusters for all weeks, de-duplicated by
    arxiv_id, sorted by relevance_score desc, and the top ``top_papers_n``
    returned.
    """
    if not weeks:
        return [], [], 0

    placeholders = ",".join("?" * len(weeks))

    # Aggregate clusters across weeks by label
    cluster_rows = conn.execute(
        f"""
        SELECT label, description, SUM(paper_count) AS total_count
          FROM clusters
         WHERE week IN ({placeholders})
         GROUP BY label
         ORDER BY total_count DESC
         LIMIT ?
        """,
        [*weeks, top_clusters_n],
    ).fetchall()

    top_clusters: list[dict[str, Any]] = [
        {
            "label": str(r["label"]),
            "description": str(r["description"]),
            "paper_count": int(r["total_count"]),
        }
        for r in cluster_rows
    ]

    # Aggregate papers across weeks; de-duplicate by arxiv_id (keep highest score)
    paper_rows = conn.execute(
        f"""
        SELECT DISTINCT pc.arxiv_id, p.title, p.authors,
               p.relevance_score, p.url_abs, p.url_pdf
          FROM paper_clusters pc
          JOIN papers p ON p.arxiv_id = pc.arxiv_id
         WHERE pc.week IN ({placeholders})
         ORDER BY p.relevance_score DESC
         LIMIT ?
        """,
        [*weeks, top_papers_n],
    ).fetchall()

    arxiv_ids = [str(r["arxiv_id"]) for r in paper_rows]
    summaries_by_id: dict[str, dict[str, Any]] = {}
    if arxiv_ids:
        sp = ",".join("?" * len(arxiv_ids))
        summ_rows = conn.execute(
            f"""
            SELECT arxiv_id, why_it_matters, novelty_signal, tags
              FROM summaries
             WHERE arxiv_id IN ({sp})
            """,
            arxiv_ids,
        ).fetchall()
        for s in summ_rows:
            summaries_by_id[str(s["arxiv_id"])] = {
                "why_it_matters": s["why_it_matters"],
                "novelty_signal": s["novelty_signal"],
                "tags": json.loads(str(s["tags"])) if s["tags"] else [],
            }

    top_papers: list[dict[str, Any]] = []
    for r in paper_rows:
        aid = str(r["arxiv_id"])
        top_papers.append(
            {
                "arxiv_id": aid,
                "title": str(r["title"]),
                "authors_display": _authors_display(str(r["authors"])),
                "relevance_score": float(r["relevance_score"])
                if r["relevance_score"] is not None
                else 0.0,
                "url_abs": str(r["url_abs"]),
                "url_pdf": str(r["url_pdf"]),
                "summary": summaries_by_id.get(aid),
            }
        )

    # Total unique papers across all weeks
    count_row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT pc.arxiv_id) AS cnt
          FROM paper_clusters pc
         WHERE pc.week IN ({placeholders})
        """,
        weeks,
    ).fetchone()
    total_count = int(count_row["cnt"]) if count_row else 0

    return top_clusters, top_papers, total_count


def _upsert_digest(
    conn: sqlite3.Connection,
    *,
    digest_id: str,
    kind: str,
    generated_at: str,
    html_path: Path,
    paper_count: int,
    trend_narrative: dict[str, Any] | None,
) -> None:
    trend_json: str | None = json.dumps(trend_narrative) if trend_narrative else None
    with db.transaction(conn):
        conn.execute(
            """
            INSERT OR REPLACE INTO digests
              (digest_id, kind, generated_at, html_path, paper_count,
               trend_narrative, sent_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                digest_id,
                kind,
                generated_at,
                str(html_path),
                paper_count,
                trend_json,
            ),
        )


def _resolve_templates_dir(templates_dir: Path | str | None) -> Path:
    if templates_dir is None:
        return _DEFAULT_TEMPLATES_DIR
    return Path(templates_dir)


def _resolve_output_dir(output_dir: Path | str) -> Path:
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Public render functions ───────────────────────────────────────────────────


def render_weekly(
    conn: sqlite3.Connection,
    *,
    week: str,
    output_dir: Path | str,
    trend_narrative: dict[str, Any] | None = None,
    templates_dir: Path | str | None = None,
) -> RenderResult:
    """Render a weekly digest HTML file and upsert the digests row.

    Parameters
    ----------
    conn:             Open SQLite connection with schema initialised.
    week:             ISO week label, e.g. '2026-W20'.
    output_dir:       Directory for the output HTML file (created if missing).
    trend_narrative:  Optional validated trend_narrative dict to embed in the
                      page headline and persist to digests.trend_narrative.
    templates_dir:    Override for the Jinja2 templates directory.
    """
    out_dir = _resolve_output_dir(output_dir)
    tpl_dir = _resolve_templates_dir(templates_dir)
    env = _make_env(tpl_dir)

    clusters, paper_count = _load_weekly_data(conn, week)

    # Build SVG chart (clusters already sorted by paper_count desc)
    chart_rows = [(c["label"], c["paper_count"]) for c in clusters]
    bar_chart_svg = _svg_bar_chart(chart_rows)

    # Extract headline from trend_narrative if available
    headline: str | None = None
    if trend_narrative:
        headline = trend_narrative.get("headline") or trend_narrative.get(
            "summary"
        )
        if not headline:
            # Fallback: stringify the whole thing isn't useful; stay None
            headline = None

    generated_at = _now_iso()
    ctx: dict[str, Any] = {
        "week": week,
        "paper_count": paper_count,
        "clusters": clusters,
        "bar_chart_svg": bar_chart_svg,
        "headline": headline,
        "generated_at": generated_at,
    }

    template = env.get_template("weekly.html.j2")
    html = template.render(**ctx)

    html_filename = f"digest-{week}.html"
    html_path = out_dir / html_filename
    html_path.write_text(html, encoding="utf-8")

    _upsert_digest(
        conn,
        digest_id=week,
        kind="weekly",
        generated_at=generated_at,
        html_path=html_path,
        paper_count=paper_count,
        trend_narrative=trend_narrative,
    )

    log.info(
        "render.weekly.complete",
        week=week,
        paper_count=paper_count,
        clusters=len(clusters),
        html_path=str(html_path),
    )
    return RenderResult(digest_id=week, html_path=html_path, paper_count=paper_count)


def render_monthly(
    conn: sqlite3.Connection,
    *,
    month: str,
    output_dir: Path | str,
    templates_dir: Path | str | None = None,
) -> RenderResult:
    """Render a monthly rollup digest HTML file and upsert the digests row.

    Parameters
    ----------
    conn:           Open SQLite connection with schema initialised.
    month:          Month label, e.g. '2026-05'.
    output_dir:     Directory for the output HTML file (created if missing).
    templates_dir:  Override for the Jinja2 templates directory.
    """
    out_dir = _resolve_output_dir(output_dir)
    tpl_dir = _resolve_templates_dir(templates_dir)
    env = _make_env(tpl_dir)

    # Find all weeks for this month by inspecting published_at dates of
    # papers in clusters.  A week is included if its clusters table entry
    # has papers published in the requested month.
    week_rows = conn.execute(
        """
        SELECT DISTINCT pc.week
          FROM paper_clusters pc
          JOIN papers p ON p.arxiv_id = pc.arxiv_id
         WHERE strftime('%Y-%m', p.published_at) = ?
         ORDER BY pc.week
        """,
        (month,),
    ).fetchall()
    weeks = [str(r["week"]) for r in week_rows]

    top_clusters, top_papers, paper_count = _load_rollup_data(conn, weeks)

    chart_rows = [(c["label"], c["paper_count"]) for c in top_clusters]
    bar_chart_svg = _svg_bar_chart(chart_rows)

    generated_at = _now_iso()
    ctx: dict[str, Any] = {
        "month": month,
        "weeks": weeks,
        "paper_count": paper_count,
        "top_clusters": top_clusters,
        "top_papers": top_papers,
        "bar_chart_svg": bar_chart_svg,
        "generated_at": generated_at,
    }

    template = env.get_template("monthly.html.j2")
    html = template.render(**ctx)

    html_filename = f"digest-{month}.html"
    html_path = out_dir / html_filename
    html_path.write_text(html, encoding="utf-8")

    _upsert_digest(
        conn,
        digest_id=month,
        kind="monthly",
        generated_at=generated_at,
        html_path=html_path,
        paper_count=paper_count,
        trend_narrative=None,
    )

    log.info(
        "render.monthly.complete",
        month=month,
        weeks_found=len(weeks),
        paper_count=paper_count,
        html_path=str(html_path),
    )
    return RenderResult(digest_id=month, html_path=html_path, paper_count=paper_count)


def render_yearly(
    conn: sqlite3.Connection,
    *,
    year: str,
    output_dir: Path | str,
    templates_dir: Path | str | None = None,
) -> RenderResult:
    """Render a yearly rollup digest HTML file and upsert the digests row.

    Parameters
    ----------
    conn:           Open SQLite connection with schema initialised.
    year:           Year label, e.g. '2026'.
    output_dir:     Directory for the output HTML file (created if missing).
    templates_dir:  Override for the Jinja2 templates directory.
    """
    out_dir = _resolve_output_dir(output_dir)
    tpl_dir = _resolve_templates_dir(templates_dir)
    env = _make_env(tpl_dir)

    week_rows = conn.execute(
        """
        SELECT DISTINCT pc.week
          FROM paper_clusters pc
          JOIN papers p ON p.arxiv_id = pc.arxiv_id
         WHERE strftime('%Y', p.published_at) = ?
         ORDER BY pc.week
        """,
        (year,),
    ).fetchall()
    weeks = [str(r["week"]) for r in week_rows]

    top_clusters, top_papers, paper_count = _load_rollup_data(conn, weeks)

    chart_rows = [(c["label"], c["paper_count"]) for c in top_clusters]
    bar_chart_svg = _svg_bar_chart(chart_rows)

    generated_at = _now_iso()
    ctx: dict[str, Any] = {
        "year": year,
        "weeks": weeks,
        "paper_count": paper_count,
        "top_clusters": top_clusters,
        "top_papers": top_papers,
        "bar_chart_svg": bar_chart_svg,
        "generated_at": generated_at,
    }

    template = env.get_template("yearly.html.j2")
    html = template.render(**ctx)

    html_filename = f"digest-{year}.html"
    html_path = out_dir / html_filename
    html_path.write_text(html, encoding="utf-8")

    _upsert_digest(
        conn,
        digest_id=year,
        kind="yearly",
        generated_at=generated_at,
        html_path=html_path,
        paper_count=paper_count,
        trend_narrative=None,
    )

    log.info(
        "render.yearly.complete",
        year=year,
        weeks_found=len(weeks),
        paper_count=paper_count,
        html_path=str(html_path),
    )
    return RenderResult(digest_id=year, html_path=html_path, paper_count=paper_count)
