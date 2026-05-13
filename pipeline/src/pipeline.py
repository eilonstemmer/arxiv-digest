"""CLI orchestrator for the arxiv-digest pipeline.

Subcommands
-----------
  weekly     Full pipeline: ingest → embed → triage → cluster → summarize
             → trend → render → notify.
  monthly    Render a monthly aggregation digest (no ingest/LLM calls).
  yearly     Render a yearly aggregation digest (no ingest/LLM calls).
  backfill   Wide ingest + embed + triage (no cluster/summarize/render/notify).
             Use to seed a fresh database.
  reprocess  Clear triage + summary state for one paper and re-run those stages.
  health     DB ping, model loadable, prompts present. Exit 0 if healthy.

All configuration is supplied via environment variables; see the module-level
constants.  The DB_PATH / CONFIG_DIR / PROMPTS_DIR / DIGESTS_OUTPUT paths are
resolved once at import time so the health command can inspect them directly;
tests override them via monkeypatch on the module attributes.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import structlog

from . import cluster, db, embed, ingest, notify, render, summarize, trend, triage

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level path constants (computed from env at import time).
# Tests override these via monkeypatch on the module attributes.
# ---------------------------------------------------------------------------

DB_PATH: Path = Path(os.environ.get("DB_PATH", "/data/digest.db"))
WEBAPP_URL: str = os.environ.get("WEBAPP_URL", "http://webapp:8080")
PROMPTS_DIR: Path = Path(os.environ.get("PROMPTS_DIR", "/prompts"))
CONFIG_DIR: Path = Path(os.environ.get("CONFIG_DIR", "/config"))
DIGESTS_OUTPUT: Path = Path(os.environ.get("DIGESTS_OUTPUT", "/data/digests"))


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# pipeline_runs helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _start_run(conn: sqlite3.Connection, kind: str) -> str:
    run_id = (
        f"{kind}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
    )
    conn.execute(
        "INSERT INTO pipeline_runs(run_id, kind, started_at, status) VALUES (?, ?, ?, 'running')",
        (run_id, kind, _now_iso()),
    )
    return run_id


def _finish_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    status: str,
    papers_ingested: int | None = None,
    papers_prefiltered: int | None = None,
    papers_triaged: int | None = None,
    papers_summarized: int | None = None,
    cost_usd: float | None = None,
    error: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE pipeline_runs
           SET finished_at = ?, status = ?, papers_ingested = ?,
               papers_prefiltered = ?, papers_triaged = ?,
               papers_summarized = ?, cost_usd = ?, error = ?
         WHERE run_id = ?
        """,
        (
            _now_iso(),
            status,
            papers_ingested,
            papers_prefiltered,
            papers_triaged,
            papers_summarized,
            cost_usd,
            error,
            run_id,
        ),
    )


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_weekly(args: argparse.Namespace, conn: sqlite3.Connection) -> int:
    """Full weekly pipeline: ingest → embed → triage → cluster → summarize
    → trend → render → notify."""
    run_id = _start_run(conn, kind="weekly")
    try:
        today = datetime.now(UTC).date()
        iso = today.isocalendar()
        week = f"{iso.year:04d}-W{iso.week:02d}"
        since = (today - timedelta(days=7)).isoformat() + "T00:00:00Z"

        # 1. Ingest
        categories = ingest.load_category_codes(CONFIG_DIR / "categories.yaml")
        ingest_result = ingest.fetch_papers(conn, categories, max_results=2000)

        # 2. Embed
        embed.embed_papers_batch(conn)

        # 3. Triage (prefilter + Haiku)
        triage_result = triage.triage(
            conn,
            profile_path=CONFIG_DIR / "profile.md",
            prompts_dir=PROMPTS_DIR,
        )

        # 4. Cluster
        cluster_result = cluster.cluster(
            conn, week=week, since=since, prompts_dir=PROMPTS_DIR
        )

        # 5. Summarize (Sonnet Batch + cross-domain hooks)
        summarize_result = summarize.summarize(
            conn,
            profile_path=CONFIG_DIR / "profile.md",
            prompts_dir=PROMPTS_DIR,
        )

        # 6. Trend
        trend_result = trend.compute_trends(
            conn,
            week=week,
            profile_path=CONFIG_DIR / "profile.md",
            prompts_dir=PROMPTS_DIR,
        )

        # 7. Render
        render_result = render.render_weekly(
            conn,
            week=week,
            output_dir=DIGESTS_OUTPUT,
            trend_narrative=trend_result.trend_payload,
        )

        # 8. Notify (only when WEBHOOK_SECRET is set)
        webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
        if webhook_secret:
            notify.notify_for_digest(
                conn,
                digest_id=render_result.digest_id,
                webapp_url=WEBAPP_URL,
                webhook_secret=webhook_secret,
            )

        cost = (
            triage_result.cost_usd
            + cluster_result.cost_usd
            + summarize_result.cost_usd
            + trend_result.cost_usd
        )

        _finish_run(
            conn,
            run_id,
            status="success",
            papers_ingested=ingest_result.new,
            papers_prefiltered=triage_result.papers_prefiltered,
            papers_triaged=triage_result.papers_haiku_requested,
            papers_summarized=summarize_result.papers_summarized_succeeded,
            cost_usd=cost,
        )
        log.info(
            "pipeline.weekly.complete",
            week=week,
            papers_ingested=ingest_result.new,
            cost_usd=round(cost, 4),
        )
        return 0
    except Exception as exc:
        log.exception("pipeline.weekly.failed")
        _finish_run(conn, run_id, status="failure", error=str(exc)[:1000])
        return 1


def cmd_monthly(args: argparse.Namespace, conn: sqlite3.Connection) -> int:
    """Render the monthly aggregation digest."""
    run_id = _start_run(conn, kind="monthly")
    try:
        result = render.render_monthly(
            conn, month=args.month, output_dir=DIGESTS_OUTPUT
        )
        _finish_run(conn, run_id, status="success")
        webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
        if webhook_secret:
            notify.notify_for_digest(
                conn,
                digest_id=result.digest_id,
                webapp_url=WEBAPP_URL,
                webhook_secret=webhook_secret,
            )
        log.info("pipeline.monthly.complete", month=args.month, digest_id=result.digest_id)
        return 0
    except Exception as exc:
        log.exception("pipeline.monthly.failed")
        _finish_run(conn, run_id, status="failure", error=str(exc)[:1000])
        return 1


def cmd_yearly(args: argparse.Namespace, conn: sqlite3.Connection) -> int:
    """Render the yearly aggregation digest."""
    run_id = _start_run(conn, kind="yearly")
    try:
        result = render.render_yearly(
            conn, year=args.year, output_dir=DIGESTS_OUTPUT
        )
        _finish_run(conn, run_id, status="success")
        webhook_secret = os.environ.get("WEBHOOK_SECRET", "")
        if webhook_secret:
            notify.notify_for_digest(
                conn,
                digest_id=result.digest_id,
                webapp_url=WEBAPP_URL,
                webhook_secret=webhook_secret,
            )
        log.info("pipeline.yearly.complete", year=args.year, digest_id=result.digest_id)
        return 0
    except Exception as exc:
        log.exception("pipeline.yearly.failed")
        _finish_run(conn, run_id, status="failure", error=str(exc)[:1000])
        return 1


def cmd_backfill(args: argparse.Namespace, conn: sqlite3.Connection) -> int:
    """Wide ingest + embed + triage for seeding a fresh database."""
    run_id = _start_run(conn, kind="backfill")
    try:
        categories = ingest.load_category_codes(CONFIG_DIR / "categories.yaml")
        ingest_result = ingest.fetch_papers(
            conn, categories, max_results=args.max_results
        )
        embed.embed_papers_batch(conn)
        triage_result = triage.triage(
            conn,
            profile_path=CONFIG_DIR / "profile.md",
            prompts_dir=PROMPTS_DIR,
        )
        _finish_run(
            conn,
            run_id,
            status="success",
            papers_ingested=ingest_result.new,
            papers_prefiltered=triage_result.papers_prefiltered,
            papers_triaged=triage_result.papers_haiku_requested,
            cost_usd=triage_result.cost_usd,
        )
        log.info(
            "pipeline.backfill.complete",
            papers_ingested=ingest_result.new,
            papers_triaged=triage_result.papers_haiku_requested,
        )
        return 0
    except Exception as exc:
        log.exception("pipeline.backfill.failed")
        _finish_run(conn, run_id, status="failure", error=str(exc)[:1000])
        return 1


def cmd_reprocess(args: argparse.Namespace, conn: sqlite3.Connection) -> int:
    """Clear triage + summary state for one paper and re-run those stages."""
    with db.transaction(conn):
        conn.execute(
            "UPDATE papers SET relevance_score = NULL, relevance_reason = NULL, "
            "triaged_at = NULL, summarized = 0 WHERE arxiv_id = ?",
            (args.arxiv_id,),
        )
        conn.execute("DELETE FROM summaries WHERE arxiv_id = ?", (args.arxiv_id,))

    triage.triage(
        conn,
        profile_path=CONFIG_DIR / "profile.md",
        prompts_dir=PROMPTS_DIR,
    )
    summarize.summarize(
        conn,
        profile_path=CONFIG_DIR / "profile.md",
        prompts_dir=PROMPTS_DIR,
    )
    log.info("pipeline.reprocess.complete", arxiv_id=args.arxiv_id)
    return 0


def cmd_health(args: argparse.Namespace, conn: sqlite3.Connection) -> int:
    """DB ping, model loadable, prompts present. Exits 0 if all healthy."""
    checks: list[tuple[str, bool, str | None]] = []

    # DB ping
    try:
        conn.execute("SELECT 1").fetchone()
        checks.append(("db", True, None))
    except Exception as exc:
        checks.append(("db", False, str(exc)))

    # Prompts present
    for name in ("triage", "summarize", "cluster_label", "trend_narrative"):
        md = PROMPTS_DIR / f"{name}.md"
        sj = PROMPTS_DIR / f"{name}.schema.json"
        if md.exists() and sj.exists():
            checks.append((f"prompt:{name}", True, None))
        else:
            missing = md if not md.exists() else sj
            checks.append((f"prompt:{name}", False, f"missing {missing}"))

    # Config dir
    if (CONFIG_DIR / "profile.md").exists() and (CONFIG_DIR / "categories.yaml").exists():
        checks.append(("config", True, None))
    else:
        checks.append(("config", False, "missing profile.md or categories.yaml"))

    # Model loadable
    try:
        embed.get_model()
        checks.append(("model", True, None))
    except Exception as exc:
        checks.append(("model", False, str(exc)))

    all_ok = all(ok for _, ok, _ in checks)
    for name, ok, err in checks:
        status = "OK" if ok else "FAIL"
        msg = f"  {status}  {name}"
        if err:
            msg += f"  ({err})"
        print(msg)
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="src.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("weekly", help="full weekly pipeline run").set_defaults(
        func=cmd_weekly
    )
    sub.add_parser(
        "health", help="run health checks; nonzero exit if anything's broken"
    ).set_defaults(func=cmd_health)

    m = sub.add_parser("monthly", help="render the monthly aggregation")
    m.add_argument("month", help="month in YYYY-MM format (e.g., '2026-05')")
    m.set_defaults(func=cmd_monthly)

    y = sub.add_parser("yearly", help="render the yearly aggregation")
    y.add_argument("year", help="year in YYYY format (e.g., '2026')")
    y.set_defaults(func=cmd_yearly)

    b = sub.add_parser("backfill", help="ingest + embed + triage with a wider window")
    b.add_argument("--max-results", type=int, default=5000)
    b.add_argument(
        "--days",
        type=int,
        default=30,
        help="(informational; arXiv API has no date range filter)",
    )
    b.set_defaults(func=cmd_backfill)

    r = sub.add_parser("reprocess", help="re-triage and re-summarize one paper")
    r.add_argument("arxiv_id")
    r.set_defaults(func=cmd_reprocess)

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    conn = db.connect(DB_PATH)
    db.init_schema(conn)
    try:
        return int(args.func(args, conn))
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
