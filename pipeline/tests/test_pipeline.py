"""Tests for src.pipeline: CLI orchestration of the full arxiv-digest pipeline.

Strategy: every pipeline stage function is monkeypatched to return a canned
result object.  Tests verify:
  - stages are called in the correct order
  - pipeline_runs rows are created and updated correctly
  - cost_usd is summed across triage + cluster + summarize + trend
  - paper counters are forwarded from stage results to pipeline_runs
  - exceptions propagate to a failure row
  - backfill calls only ingest + embed + triage
  - reprocess clears paper state and re-runs triage + summarize
  - monthly / yearly dispatch to the right render functions
  - health returns 0 / 1 depending on system state
  - notify is called / skipped based on WEBHOOK_SECRET env var

Design decision: PROMPTS_DIR and CONFIG_DIR are module attributes computed at
import time from env vars.  Tests override them directly on the pipeline module
(``monkeypatch.setattr(pipeline, "PROMPTS_DIR", ...)``) so each test can
control which paths are "present" without touching the real filesystem.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src import db, pipeline
from src.cluster import ClusterResult
from src.ingest import IngestResult
from src.notify import NotifyResult
from src.render import RenderResult
from src.summarize import SummarizeResult
from src.trend import TrendResult
from src.triage import TriageResult

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def _patch_db_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the module-level DB_PATH so main() opens a tmp DB."""
    monkeypatch.setattr(pipeline, "DB_PATH", tmp_path / "pipeline_test.db")


@pytest.fixture(autouse=True)
def _no_webhook_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default, no WEBHOOK_SECRET so notify is skipped."""
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)


# ── Canned stage results ────────────────────────────────────────────────────

_INGEST = IngestResult(fetched=200, new=150, duplicates=50, errors=0)
_TRIAGE = TriageResult(
    papers_eligible=120,
    papers_prefiltered=48,
    papers_haiku_requested=72,
    papers_haiku_succeeded=70,
    papers_haiku_failed=2,
    cost_usd=0.10,
)
_CLUSTER = ClusterResult(
    week="2026-W20",
    papers_clustered=60,
    noise_count=5,
    clusters_formed=8,
    label_succeeded=8,
    label_failed=0,
    cost_usd=0.20,
)
_SUMMARIZE = SummarizeResult(
    papers_eligible=70,
    papers_summarized_succeeded=65,
    papers_summarized_failed=5,
    total_cross_domain_hooks=10,
    cost_usd=0.50,
)
_TREND = TrendResult(
    week="2026-W20",
    clusters_processed=8,
    trend_payload={"headline": "More LLM papers"},
    cost_usd=0.05,
)
_RENDER_WEEKLY = RenderResult(
    digest_id="2026-W20",
    html_path=Path("/data/digests/digest-2026-W20.html"),
    paper_count=65,
)
_RENDER_MONTHLY = RenderResult(
    digest_id="2026-05",
    html_path=Path("/data/digests/digest-2026-05.html"),
    paper_count=200,
)
_RENDER_YEARLY = RenderResult(
    digest_id="2026",
    html_path=Path("/data/digests/digest-2026.html"),
    paper_count=2000,
)
_NOTIFY = NotifyResult(
    digest_id="2026-W20",
    webhook_sent=True,
    webhook_error=None,
    email_attempted=False,
    email_sent=False,
    email_error=None,
)

EXPECTED_COST = _TRIAGE.cost_usd + _CLUSTER.cost_usd + _SUMMARIZE.cost_usd + _TREND.cost_usd


def _patch_all_stages(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Monkeypatch every stage function to return canned results.
    Returns a shared call-order list that each fake appends its name to."""
    order: list[str] = []

    def fake_load_cats(path: Any) -> list[str]:
        return ["cs.AI", "cs.LG"]

    def fake_fetch(conn: Any, cats: Any, **kw: Any) -> IngestResult:
        order.append("ingest")
        return _INGEST

    def fake_embed(conn: Any, **kw: Any) -> int:
        order.append("embed")
        return 150

    def fake_triage(conn: Any, **kw: Any) -> TriageResult:
        order.append("triage")
        return _TRIAGE

    def fake_cluster(conn: Any, **kw: Any) -> ClusterResult:
        order.append("cluster")
        return _CLUSTER

    def fake_summarize(conn: Any, **kw: Any) -> SummarizeResult:
        order.append("summarize")
        return _SUMMARIZE

    def fake_trend(conn: Any, **kw: Any) -> TrendResult:
        order.append("trend")
        return _TREND

    def fake_render_weekly(conn: Any, **kw: Any) -> RenderResult:
        order.append("render_weekly")
        return _RENDER_WEEKLY

    def fake_notify(conn: Any, **kw: Any) -> NotifyResult:
        order.append("notify")
        return _NOTIFY

    monkeypatch.setattr(pipeline.ingest, "load_category_codes", fake_load_cats)
    monkeypatch.setattr(pipeline.ingest, "fetch_papers", fake_fetch)
    monkeypatch.setattr(pipeline.embed, "embed_papers_batch", fake_embed)
    monkeypatch.setattr(pipeline.triage, "triage", fake_triage)
    monkeypatch.setattr(pipeline.cluster, "cluster", fake_cluster)
    monkeypatch.setattr(pipeline.summarize, "summarize", fake_summarize)
    monkeypatch.setattr(pipeline.trend, "compute_trends", fake_trend)
    monkeypatch.setattr(pipeline.render, "render_weekly", fake_render_weekly)
    monkeypatch.setattr(pipeline.notify, "notify_for_digest", fake_notify)

    return order


# ---------------------------------------------------------------------------
# weekly: stage ordering
# ---------------------------------------------------------------------------


def test_weekly_stage_order(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """weekly invokes stages in the correct order."""
    order = _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    rc = pipeline.cmd_weekly(args, conn)
    assert rc == 0
    assert order == ["ingest", "embed", "triage", "cluster", "summarize", "trend", "render_weekly"]


def test_weekly_notify_not_called_without_secret(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """notify is NOT called when WEBHOOK_SECRET env var is absent."""
    order = _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)
    assert "notify" not in order


def test_weekly_notify_called_with_secret(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """notify IS called when WEBHOOK_SECRET is set."""
    monkeypatch.setenv("WEBHOOK_SECRET", "supersecret")
    order = _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)
    assert "notify" in order


# ---------------------------------------------------------------------------
# weekly: pipeline_runs row creation + update
# ---------------------------------------------------------------------------


def test_weekly_creates_running_row(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pipeline_runs row with status='running' exists while the run is in flight."""
    # Capture the run_id immediately after _start_run by intercepting cluster
    captured: dict[str, Any] = {}

    def spy_cluster(c: Any, **kw: Any) -> ClusterResult:
        row = c.execute(
            "SELECT status FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        captured["status_mid"] = row["status"] if row else None
        return _CLUSTER

    _patch_all_stages(monkeypatch)
    monkeypatch.setattr(pipeline.cluster, "cluster", spy_cluster)

    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)

    assert captured["status_mid"] == "running"


def test_weekly_updates_run_to_success(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After a successful run, pipeline_runs.status == 'success'."""
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    rc = pipeline.cmd_weekly(args, conn)

    assert rc == 0
    row = conn.execute(
        "SELECT status, finished_at FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["status"] == "success"
    assert row["finished_at"] is not None


# ---------------------------------------------------------------------------
# weekly: pipeline_runs counter columns
# ---------------------------------------------------------------------------


def test_weekly_cost_usd_is_summed(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cost_usd == triage + cluster + summarize + trend costs."""
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)

    row = conn.execute(
        "SELECT cost_usd FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert abs(row["cost_usd"] - EXPECTED_COST) < 1e-9


def test_weekly_papers_ingested_from_ingest_result(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)

    row = conn.execute(
        "SELECT papers_ingested FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["papers_ingested"] == _INGEST.new


def test_weekly_papers_prefiltered_from_triage_result(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)

    row = conn.execute(
        "SELECT papers_prefiltered FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["papers_prefiltered"] == _TRIAGE.papers_prefiltered


def test_weekly_papers_triaged_from_triage_result(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)

    row = conn.execute(
        "SELECT papers_triaged FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["papers_triaged"] == _TRIAGE.papers_haiku_requested


def test_weekly_papers_summarized_from_summarize_result(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["weekly"])
    pipeline.cmd_weekly(args, conn)

    row = conn.execute(
        "SELECT papers_summarized FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["papers_summarized"] == _SUMMARIZE.papers_summarized_succeeded


# ---------------------------------------------------------------------------
# weekly: exception → failure row
# ---------------------------------------------------------------------------


def test_weekly_exception_marks_failure(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception in any stage marks the run as failure with an error message."""
    boom_msg = "synthetic boom"

    def fake_fetch(conn: Any, cats: Any, **kw: Any) -> IngestResult:
        raise RuntimeError(boom_msg)

    monkeypatch.setattr(pipeline.ingest, "load_category_codes", lambda p: ["cs.AI"])
    monkeypatch.setattr(pipeline.ingest, "fetch_papers", fake_fetch)

    args = pipeline.build_parser().parse_args(["weekly"])
    rc = pipeline.cmd_weekly(args, conn)

    assert rc == 1
    row = conn.execute(
        "SELECT status, error FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["status"] == "failure"
    assert boom_msg in row["error"]


# ---------------------------------------------------------------------------
# backfill: only ingest + embed + triage
# ---------------------------------------------------------------------------


def test_backfill_calls_only_ingest_embed_triage(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["backfill"])
    rc = pipeline.cmd_backfill(args, conn)

    assert rc == 0
    assert order == ["ingest", "embed", "triage"]


def test_backfill_does_not_call_cluster_summarize_render_notify(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["backfill"])
    pipeline.cmd_backfill(args, conn)

    for stage in ("cluster", "summarize", "trend", "render_weekly", "notify"):
        assert stage not in order


def test_backfill_pipeline_run_success(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_all_stages(monkeypatch)
    args = pipeline.build_parser().parse_args(["backfill"])
    pipeline.cmd_backfill(args, conn)

    row = conn.execute(
        "SELECT kind, status, papers_ingested, papers_triaged, cost_usd "
        "FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["kind"] == "backfill"
    assert row["status"] == "success"
    assert row["papers_ingested"] == _INGEST.new
    assert row["papers_triaged"] == _TRIAGE.papers_haiku_requested
    assert abs(row["cost_usd"] - _TRIAGE.cost_usd) < 1e-9


def test_backfill_default_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """--max-results defaults to 5000."""
    args = pipeline.build_parser().parse_args(["backfill"])
    assert args.max_results == 5000


def test_backfill_custom_max_results(monkeypatch: pytest.MonkeyPatch) -> None:
    args = pipeline.build_parser().parse_args(["backfill", "--max-results", "10000"])
    assert args.max_results == 10000


def test_backfill_exception_marks_failure(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_fetch(conn: Any, cats: Any, **kw: Any) -> IngestResult:
        raise ValueError("backfill explosion")

    monkeypatch.setattr(pipeline.ingest, "load_category_codes", lambda p: ["cs.AI"])
    monkeypatch.setattr(pipeline.ingest, "fetch_papers", fake_fetch)

    args = pipeline.build_parser().parse_args(["backfill"])
    rc = pipeline.cmd_backfill(args, conn)

    assert rc == 1
    row = conn.execute(
        "SELECT status, error FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["status"] == "failure"
    assert "backfill explosion" in row["error"]


# ---------------------------------------------------------------------------
# reprocess
# ---------------------------------------------------------------------------


def _insert_paper(conn: sqlite3.Connection, arxiv_id: str = "2401.00001") -> None:
    """Insert a minimal paper row with triage + summary state to clear."""
    conn.execute(
        """INSERT INTO papers(arxiv_id, title, authors, abstract,
               primary_category, categories, published_at, fetched_at,
               url_abs, url_pdf,
               relevance_score, relevance_reason, triaged_at, summarized)
           VALUES (?, 'T', '[]', 'A', 'cs.AI', '[]', '2026-01-01T00:00:00Z',
                   '2026-01-01T00:00:00Z', 'http://x', 'http://x',
                   8.5, 'relevant', '2026-01-01T00:00:01Z', 1)""",
        (arxiv_id,),
    )
    conn.execute(
        """INSERT INTO summaries(arxiv_id, problem, method, key_result,
               why_it_matters, cross_domain_hooks, novelty_signal, tags,
               generated_at, model)
           VALUES (?, 'P', 'M', 'KR', 'WIM', '[]', 'notable', '[]',
                   '2026-01-01T00:00:02Z', 'claude-test')""",
        (arxiv_id,),
    )


def test_reprocess_clears_triage_and_summary_state(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reprocess clears relevant paper columns and deletes the summary row."""
    arxiv_id = "2401.00001"
    _insert_paper(conn, arxiv_id)

    triage_called: list[bool] = []
    summarize_called: list[bool] = []

    def fake_triage(c: Any, **kw: Any) -> TriageResult:
        triage_called.append(True)
        return _TRIAGE

    def fake_summarize(c: Any, **kw: Any) -> SummarizeResult:
        summarize_called.append(True)
        return _SUMMARIZE

    monkeypatch.setattr(pipeline.triage, "triage", fake_triage)
    monkeypatch.setattr(pipeline.summarize, "summarize", fake_summarize)

    args = pipeline.build_parser().parse_args(["reprocess", arxiv_id])
    rc = pipeline.cmd_reprocess(args, conn)
    assert rc == 0

    # Paper state should be cleared
    row = conn.execute(
        "SELECT relevance_score, relevance_reason, triaged_at, summarized "
        "FROM papers WHERE arxiv_id = ?",
        (arxiv_id,),
    ).fetchone()
    assert row["relevance_score"] is None
    assert row["relevance_reason"] is None
    assert row["triaged_at"] is None
    assert row["summarized"] == 0

    # Summary row deleted
    summary_row = conn.execute(
        "SELECT arxiv_id FROM summaries WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    assert summary_row is None

    # Both stages were called
    assert triage_called
    assert summarize_called


def test_reprocess_calls_triage_then_summarize(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reprocess calls triage before summarize."""
    arxiv_id = "2401.00002"
    _insert_paper(conn, arxiv_id)

    order: list[str] = []

    monkeypatch.setattr(
        pipeline.triage, "triage", lambda c, **kw: (order.append("triage") or _TRIAGE)
    )
    monkeypatch.setattr(
        pipeline.summarize,
        "summarize",
        lambda c, **kw: (order.append("summarize") or _SUMMARIZE),
    )

    args = pipeline.build_parser().parse_args(["reprocess", arxiv_id])
    pipeline.cmd_reprocess(args, conn)

    assert order == ["triage", "summarize"]


# ---------------------------------------------------------------------------
# monthly
# ---------------------------------------------------------------------------


def test_monthly_calls_render_monthly(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_with: dict[str, Any] = {}

    def fake_render_monthly(c: Any, *, month: str, output_dir: Any, **kw: Any) -> RenderResult:
        called_with["month"] = month
        called_with["output_dir"] = output_dir
        return _RENDER_MONTHLY

    monkeypatch.setattr(pipeline.render, "render_monthly", fake_render_monthly)

    args = pipeline.build_parser().parse_args(["monthly", "2026-05"])
    rc = pipeline.cmd_monthly(args, conn)

    assert rc == 0
    assert called_with["month"] == "2026-05"


def test_monthly_pipeline_run_success(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline.render, "render_monthly", lambda c, **kw: _RENDER_MONTHLY
    )

    args = pipeline.build_parser().parse_args(["monthly", "2026-05"])
    pipeline.cmd_monthly(args, conn)

    row = conn.execute(
        "SELECT kind, status FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["kind"] == "monthly"
    assert row["status"] == "success"


def test_monthly_exception_marks_failure(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline.render, "render_monthly", lambda c, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    args = pipeline.build_parser().parse_args(["monthly", "2026-05"])
    rc = pipeline.cmd_monthly(args, conn)

    assert rc == 1
    row = conn.execute(
        "SELECT status FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["status"] == "failure"


def test_monthly_notify_called_with_secret(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBHOOK_SECRET", "secret123")
    notify_called: list[str] = []

    monkeypatch.setattr(
        pipeline.render, "render_monthly", lambda c, **kw: _RENDER_MONTHLY
    )
    monkeypatch.setattr(
        pipeline.notify,
        "notify_for_digest",
        lambda c, *, digest_id, **kw: notify_called.append(digest_id) or _NOTIFY,
    )

    args = pipeline.build_parser().parse_args(["monthly", "2026-05"])
    pipeline.cmd_monthly(args, conn)

    assert notify_called == [_RENDER_MONTHLY.digest_id]


def test_monthly_notify_skipped_without_secret(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notify_called: list[bool] = []

    monkeypatch.setattr(
        pipeline.render, "render_monthly", lambda c, **kw: _RENDER_MONTHLY
    )
    monkeypatch.setattr(
        pipeline.notify,
        "notify_for_digest",
        lambda c, **kw: notify_called.append(True) or _NOTIFY,
    )

    args = pipeline.build_parser().parse_args(["monthly", "2026-05"])
    pipeline.cmd_monthly(args, conn)

    assert not notify_called


# ---------------------------------------------------------------------------
# yearly
# ---------------------------------------------------------------------------


def test_yearly_calls_render_yearly(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called_with: dict[str, Any] = {}

    def fake_render_yearly(c: Any, *, year: str, output_dir: Any, **kw: Any) -> RenderResult:
        called_with["year"] = year
        return _RENDER_YEARLY

    monkeypatch.setattr(pipeline.render, "render_yearly", fake_render_yearly)

    args = pipeline.build_parser().parse_args(["yearly", "2026"])
    rc = pipeline.cmd_yearly(args, conn)

    assert rc == 0
    assert called_with["year"] == "2026"


def test_yearly_pipeline_run_success(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline.render, "render_yearly", lambda c, **kw: _RENDER_YEARLY
    )

    args = pipeline.build_parser().parse_args(["yearly", "2026"])
    pipeline.cmd_yearly(args, conn)

    row = conn.execute(
        "SELECT kind, status FROM pipeline_runs ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    assert row["kind"] == "yearly"
    assert row["status"] == "success"


def test_yearly_exception_marks_failure(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pipeline.render, "render_yearly", lambda c, **kw: (_ for _ in ()).throw(RuntimeError("y"))
    )

    args = pipeline.build_parser().parse_args(["yearly", "2026"])
    rc = pipeline.cmd_yearly(args, conn)

    assert rc == 1


def test_yearly_notify_called_with_secret(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WEBHOOK_SECRET", "mysecret")
    notify_called: list[str] = []

    monkeypatch.setattr(
        pipeline.render, "render_yearly", lambda c, **kw: _RENDER_YEARLY
    )
    monkeypatch.setattr(
        pipeline.notify,
        "notify_for_digest",
        lambda c, *, digest_id, **kw: notify_called.append(digest_id) or _NOTIFY,
    )

    args = pipeline.build_parser().parse_args(["yearly", "2026"])
    pipeline.cmd_yearly(args, conn)

    assert notify_called == [_RENDER_YEARLY.digest_id]


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


def test_health_all_ok(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """health returns 0 when DB, prompts, config, and model are all OK."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("triage", "summarize", "cluster_label", "trend_narrative"):
        (prompts / f"{name}.md").write_text("prompt")
        (prompts / f"{name}.schema.json").write_text("{}")

    config = tmp_path / "config"
    config.mkdir()
    (config / "profile.md").write_text("profile")
    (config / "categories.yaml").write_text("groups: {}")

    monkeypatch.setattr(pipeline, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(pipeline, "CONFIG_DIR", config)
    monkeypatch.setattr(pipeline.embed, "get_model", lambda: MagicMock())

    args = pipeline.build_parser().parse_args(["health"])
    rc = pipeline.cmd_health(args, conn)
    assert rc == 0

    out = capsys.readouterr().out
    assert "OK" in out
    assert "FAIL" not in out


def test_health_missing_prompts_returns_1(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """health returns 1 when prompt files are missing."""
    # prompts dir exists but is empty
    prompts = tmp_path / "prompts"
    prompts.mkdir()

    config = tmp_path / "config"
    config.mkdir()
    (config / "profile.md").write_text("profile")
    (config / "categories.yaml").write_text("groups: {}")

    monkeypatch.setattr(pipeline, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(pipeline, "CONFIG_DIR", config)
    monkeypatch.setattr(pipeline.embed, "get_model", lambda: MagicMock())

    args = pipeline.build_parser().parse_args(["health"])
    rc = pipeline.cmd_health(args, conn)
    assert rc == 1

    out = capsys.readouterr().out
    assert "FAIL" in out


def test_health_missing_config_returns_1(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """health returns 1 when config files are missing."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("triage", "summarize", "cluster_label", "trend_narrative"):
        (prompts / f"{name}.md").write_text("p")
        (prompts / f"{name}.schema.json").write_text("{}")

    # config dir is absent (not created)
    config = tmp_path / "config_missing"

    monkeypatch.setattr(pipeline, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(pipeline, "CONFIG_DIR", config)
    monkeypatch.setattr(pipeline.embed, "get_model", lambda: MagicMock())

    args = pipeline.build_parser().parse_args(["health"])
    rc = pipeline.cmd_health(args, conn)
    assert rc == 1

    out = capsys.readouterr().out
    assert "FAIL" in out


def test_health_db_unreachable_returns_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """health returns 1 when the DB connection is closed (simulating unreachable DB)."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("triage", "summarize", "cluster_label", "trend_narrative"):
        (prompts / f"{name}.md").write_text("p")
        (prompts / f"{name}.schema.json").write_text("{}")

    config = tmp_path / "config"
    config.mkdir()
    (config / "profile.md").write_text("profile")
    (config / "categories.yaml").write_text("groups: {}")

    monkeypatch.setattr(pipeline, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(pipeline, "CONFIG_DIR", config)
    monkeypatch.setattr(pipeline.embed, "get_model", lambda: MagicMock())

    # Create a real conn then close it so execute() raises
    bad_conn = db.connect(tmp_path / "closed.db")
    db.init_schema(bad_conn)
    bad_conn.close()  # now it will raise on execute

    args = pipeline.build_parser().parse_args(["health"])
    rc = pipeline.cmd_health(args, bad_conn)
    assert rc == 1

    out = capsys.readouterr().out
    assert "FAIL" in out


def test_health_model_fail_returns_1(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """health returns 1 when the embedding model cannot be loaded."""
    prompts = tmp_path / "prompts"
    prompts.mkdir()
    for name in ("triage", "summarize", "cluster_label", "trend_narrative"):
        (prompts / f"{name}.md").write_text("p")
        (prompts / f"{name}.schema.json").write_text("{}")

    config = tmp_path / "config"
    config.mkdir()
    (config / "profile.md").write_text("profile")
    (config / "categories.yaml").write_text("groups: {}")

    monkeypatch.setattr(pipeline, "PROMPTS_DIR", prompts)
    monkeypatch.setattr(pipeline, "CONFIG_DIR", config)
    monkeypatch.setattr(
        pipeline.embed, "get_model", lambda: (_ for _ in ()).throw(RuntimeError("no model"))
    )

    args = pipeline.build_parser().parse_args(["health"])
    rc = pipeline.cmd_health(args, conn)
    assert rc == 1


# ---------------------------------------------------------------------------
# main() argparse dispatch
# ---------------------------------------------------------------------------


def test_main_weekly_dispatches_to_cmd_weekly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """main(["weekly"]) calls cmd_weekly via argparse dispatch."""
    called: list[bool] = []

    def fake_cmd_weekly(args: Any, conn: Any) -> int:
        called.append(True)
        return 0

    monkeypatch.setattr(pipeline, "cmd_weekly", fake_cmd_weekly)

    rc = pipeline.main(["weekly"])
    assert rc == 0
    assert called


def test_main_monthly_dispatches_with_arg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """main(["monthly", "2026-05"]) passes month='2026-05' to cmd_monthly."""
    captured: dict[str, str] = {}

    def fake_cmd_monthly(args: Any, conn: Any) -> int:
        captured["month"] = args.month
        return 0

    monkeypatch.setattr(pipeline, "cmd_monthly", fake_cmd_monthly)

    rc = pipeline.main(["monthly", "2026-05"])
    assert rc == 0
    assert captured["month"] == "2026-05"


def test_main_yearly_dispatches_with_arg(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """main(["yearly", "2026"]) passes year='2026' to cmd_yearly."""
    captured: dict[str, str] = {}

    def fake_cmd_yearly(args: Any, conn: Any) -> int:
        captured["year"] = args.year
        return 0

    monkeypatch.setattr(pipeline, "cmd_yearly", fake_cmd_yearly)

    rc = pipeline.main(["yearly", "2026"])
    assert rc == 0
    assert captured["year"] == "2026"


def test_main_backfill_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        pipeline, "cmd_backfill", lambda a, c: called.append(True) or 0
    )
    rc = pipeline.main(["backfill"])
    assert rc == 0
    assert called


def test_main_reprocess_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setattr(
        pipeline,
        "cmd_reprocess",
        lambda a, c: captured.update({"arxiv_id": a.arxiv_id}) or 0,
    )
    rc = pipeline.main(["reprocess", "2401.00001"])
    assert rc == 0
    assert captured["arxiv_id"] == "2401.00001"


def test_main_health_dispatches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called: list[bool] = []
    monkeypatch.setattr(
        pipeline, "cmd_health", lambda a, c: called.append(True) or 0
    )
    rc = pipeline.main(["health"])
    assert rc == 0
    assert called
