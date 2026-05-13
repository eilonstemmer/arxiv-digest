"""Tests for src.trend: ISO week arithmetic, delta computation, velocity
classification, is_new detection, DB persistence, idempotency, narrative
generation, failure paths, and cost accumulation.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from src import db, llm, settings, trend

REPO_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (*settings.USER_SETTING_KEYS, *settings.MANAGED_SETTING_KEYS):
        monkeypatch.delenv(key.upper(), raising=False)


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    db.set_setting(c, "anthropic_api_key", "test-key")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Helpers for inserting test data
# ---------------------------------------------------------------------------


def _insert_paper(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    title: str = "A Test Paper",
    relevance_score: float = 5.0,
) -> None:
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, authors, abstract,
                           primary_category, categories,
                           published_at, fetched_at, url_abs, url_pdf,
                           relevance_score)
        VALUES (?, ?, '["A"]', 'Abstract.', 'cs.AI', '["cs.AI"]',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z',
                'https://arxiv.org/abs/' || ?, 'https://arxiv.org/pdf/' || ?,
                ?)
        """,
        (arxiv_id, title, arxiv_id, arxiv_id, relevance_score),
    )


def _insert_cluster(
    conn: sqlite3.Connection,
    week: str,
    cluster_id: int,
    *,
    label: str = "Test Cluster",
    description: str = "A test cluster.",
    paper_count: int = 10,
) -> None:
    conn.execute(
        """
        INSERT INTO clusters(week, cluster_id, label, description,
                             subtopics, paper_count)
        VALUES (?, ?, ?, ?, '[]', ?)
        """,
        (week, cluster_id, label, description, paper_count),
    )


def _insert_paper_cluster(
    conn: sqlite3.Connection,
    arxiv_id: str,
    week: str,
    cluster_id: int,
) -> None:
    conn.execute(
        "INSERT INTO paper_clusters(arxiv_id, week, cluster_id) VALUES (?, ?, ?)",
        (arxiv_id, week, cluster_id),
    )


def _success_narrative() -> llm.CallResult:
    return llm.CallResult(
        custom_id="",
        status="success",
        data={
            "headline": "LLM Reasoning clusters surge 45% over four-week baseline.",
            "movements": [
                {
                    "cluster_label": "LLM Reasoning",
                    "direction": "accelerating",
                    "narrative": (
                        "The LLM Reasoning cluster grew 45% week-over-4-week "
                        "and 30% over 12 weeks, driven by chain-of-thought "
                        "scaling papers. Operators interested in reasoning "
                        "should watch closely."
                    ),
                    "key_papers": ["2401.00001"],
                }
            ],
            "cross_domain_observation": "",
            "weekly_question": (
                "Does the new chain-of-thought variant generalize beyond "
                "English-language math benchmarks?"
            ),
        },
        error=None,
        raw_text=None,
        input_tokens=500,
        output_tokens=150,
        cached_input_tokens=0,
        cost_usd=0.0075,
    )


def _schema_failed_result() -> llm.CallResult:
    return llm.CallResult(
        custom_id="",
        status="schema_failed",
        data=None,
        error="schema validation failed: 'headline' is a required property",
        raw_text="{}",
        input_tokens=500,
        output_tokens=10,
        cached_input_tokens=0,
        cost_usd=0.002,
    )


def _api_failed_result() -> llm.CallResult:
    return llm.CallResult(
        custom_id="",
        status="api_failed",
        data=None,
        error="Connection refused",
        raw_text=None,
        input_tokens=0,
        output_tokens=0,
        cached_input_tokens=0,
        cost_usd=0.0,
    )


# ---------------------------------------------------------------------------
# _prior_week: ISO week arithmetic + year-rollover edge cases
# ---------------------------------------------------------------------------


def test_prior_week_simple() -> None:
    assert trend._prior_week("2026-W20", 1) == "2026-W19"


def test_prior_week_multi_week() -> None:
    assert trend._prior_week("2026-W20", 4) == "2026-W16"


def test_prior_week_year_rollover_w01_minus_1() -> None:
    # 2026-W01 Monday = 2025-12-29.  One week before = 2025-12-22 = 2025-W52.
    result = trend._prior_week("2026-W01", 1)
    assert result == "2025-W52"


def test_prior_week_year_rollover_w02_minus_2() -> None:
    # 2026-W02 minus 2 = 2025-W52
    result = trend._prior_week("2026-W02", 2)
    assert result == "2025-W52"


def test_prior_week_w53_year() -> None:
    # 2020 had W53. 2021-W01 minus 1 should land in 2020-W53.
    result = trend._prior_week("2021-W01", 1)
    assert result == "2020-W53"


def test_prior_week_52_weeks_back_same_year() -> None:
    # 52 weeks before 2026-W20 should be 2025-W20 (52 weeks exactly).
    result = trend._prior_week("2026-W20", 52)
    assert result == "2025-W20"


def test_prior_week_large_offset() -> None:
    # 53 weeks back to verify crossing W53 year boundary if needed.
    result = trend._prior_week("2026-W20", 53)
    assert result == "2025-W19"


# ---------------------------------------------------------------------------
# _compute_delta
# ---------------------------------------------------------------------------


def test_compute_delta_positive() -> None:
    d = trend._compute_delta(15, 10)
    assert d == pytest.approx(0.5)


def test_compute_delta_negative() -> None:
    d = trend._compute_delta(5, 10)
    assert d == pytest.approx(-0.5)


def test_compute_delta_zero_prior_returns_none() -> None:
    assert trend._compute_delta(5, 0) is None


def test_compute_delta_none_prior_returns_none() -> None:
    assert trend._compute_delta(5, None) is None


def test_compute_delta_no_change() -> None:
    assert trend._compute_delta(10, 10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _classify_velocity
# ---------------------------------------------------------------------------


def test_velocity_accelerating() -> None:
    # w4 > 0.3 AND w12 > 0.2
    assert trend._classify_velocity(0.4, 0.3) == "accelerating"


def test_velocity_accelerating_boundary() -> None:
    # Exactly at boundary — spec says > not >=
    assert trend._classify_velocity(0.3, 0.2) != "accelerating"


def test_velocity_growing() -> None:
    # w4 > 0.1 but NOT accelerating
    assert trend._classify_velocity(0.2, 0.05) == "growing"


def test_velocity_growing_w12_none() -> None:
    # w12 is None so can't be accelerating; w4 > 0.1 → growing
    assert trend._classify_velocity(0.35, None) == "growing"


def test_velocity_shrinking() -> None:
    assert trend._classify_velocity(-0.2, -0.1) == "shrinking"


def test_velocity_flat_small_positive() -> None:
    # w4 = 0.05 → not > 0.1, not < -0.1 → flat
    assert trend._classify_velocity(0.05, 0.01) == "flat"


def test_velocity_flat_small_negative() -> None:
    assert trend._classify_velocity(-0.05, -0.03) == "flat"


def test_velocity_flat_none_w4() -> None:
    assert trend._classify_velocity(None, None) == "flat"


def test_velocity_flat_none_w4_w12_present() -> None:
    assert trend._classify_velocity(None, 0.5) == "flat"


# ---------------------------------------------------------------------------
# is_new detection
# ---------------------------------------------------------------------------


def test_is_new_when_no_prior_history(conn: sqlite3.Connection) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Brand New Cluster")
    assert trend._is_new(conn, "Brand New Cluster", "2026-W20") == 1


def test_is_new_false_when_appeared_recently(conn: sqlite3.Connection) -> None:
    _insert_cluster(conn, "2026-W19", 0, label="Old Cluster")
    _insert_cluster(conn, "2026-W20", 0, label="Old Cluster")
    assert trend._is_new(conn, "Old Cluster", "2026-W20") == 0


def test_is_new_false_when_appeared_12_weeks_ago(conn: sqlite3.Connection) -> None:
    prior = trend._prior_week("2026-W20", 12)
    _insert_cluster(conn, prior, 0, label="Borderline Cluster")
    _insert_cluster(conn, "2026-W20", 0, label="Borderline Cluster")
    # 12 weeks back is still within the look-back window (w-1 through w-12)
    assert trend._is_new(conn, "Borderline Cluster", "2026-W20") == 0


def test_is_new_true_when_only_appeared_13_plus_weeks_ago(
    conn: sqlite3.Connection,
) -> None:
    # 13 weeks ago is outside the look-back window
    prior = trend._prior_week("2026-W20", 13)
    _insert_cluster(conn, prior, 0, label="Ancient Cluster")
    _insert_cluster(conn, "2026-W20", 0, label="Ancient Cluster")
    assert trend._is_new(conn, "Ancient Cluster", "2026-W20") == 1


# ---------------------------------------------------------------------------
# cluster_trends rows written
# ---------------------------------------------------------------------------


def test_cluster_trends_persisted(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="LLM Reasoning", paper_count=15)
    _insert_cluster(conn, "2026-W19", 0, label="LLM Reasoning", paper_count=10)

    _insert_paper(conn, "2401.00001")
    _insert_paper_cluster(conn, "2401.00001", "2026-W20", 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_narrative())

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.clusters_processed == 1

    row = conn.execute(
        "SELECT * FROM cluster_trends WHERE week = '2026-W20' AND cluster_id = 0"
    ).fetchone()
    assert row is not None
    # delta_vs_w1 = (15 - 10) / 10 = 0.5
    assert float(row["delta_vs_w1"]) == pytest.approx(0.5)
    # No w4/w12/w52 prior → None
    assert row["delta_vs_w4"] is None
    assert row["delta_vs_w12"] is None
    assert row["delta_vs_w52"] is None
    assert int(row["is_new"]) == 0  # appeared at w-1
    assert str(row["velocity_class"]) == "flat"  # w4 is None → flat


def test_cluster_trends_velocity_accelerating(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    week = "2026-W20"
    _insert_cluster(conn, week, 0, label="Hot Topic", paper_count=15)
    # w4 prior: 10 → delta_vs_w4 = 0.5
    _insert_cluster(conn, trend._prior_week(week, 4), 0, label="Hot Topic", paper_count=10)
    # w12 prior: 12 → delta_vs_w12 = (15-12)/12 = 0.25
    _insert_cluster(conn, trend._prior_week(week, 12), 0, label="Hot Topic", paper_count=12)

    _insert_paper(conn, "2401.00002")
    _insert_paper_cluster(conn, "2401.00002", week, 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_narrative())

    trend.compute_trends(conn, week=week, prompts_dir=REPO_PROMPTS_DIR)

    row = conn.execute(
        "SELECT velocity_class FROM cluster_trends WHERE week = ? AND cluster_id = 0",
        (week,),
    ).fetchone()
    assert str(row["velocity_class"]) == "accelerating"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_idempotent_rerun(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Stable Cluster", paper_count=8)
    _insert_paper(conn, "2401.00003")
    _insert_paper_cluster(conn, "2401.00003", "2026-W20", 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_narrative())

    trend.compute_trends(conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR)
    trend.compute_trends(conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR)

    count = conn.execute(
        "SELECT COUNT(*) AS c FROM cluster_trends WHERE week = '2026-W20'"
    ).fetchone()["c"]
    assert int(count) == 1


# ---------------------------------------------------------------------------
# Narrative call: success → payload populated
# ---------------------------------------------------------------------------


def test_narrative_success(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="LLM Reasoning", paper_count=10)
    _insert_paper(conn, "2401.00001")
    _insert_paper_cluster(conn, "2401.00001", "2026-W20", 0)

    captured_kwargs: list[dict[str, Any]] = []

    def fake_direct(**kw: Any) -> llm.CallResult:
        captured_kwargs.append(kw)
        return _success_narrative()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.trend_payload is not None
    assert "headline" in result.trend_payload
    assert result.cost_usd == pytest.approx(0.0075)
    assert len(captured_kwargs) == 1  # exactly one LLM call


# ---------------------------------------------------------------------------
# Schema failure → payload is None, function doesn't raise
# ---------------------------------------------------------------------------


def test_schema_failure_returns_none_payload(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Cluster A", paper_count=5)
    _insert_paper(conn, "2401.00004")
    _insert_paper_cluster(conn, "2401.00004", "2026-W20", 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _schema_failed_result())

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.trend_payload is None
    assert result.clusters_processed == 1
    # Cost still reflects what was spent on the failed call
    assert result.cost_usd == pytest.approx(0.002)


def test_api_failure_returns_none_payload(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Cluster B", paper_count=5)
    _insert_paper(conn, "2401.00005")
    _insert_paper_cluster(conn, "2401.00005", "2026-W20", 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _api_failed_result())

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.trend_payload is None
    assert result.clusters_processed == 1


# ---------------------------------------------------------------------------
# Empty clusters → no LLM call, clusters_processed = 0
# ---------------------------------------------------------------------------


def test_empty_clusters_no_llm_call(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(
        llm, "call_direct", lambda **_: calls.append("called") or _success_narrative()
    )

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.clusters_processed == 0
    assert result.trend_payload is None
    assert result.cost_usd == 0.0
    assert calls == []


# ---------------------------------------------------------------------------
# Top-N paper selection (ordered by relevance_score DESC)
# ---------------------------------------------------------------------------


def test_top_papers_ordered_by_relevance(conn: sqlite3.Connection) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Ranked Cluster", paper_count=3)

    # Insert 3 papers with different relevance scores.
    for arxiv_id, score in [("2401.00010", 9.0), ("2401.00011", 3.0), ("2401.00012", 7.5)]:
        _insert_paper(conn, arxiv_id, title=f"Paper {arxiv_id}", relevance_score=score)
        _insert_paper_cluster(conn, arxiv_id, "2026-W20", 0)

    top = trend._top_papers(conn, "2026-W20", 0, top_n=3)

    assert len(top) == 3
    # Should be sorted descending: 9.0, 7.5, 3.0
    assert top[0]["arxiv_id"] == "2401.00010"
    assert top[1]["arxiv_id"] == "2401.00012"
    assert top[2]["arxiv_id"] == "2401.00011"


def test_top_papers_limit(conn: sqlite3.Connection) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Big Cluster", paper_count=10)

    for i in range(8):
        arxiv_id = f"2401.{i:05d}"
        _insert_paper(conn, arxiv_id, relevance_score=float(i))
        _insert_paper_cluster(conn, arxiv_id, "2026-W20", 0)

    top = trend._top_papers(conn, "2026-W20", 0, top_n=5)
    assert len(top) == 5


def test_top_papers_included_in_prompt(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Focus Cluster", paper_count=2)
    _insert_paper(conn, "2401.99001", title="High Relevance Paper", relevance_score=9.5)
    _insert_paper(conn, "2401.99002", title="Low Relevance Paper", relevance_score=1.0)
    _insert_paper_cluster(conn, "2401.99001", "2026-W20", 0)
    _insert_paper_cluster(conn, "2401.99002", "2026-W20", 0)

    captured_user: list[str] = []

    def fake_direct(**kw: Any) -> llm.CallResult:
        captured_user.append(kw["user"])
        return _success_narrative()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    trend.compute_trends(conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR)

    user_msg = captured_user[0]
    assert "2401.99001" in user_msg
    assert "High Relevance Paper" in user_msg
    # The Handlebars loop markers must be substituted out
    assert "{{#each CLUSTERS}}" not in user_msg
    assert "{{#each this.top_papers}}" not in user_msg


# ---------------------------------------------------------------------------
# Cost accumulation
# ---------------------------------------------------------------------------


def test_cost_from_llm_call(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="Cost Cluster", paper_count=5)
    _insert_paper(conn, "2401.55555")
    _insert_paper_cluster(conn, "2401.55555", "2026-W20", 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_narrative())

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.cost_usd == pytest.approx(0.0075)


# ---------------------------------------------------------------------------
# Settings plumbing: summarize_model + anthropic_api_key forwarded
# ---------------------------------------------------------------------------


def test_settings_model_forwarded(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db.set_setting(conn, "summarize_model", "claude-sonnet-4-6")

    _insert_cluster(conn, "2026-W20", 0, label="Settings Test Cluster", paper_count=3)
    _insert_paper(conn, "2401.77777")
    _insert_paper_cluster(conn, "2401.77777", "2026-W20", 0)

    captured: list[str] = []

    def fake_direct(**kw: Any) -> llm.CallResult:
        captured.append(kw["model"])
        return _success_narrative()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    trend.compute_trends(conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR)

    assert captured == ["claude-sonnet-4-6"]


def test_settings_api_key_forwarded(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="API Key Test", paper_count=2)
    _insert_paper(conn, "2401.88888")
    _insert_paper_cluster(conn, "2401.88888", "2026-W20", 0)

    captured_keys: list[str] = []

    def fake_direct(**kw: Any) -> llm.CallResult:
        captured_keys.append(kw["api_key"])
        return _success_narrative()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    trend.compute_trends(conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR)

    assert captured_keys == ["test-key"]


# ---------------------------------------------------------------------------
# Multiple clusters: all rows written
# ---------------------------------------------------------------------------


def test_multiple_clusters_all_persisted(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for cid, label in [(0, "Cluster Alpha"), (1, "Cluster Beta"), (2, "Cluster Gamma")]:
        _insert_cluster(conn, "2026-W20", cid, label=label, paper_count=5)
        arxiv_id = f"2401.{cid:05d}"
        _insert_paper(conn, arxiv_id)
        _insert_paper_cluster(conn, arxiv_id, "2026-W20", cid)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_narrative())

    result = trend.compute_trends(
        conn, week="2026-W20", prompts_dir=REPO_PROMPTS_DIR
    )

    assert result.clusters_processed == 3

    rows = conn.execute(
        "SELECT cluster_id FROM cluster_trends WHERE week = '2026-W20' ORDER BY cluster_id"
    ).fetchall()
    assert [int(r["cluster_id"]) for r in rows] == [0, 1, 2]


# ---------------------------------------------------------------------------
# Profile path plumbing
# ---------------------------------------------------------------------------


def test_profile_content_included_in_prompt(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_file = tmp_path / "profile.md"
    profile_file.write_text("# My Research Interests\nI study large language models.", encoding="utf-8")

    _insert_cluster(conn, "2026-W20", 0, label="LLM Research", paper_count=3)
    _insert_paper(conn, "2401.44444")
    _insert_paper_cluster(conn, "2401.44444", "2026-W20", 0)

    captured_user: list[str] = []

    def fake_direct(**kw: Any) -> llm.CallResult:
        captured_user.append(kw["user"])
        return _success_narrative()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    trend.compute_trends(
        conn,
        week="2026-W20",
        profile_path=profile_file,
        prompts_dir=REPO_PROMPTS_DIR,
    )

    assert "I study large language models." in captured_user[0]


def test_missing_profile_path_uses_empty_string(
    conn: sqlite3.Connection,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert_cluster(conn, "2026-W20", 0, label="No Profile Cluster", paper_count=2)
    _insert_paper(conn, "2401.33333")
    _insert_paper_cluster(conn, "2401.33333", "2026-W20", 0)

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_narrative())

    # Should not raise even when profile_path is None
    result = trend.compute_trends(
        conn, week="2026-W20", profile_path=None, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.clusters_processed == 1
