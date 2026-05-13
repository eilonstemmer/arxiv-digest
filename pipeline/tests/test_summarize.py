"""Tests for src.summarize: eligibility, FAISS window, cross-domain candidate
selection, prompt rendering, batch wiring, DB writeback, cost accumulation.

Two autouse fixtures isolate from external state:
  * _clean_env wipes settings env vars so DB settings take precedence

llm.run_batch is monkeypatched per test to return canned CallResults.
No real Anthropic calls are made.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from src import db, embed, llm, settings, summarize

REPO_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"

# Fixed "now" so all tests that depend on the 90-day window are deterministic.
NOW = datetime(2024, 4, 15, 12, 0, 0, tzinfo=UTC)
# Papers published within the window
RECENT_DATE = "2024-04-01T00:00:00Z"
# Papers published just outside the 90-day window
OLD_DATE = (NOW - timedelta(days=91)).strftime("%Y-%m-%dT00:00:00Z")


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


@pytest.fixture
def profile_path(tmp_path: Path) -> Path:
    p = tmp_path / "profile.md"
    p.write_text(
        "# Operator profile\n\n## Primary interests\n"
        "- agent memory for long-horizon tasks\n"
        "- cross-domain learning\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# Helper: deterministic unit vectors pointing near a particular axis
# ---------------------------------------------------------------------------


def _near_axis(
    axis: int, *, noise: float = 0.01, seed: int = 0
) -> npt.NDArray[np.float32]:
    """384-dim unit vector strongly pointed along `axis` with small noise."""
    v = np.zeros(embed.EMBEDDING_DIM, dtype=np.float32)
    v[axis] = 1.0
    if noise > 0:
        rng = np.random.default_rng(seed)
        v = v + rng.normal(0, noise, embed.EMBEDDING_DIM).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ---------------------------------------------------------------------------
# Helper: insert a paper with optional embedding
# ---------------------------------------------------------------------------


def _insert(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    title: str = "Paper Title",
    abstract: str = "Abstract text.",
    primary_category: str = "cs.AI",
    categories: list[str] | None = None,
    relevance_score: float = 7.0,
    summarized: int = 0,
    published_at: str = RECENT_DATE,
    vec: npt.NDArray[np.float32] | None = None,
) -> None:
    cats = categories if categories is not None else [primary_category]
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, authors, abstract,
                           primary_category, categories, published_at,
                           fetched_at, url_abs, url_pdf,
                           relevance_score, summarized)
        VALUES (?, ?, '["A. Author"]', ?, ?, ?, ?,
                '2024-01-01T00:00:00Z', ?, ?,
                ?, ?)
        """,
        (
            arxiv_id,
            title,
            abstract,
            primary_category,
            json.dumps(cats),
            published_at,
            f"https://arxiv.org/abs/{arxiv_id}",
            f"https://arxiv.org/pdf/{arxiv_id}",
            relevance_score,
            summarized,
        ),
    )
    if vec is not None:
        embed.store_embedding(conn, arxiv_id, vec)


# ---------------------------------------------------------------------------
# Canned successful CallResult for one paper
# ---------------------------------------------------------------------------


def _success_result(
    arxiv_id: str,
    *,
    cost_usd: float = 0.005,
    cross_domain_hooks: list[dict[str, Any]] | None = None,
) -> llm.CallResult:
    hooks = cross_domain_hooks if cross_domain_hooks is not None else []
    return llm.CallResult(
        custom_id=arxiv_id,
        status="success",
        data={
            "problem": "The paper addresses a gap in reinforcement learning.",
            "method": "Uses a novel transformer-based approach.",
            "key_result": "Achieves 95% accuracy on benchmark datasets.",
            "why_it_matters": "Directly relevant to agent memory tasks.",
            "novelty_signal": "notable",
            "cross_domain_hooks": hooks,
            "tags": ["agent-memory", "reinforcement-learning"],
        },
        error=None,
        raw_text=None,
        input_tokens=400,
        output_tokens=150,
        cached_input_tokens=300,
        cost_usd=cost_usd,
    )


def _all_success_batch(**kw: Any) -> list[llm.CallResult]:
    return [_success_result(r.custom_id) for r in kw["requests"]]


# ---------------------------------------------------------------------------
# Eligibility filters
# ---------------------------------------------------------------------------


def test_eligibility_requires_positive_relevance_score(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # relevance_score = 0 should be excluded
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=0.0)
    _insert(conn, "2401.00002", vec=_near_axis(1), relevance_score=5.0)

    submitted: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.papers_eligible == 1
    assert submitted == ["2401.00002"]


def test_eligibility_excludes_already_summarized(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0, summarized=1)
    _insert(conn, "2401.00002", vec=_near_axis(1), relevance_score=5.0, summarized=0)

    submitted: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert "2401.00001" not in submitted
    assert "2401.00002" in submitted


def test_eligibility_requires_embedding(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", relevance_score=5.0)  # no embedding
    _insert(conn, "2401.00002", vec=_near_axis(0), relevance_score=5.0)

    submitted: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert "2401.00001" not in submitted
    assert "2401.00002" in submitted


def test_eligibility_ordered_by_relevance_desc(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=3.0)
    _insert(conn, "2401.00002", vec=_near_axis(1), relevance_score=9.0)
    _insert(conn, "2401.00003", vec=_near_axis(2), relevance_score=6.0)

    order: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        order.extend(r.custom_id for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert order == ["2401.00002", "2401.00003", "2401.00001"]


def test_digest_top_n_limits_eligible(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(5):
        _insert(conn, f"2401.0000{i}", vec=_near_axis(i), relevance_score=float(i + 1))

    db.set_setting(conn, "digest_top_n", "3")

    submitted: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.papers_eligible == 3
    assert len(submitted) == 3


# ---------------------------------------------------------------------------
# FAISS 90-day window
# ---------------------------------------------------------------------------


def test_faiss_window_excludes_old_papers(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Papers older than 90 days should not appear in cross-domain candidates."""
    # Paper to summarize: cs.AI, axis 0
    _insert(
        conn,
        "2401.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
        published_at=RECENT_DATE,
    )
    # Potential candidate: different category, same axis, but OLD
    _insert(
        conn,
        "2401.old_candidate",
        vec=_near_axis(0, noise=0.001, seed=1),
        primary_category="q-bio.GN",
        relevance_score=0.0,
        published_at=OLD_DATE,
    )

    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    # The old candidate should NOT appear in the user message
    assert len(captured_messages) == 1
    assert "2401.old_candidate" not in captured_messages[0]


def test_faiss_window_includes_recent_papers(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Papers within 90 days should be candidates if they meet criteria."""
    _insert(
        conn,
        "2401.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
        published_at=RECENT_DATE,
    )
    # Very similar but different category, recent
    _insert(
        conn,
        "2401.recent_candidate",
        vec=_near_axis(0, noise=0.001, seed=2),
        primary_category="q-bio.GN",
        relevance_score=0.0,
        published_at=RECENT_DATE,
    )

    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured_messages) == 1
    assert "2401.recent_candidate" in captured_messages[0]


def test_now_parameter_controls_window(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Injecting `now` shifts the 90-day window accordingly."""
    target_published = "2020-01-15T00:00:00Z"
    candidate_published = "2020-01-01T00:00:00Z"

    _insert(
        conn,
        "2020.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
        published_at=target_published,
    )
    _insert(
        conn,
        "2020.candidate",
        vec=_near_axis(0, noise=0.001, seed=5),
        primary_category="q-bio.GN",
        relevance_score=0.0,
        published_at=candidate_published,
    )

    custom_now = datetime(2020, 2, 1, tzinfo=UTC)
    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn,
        profile_path=profile_path,
        prompts_dir=REPO_PROMPTS_DIR,
        now=custom_now,
    )
    assert len(captured_messages) == 1
    # candidate is 45 days before now, so within 90-day window
    assert "2020.candidate" in captured_messages[0]


# ---------------------------------------------------------------------------
# Cross-domain candidates selection
# ---------------------------------------------------------------------------


def test_cross_domain_requires_different_category(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-category papers must not appear as candidates."""
    _insert(
        conn,
        "2401.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
    )
    # Very similar but SAME category
    _insert(
        conn,
        "2401.same_cat",
        vec=_near_axis(0, noise=0.001, seed=3),
        primary_category="cs.AI",
        relevance_score=0.0,
    )

    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured_messages) == 1
    assert "2401.same_cat" not in captured_messages[0]


def test_cross_domain_requires_sim_above_threshold(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Papers with cosine similarity <= 0.65 must not appear as candidates."""
    _insert(
        conn,
        "2401.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
    )
    # Very different axis → low similarity
    _insert(
        conn,
        "2401.low_sim",
        vec=_near_axis(100, noise=0.001),  # orthogonal to axis 0
        primary_category="q-bio.GN",
        relevance_score=0.0,
    )

    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured_messages) == 1
    assert "2401.low_sim" not in captured_messages[0]


def test_cross_domain_top_3_maximum(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At most 3 cross-domain candidates should appear."""
    _insert(
        conn,
        "2401.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
    )
    # 5 very similar papers from a different category
    for i in range(5):
        _insert(
            conn,
            f"2401.cand{i:03d}",
            vec=_near_axis(0, noise=0.001, seed=10 + i),
            primary_category="q-bio.GN",
            relevance_score=0.0,
        )

    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured_messages) == 1
    msg = captured_messages[0]
    cand_count = msg.count("2401.cand")
    assert cand_count <= 3


# ---------------------------------------------------------------------------
# Empty candidates — no-candidates block
# ---------------------------------------------------------------------------


def test_empty_candidates_renders_no_candidates_block(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no candidates qualify, the no-candidates note must appear."""
    _insert(
        conn,
        "2401.lone",
        vec=_near_axis(0),
        primary_category="cs.AI",
        relevance_score=8.0,
    )
    # No other papers exist, so no candidates

    captured_messages: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_messages.extend(r.user_message for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured_messages) == 1
    msg = captured_messages[0]
    # Handlebars loops must be gone
    assert "{{#each CANDIDATES}}" not in msg
    assert "{{#if NO_CANDIDATES}}" not in msg
    # The no-candidates hint must be present
    assert "No candidate cross-domain papers" in msg


def test_empty_candidates_returns_empty_hooks_from_mock(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Sonnet returns empty cross_domain_hooks, result.total is 0."""
    _insert(
        conn,
        "2401.lone",
        vec=_near_axis(0),
        primary_category="cs.AI",
        relevance_score=8.0,
    )
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.total_cross_domain_hooks == 0


# ---------------------------------------------------------------------------
# BatchRequest construction
# ---------------------------------------------------------------------------


def test_batch_request_custom_id_is_arxiv_id(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00042", vec=_near_axis(0), relevance_score=5.0)

    captured: list[llm.BatchRequest] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured.extend(kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured) == 1
    assert captured[0].custom_id == "2401.00042"


def test_user_message_contains_paper_fields(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(
        conn,
        "2401.00001",
        title="Transformer Memory Paper",
        abstract="We study memory in transformers.",
        primary_category="cs.LG",
        categories=["cs.LG", "cs.AI"],
        vec=_near_axis(0),
        relevance_score=7.0,
    )

    captured: list[llm.BatchRequest] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured.extend(kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured) == 1
    msg = captured[0].user_message
    assert "2401.00001" in msg
    assert "Transformer Memory Paper" in msg
    assert "We study memory in transformers." in msg
    assert "cs.LG" in msg
    assert "A. Author" in msg


def test_user_message_no_handlebars_remaining(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)

    captured: list[llm.BatchRequest] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured.extend(kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    msg = captured[0].user_message
    assert "{{" not in msg
    assert "}}" not in msg


def test_system_prompt_contains_profile(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)

    captured_system: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured_system.append(kw["system"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured_system) == 1
    sys_msg = captured_system[0]
    assert "{{CONDENSED_PROFILE}}" not in sys_msg
    assert "agent memory for long-horizon tasks" in sys_msg


def test_user_message_contains_candidate_info(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate details should appear in the user message when present."""
    _insert(
        conn,
        "2401.target",
        vec=_near_axis(0, noise=0.001),
        primary_category="cs.AI",
        relevance_score=8.0,
    )
    _insert(
        conn,
        "2401.cand",
        vec=_near_axis(0, noise=0.001, seed=7),
        primary_category="q-bio.GN",
        title="Cross-domain Paper",
        relevance_score=0.0,
    )

    captured: list[llm.BatchRequest] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured.extend(kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert len(captured) == 1
    msg = captured[0].user_message
    assert "2401.cand" in msg
    assert "q-bio.GN" in msg
    assert "Cross-domain Paper" in msg


# ---------------------------------------------------------------------------
# Successful response writeback
# ---------------------------------------------------------------------------


def test_success_inserts_summary_row(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.papers_summarized_succeeded == 1

    row = conn.execute(
        "SELECT * FROM summaries WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert row is not None
    assert row["problem"] == "The paper addresses a gap in reinforcement learning."
    assert row["novelty_signal"] == "notable"
    assert json.loads(row["tags"]) == ["agent-memory", "reinforcement-learning"]
    assert json.loads(row["cross_domain_hooks"]) == []
    assert row["model"] == settings.DEFAULT_SUMMARIZE_MODEL


def test_success_marks_paper_summarized(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    paper = conn.execute(
        "SELECT summarized FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert paper["summarized"] == 1


def test_cross_domain_hooks_stored_as_json(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)

    hooks = [
        {
            "arxiv_id": "2401.00042",
            "connection": "Both use contrastive learning for representation.",
            "strength": "strong",
        }
    ]

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        return [_success_result(r.custom_id, cross_domain_hooks=hooks) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.total_cross_domain_hooks == 1

    row = conn.execute(
        "SELECT cross_domain_hooks FROM summaries WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    stored = json.loads(row["cross_domain_hooks"])
    assert len(stored) == 1
    assert stored[0]["strength"] == "strong"


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_schema_failed_leaves_summarized_zero(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)

    monkeypatch.setattr(
        llm,
        "run_batch",
        lambda **kw: [
            llm.CallResult(
                custom_id="2401.00001",
                status="schema_failed",
                data=None,
                error="bad JSON",
                raw_text="garbage",
                input_tokens=100,
                output_tokens=20,
                cached_input_tokens=0,
                cost_usd=0.001,
            )
        ],
    )
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.papers_summarized_failed == 1
    assert result.papers_summarized_succeeded == 0

    paper = conn.execute(
        "SELECT summarized FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert paper["summarized"] == 0

    summary = conn.execute(
        "SELECT * FROM summaries WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert summary is None


def test_api_failed_leaves_summarized_zero(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)

    monkeypatch.setattr(
        llm,
        "run_batch",
        lambda **kw: [
            llm.CallResult(
                custom_id="2401.00001",
                status="api_failed",
                data=None,
                error="connection timeout",
                raw_text=None,
                input_tokens=0,
                output_tokens=0,
                cached_input_tokens=0,
                cost_usd=0.0,
            )
        ],
    )
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.papers_summarized_failed == 1

    paper = conn.execute(
        "SELECT summarized FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert paper["summarized"] == 0


# ---------------------------------------------------------------------------
# cost_usd accumulation
# ---------------------------------------------------------------------------


def test_cost_accumulates_across_results(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(3):
        _insert(conn, f"2401.0000{i}", vec=_near_axis(i), relevance_score=5.0)

    monkeypatch.setattr(
        llm,
        "run_batch",
        lambda **kw: [_success_result(r.custom_id, cost_usd=0.005) for r in kw["requests"]],
    )
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.cost_usd == pytest.approx(0.015)


def test_cost_includes_failed_results(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)

    monkeypatch.setattr(
        llm,
        "run_batch",
        lambda **kw: [
            llm.CallResult(
                custom_id="2401.00001",
                status="api_failed",
                data=None,
                error="err",
                raw_text=None,
                input_tokens=100,
                output_tokens=0,
                cached_input_tokens=0,
                cost_usd=0.002,
            )
        ],
    )
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.cost_usd == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Settings plumbing
# ---------------------------------------------------------------------------


def test_summarize_model_passed_to_run_batch(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)
    db.set_setting(conn, "summarize_model", "claude-sonnet-4-6")

    captured: dict[str, Any] = {}

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured.update(kw)
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert captured["model"] == "claude-sonnet-4-6"


def test_api_key_passed_to_run_batch(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)
    db.set_setting(conn, "anthropic_api_key", "my-secret-key")

    captured: dict[str, Any] = {}

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        captured.update(kw)
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert captured["api_key"] == "my-secret-key"


def test_digest_top_n_default(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When digest_top_n is not set, should default to 80."""
    for i in range(5):
        _insert(conn, f"2401.{i:05d}", vec=_near_axis(i), relevance_score=float(i + 1))

    submitted: list[str] = []

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return [_success_result(r.custom_id) for r in kw["requests"]]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    # All 5 papers are within default top_n=80
    assert result.papers_eligible == 5


def test_missing_api_key_raises(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", vec=_near_axis(0), relevance_score=5.0)
    db.delete_setting(conn, "anthropic_api_key")
    with pytest.raises(settings.MissingSettingError, match="anthropic_api_key"):
        summarize.summarize(
            conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
        )


# ---------------------------------------------------------------------------
# Empty DB
# ---------------------------------------------------------------------------


def test_empty_db_returns_zeros(
    conn: sqlite3.Connection, profile_path: Path
) -> None:
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    assert result.papers_eligible == 0
    assert result.papers_summarized_succeeded == 0
    assert result.papers_summarized_failed == 0
    assert result.total_cross_domain_hooks == 0
    assert result.cost_usd == 0.0


# ---------------------------------------------------------------------------
# total_cross_domain_hooks accumulation
# ---------------------------------------------------------------------------


def test_total_cross_domain_hooks_accumulated(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(3):
        _insert(conn, f"2401.0000{i}", vec=_near_axis(i), relevance_score=5.0)

    hooks_per_paper = [
        {"arxiv_id": "2401.00042", "connection": "A real bridge.", "strength": "strong"},
        {"arxiv_id": "2401.00043", "connection": "Another bridge.", "strength": "moderate"},
    ]

    def fake_batch(**kw: Any) -> list[llm.CallResult]:
        return [
            _success_result(r.custom_id, cross_domain_hooks=hooks_per_paper)
            for r in kw["requests"]
        ]

    monkeypatch.setattr(llm, "run_batch", fake_batch)
    result = summarize.summarize(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR, now=NOW
    )
    # 3 papers x 2 hooks each = 6
    assert result.total_cross_domain_hooks == 6


# ---------------------------------------------------------------------------
# _render_candidates_block unit tests
# ---------------------------------------------------------------------------


def test_render_with_candidates_replaces_each_block() -> None:
    template = (
        "## Cross-domain\n"
        "{{#each CANDIDATES}}\n"
        "- **{{this.arxiv_id}}** (sim={{this.similarity}}, category={{this.primary_category}})\n"
        "  Title: {{this.title}}\n"
        "  Abstract: {{this.abstract}}\n"
        "{{/each}}\n"
        "{{#if NO_CANDIDATES}}\n"
        "No candidates.\n"
        "{{/if}}"
    )
    candidates = [
        summarize._Candidate(
            arxiv_id="2401.12345",
            title="Test Title",
            abstract="Test abstract.",
            primary_category="q-bio.GN",
            similarity=0.85,
        )
    ]
    out = summarize._render_candidates_block(template, candidates)
    assert "2401.12345" in out
    assert "q-bio.GN" in out
    assert "{{#each CANDIDATES}}" not in out
    assert "{{#if NO_CANDIDATES}}" not in out
    # No-candidates block must be gone
    assert "No candidates." not in out


def test_render_without_candidates_uses_no_candidates_block() -> None:
    template = (
        "{{#each CANDIDATES}}\n"
        "- **{{this.arxiv_id}}**\n"
        "{{/each}}\n"
        "{{#if NO_CANDIDATES}}\n"
        "No candidates found.\n"
        "{{/if}}"
    )
    out = summarize._render_candidates_block(template, [])
    assert "{{#each CANDIDATES}}" not in out
    assert "{{#if NO_CANDIDATES}}" not in out
    assert "No candidate cross-domain papers" in out


def test_render_with_special_chars_in_title() -> None:
    """Backslashes and regex replacement groups in titles must not break rendering."""
    template = "{{#each CANDIDATES}}CONTENT{{/each}}{{#if NO_CANDIDATES}}NOCAN{{/if}}"
    candidates = [
        summarize._Candidate(
            arxiv_id="2401.00001",
            title=r"Paper with \1 and \g<group>",
            abstract="Normal abstract.",
            primary_category="math.NT",
            similarity=0.9,
        )
    ]
    out = summarize._render_candidates_block(template, candidates)
    assert r"\1" in out
    assert r"\g<group>" in out
