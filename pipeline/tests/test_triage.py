"""Tests for src.triage: profile embedding cache, prefilter scoring,
Haiku batch wiring, end-to-end DB state changes.

Two autouse fixtures isolate from external state:
  * _fake_embed_model swaps embed._model for a deterministic stand-in
  * _clean_env wipes settings env vars so DB settings take precedence

llm.run_batch is monkeypatched per test to return canned CallResults.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from src import db, embed, llm, settings, triage

REPO_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class _FakeEmbedModel:
    """Deterministic MD5-seeded stand-in for SentenceTransformer."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> npt.NDArray[np.float32]:
        out = np.zeros((len(sentences), embed.EMBEDDING_DIM), dtype=np.float32)
        for i, s in enumerate(sentences):
            digest = hashlib.md5(s.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(embed.EMBEDDING_DIM).astype(np.float32)
            n = float(np.linalg.norm(v))
            if n > 0:
                v = v / n
            out[i] = v
        return out


@pytest.fixture(autouse=True)
def _fake_embed_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "_model", _FakeEmbedModel())


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
        "- tool-use verification\n",
        encoding="utf-8",
    )
    return p


def _insert(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    title: str = "Title",
    abstract: str = "Abstract.",
    published_at: str = "2024-01-01T00:00:00Z",
    embedding_text: str | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, authors, abstract, primary_category,
                           categories, published_at, fetched_at, url_abs, url_pdf)
        VALUES (?, ?, '["A. Author"]', ?, 'cs.AI', '["cs.AI"]', ?,
                '2024-01-01T00:00:00Z', ?, ?)
        """,
        (
            arxiv_id,
            title,
            abstract,
            published_at,
            f"https://arxiv.org/abs/{arxiv_id}",
            f"https://arxiv.org/pdf/{arxiv_id}",
        ),
    )
    if embedding_text is not None:
        embed.store_embedding(
            conn, arxiv_id, embed.embed_text(embedding_text)
        )


def _success_result(
    custom_id: str,
    *,
    score: float = 7.0,
    reason: str = "ok",
    cost_usd: float = 0.0001,
) -> llm.CallResult:
    return llm.CallResult(
        custom_id=custom_id,
        status="success",
        data={
            "relevance_score": score,
            "reason": reason,
            "matched_interests": [],
            "anti_interest_flags": [],
        },
        error=None,
        raw_text=None,
        input_tokens=100,
        output_tokens=30,
        cached_input_tokens=80,
        cost_usd=cost_usd,
    )


def _all_success_batch(**kw: Any) -> list[llm.CallResult]:
    return [_success_result(r.custom_id) for r in kw["requests"]]


# ---- profile_embedding -----------------------------------------------------


def test_profile_embedding_first_call_computes_and_caches(
    conn: sqlite3.Connection, profile_path: Path
) -> None:
    vec = triage.profile_embedding(conn, profile_path=profile_path)
    assert vec.shape == (embed.EMBEDDING_DIM,)
    assert settings.profile_embedding_hash(conn) is not None
    assert settings.profile_embedding_b64(conn) is not None


def test_profile_embedding_second_call_uses_cache(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vec1 = triage.profile_embedding(conn, profile_path=profile_path)
    call_count = 0
    real_embed_text = embed.embed_text

    def counted(text: str) -> npt.NDArray[np.float32]:
        nonlocal call_count
        call_count += 1
        return real_embed_text(text)

    monkeypatch.setattr(embed, "embed_text", counted)
    vec2 = triage.profile_embedding(conn, profile_path=profile_path)
    np.testing.assert_array_equal(vec1, vec2)
    assert call_count == 0


def test_profile_embedding_invalidates_on_content_change(
    conn: sqlite3.Connection, profile_path: Path
) -> None:
    vec1 = triage.profile_embedding(conn, profile_path=profile_path)
    profile_path.write_text("# Different content\n", encoding="utf-8")
    vec2 = triage.profile_embedding(conn, profile_path=profile_path)
    assert not np.array_equal(vec1, vec2)


# ---- _eligible_papers ------------------------------------------------------


def test_eligible_excludes_papers_without_embedding(
    conn: sqlite3.Connection,
) -> None:
    _insert(conn, "2401.00001", embedding_text="a")
    _insert(conn, "2401.00002")  # no embedding
    rows = triage._eligible_papers(conn)
    ids = [str(r["arxiv_id"]) for r in rows]
    assert ids == ["2401.00001"]


def test_eligible_excludes_already_triaged(conn: sqlite3.Connection) -> None:
    _insert(conn, "2401.00001", embedding_text="a")
    _insert(conn, "2401.00002", embedding_text="b")
    conn.execute(
        "UPDATE papers SET triaged_at = '2024-01-01' WHERE arxiv_id = '2401.00001'"
    )
    rows = triage._eligible_papers(conn)
    assert [str(r["arxiv_id"]) for r in rows] == ["2401.00002"]


def test_eligible_ordered_by_published_desc(conn: sqlite3.Connection) -> None:
    _insert(
        conn,
        "2401.00001",
        embedding_text="a",
        published_at="2020-01-01T00:00:00Z",
    )
    _insert(
        conn,
        "2401.00002",
        embedding_text="b",
        published_at="2024-01-01T00:00:00Z",
    )
    rows = triage._eligible_papers(conn)
    assert [str(r["arxiv_id"]) for r in rows] == ["2401.00002", "2401.00001"]


# ---- prefilter ------------------------------------------------------------


def test_prefilter_keeps_top_fraction(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(4):
        _insert(conn, f"2401.0000{i + 1}", embedding_text=f"paper{i}")

    submitted: list[str] = []

    def fake_run_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return _all_success_batch(**kw)

    monkeypatch.setattr(llm, "run_batch", fake_run_batch)
    db.set_setting(conn, "triage_prefilter_keep_fraction", "0.5")

    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.papers_eligible == 4
    assert result.papers_prefiltered == 2
    assert result.papers_haiku_requested == 2
    assert len(submitted) == 2


def test_prefilter_marks_drops_as_triaged_with_zero(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(4):
        _insert(conn, f"2401.0000{i + 1}", embedding_text=f"p{i}")
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    db.set_setting(conn, "triage_prefilter_keep_fraction", "0.5")

    triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )

    rows = conn.execute(
        "SELECT arxiv_id, relevance_score, relevance_reason, triaged_at FROM papers "
        "WHERE relevance_reason = ?",
        (triage.PREFILTER_DROPPED_REASON,),
    ).fetchall()
    assert len(rows) == 2
    for r in rows:
        assert r["relevance_score"] == 0.0
        assert r["triaged_at"] is not None


def test_prefilter_writes_score_for_every_paper(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(4):
        _insert(conn, f"2401.0000{i + 1}", embedding_text=f"p{i}")
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    db.set_setting(conn, "triage_prefilter_keep_fraction", "0.5")

    triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )

    rows = conn.execute(
        "SELECT prefilter_score FROM papers"
    ).fetchall()
    assert len(rows) == 4
    assert all(r["prefilter_score"] is not None for r in rows)


def test_prefilter_keep_fraction_floors_at_one(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # keep_fraction=0.01 with 4 papers rounds to 0, but min is 1
    _insert(conn, "2401.00001", embedding_text="a")
    _insert(conn, "2401.00002", embedding_text="b")
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    db.set_setting(conn, "triage_prefilter_keep_fraction", "0.01")

    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.papers_haiku_requested >= 1


# ---- prefilter disabled ----------------------------------------------------


def test_prefilter_disabled_sends_all_to_haiku(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(4):
        _insert(conn, f"2401.0000{i + 1}", embedding_text=f"p{i}")

    submitted: list[str] = []

    def fake_run_batch(**kw: Any) -> list[llm.CallResult]:
        submitted.extend(r.custom_id for r in kw["requests"])
        return _all_success_batch(**kw)

    monkeypatch.setattr(llm, "run_batch", fake_run_batch)
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.papers_prefiltered == 0
    assert result.papers_haiku_requested == 4
    assert len(submitted) == 4


def test_prefilter_disabled_skips_profile_embedding(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", embedding_text="a")
    monkeypatch.setattr(llm, "run_batch", _all_success_batch)
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    # Profile embedding cache should NOT be populated.
    assert settings.profile_embedding_hash(conn) is None


# ---- Haiku result writeback -----------------------------------------------


def test_writes_successful_haiku_response(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", embedding_text="a")

    def fake_run_batch(**kw: Any) -> list[llm.CallResult]:
        return [
            _success_result(
                "2401.00001",
                score=8.5,
                reason="directly advances primary interest in agent memory",
            )
        ]

    monkeypatch.setattr(llm, "run_batch", fake_run_batch)
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.papers_haiku_succeeded == 1

    row = conn.execute(
        "SELECT relevance_score, relevance_reason, triaged_at FROM papers "
        "WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert row["relevance_score"] == 8.5
    assert "agent memory" in row["relevance_reason"]
    assert row["triaged_at"] is not None


def test_failed_haiku_result_marked_triaged_with_zero(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", embedding_text="a")
    monkeypatch.setattr(
        llm,
        "run_batch",
        lambda **kw: [
            llm.CallResult(
                custom_id="2401.00001",
                status="schema_failed",
                data=None,
                error="invalid JSON",
                raw_text="garbage",
                input_tokens=100,
                output_tokens=20,
                cached_input_tokens=0,
                cost_usd=0.0001,
            )
        ],
    )
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.papers_haiku_failed == 1
    row = conn.execute(
        "SELECT relevance_score, relevance_reason, triaged_at FROM papers "
        "WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert row["relevance_score"] == 0.0
    assert "triage failed" in row["relevance_reason"]
    assert "invalid JSON" in row["relevance_reason"]
    assert row["triaged_at"] is not None


def test_user_message_includes_paper_fields(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(
        conn,
        "2401.00001",
        title="My Paper Title",
        abstract="Abstract content here.",
        embedding_text="x",
    )

    captured: list[llm.BatchRequest] = []

    def fake_run_batch(**kw: Any) -> list[llm.CallResult]:
        captured.extend(kw["requests"])
        return _all_success_batch(**kw)

    monkeypatch.setattr(llm, "run_batch", fake_run_batch)
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert len(captured) == 1
    msg = captured[0].user_message
    assert "My Paper Title" in msg
    assert "Abstract content here." in msg
    assert "cs.AI" in msg


def test_system_prompt_includes_profile_text(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", embedding_text="a")

    captured_system: list[str] = []

    def fake_run_batch(**kw: Any) -> list[llm.CallResult]:
        captured_system.append(kw["system"])
        return _all_success_batch(**kw)

    monkeypatch.setattr(llm, "run_batch", fake_run_batch)
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert "{{PROFILE_MARKDOWN}}" not in captured_system[0]
    assert "agent memory for long-horizon tasks" in captured_system[0]


# ---- no-op + cost ---------------------------------------------------------


def test_no_eligible_papers_returns_zero(
    conn: sqlite3.Connection, profile_path: Path
) -> None:
    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.papers_eligible == 0
    assert result.papers_haiku_requested == 0
    assert result.cost_usd == 0.0


def test_cost_accumulates_per_result(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for i in range(3):
        _insert(conn, f"2401.0000{i + 1}", embedding_text=f"p{i}")

    monkeypatch.setattr(
        llm,
        "run_batch",
        lambda **kw: [
            _success_result(r.custom_id, cost_usd=0.0025)
            for r in kw["requests"]
        ],
    )
    db.set_setting(conn, "triage_prefilter_enabled", "false")

    result = triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.cost_usd == pytest.approx(0.0025 * 3)


# ---- model + api_key plumbing ---------------------------------------------


def test_model_setting_passed_to_run_batch(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _insert(conn, "2401.00001", embedding_text="a")
    captured: dict[str, Any] = {}

    def fake_run_batch(**kw: Any) -> list[llm.CallResult]:
        captured.update(kw)
        return _all_success_batch(**kw)

    monkeypatch.setattr(llm, "run_batch", fake_run_batch)
    db.set_setting(conn, "triage_prefilter_enabled", "false")
    db.set_setting(conn, "triage_model", "claude-haiku-4-5-20251001")
    db.set_setting(conn, "anthropic_api_key", "key-abc")

    triage.triage(
        conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
    )
    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["api_key"] == "key-abc"


def test_missing_api_key_raises(
    conn: sqlite3.Connection,
    profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # _insert needs to happen to make eligibility check trigger the API key lookup
    _insert(conn, "2401.00001", embedding_text="a")
    db.delete_setting(conn, "anthropic_api_key")
    with pytest.raises(settings.MissingSettingError, match="anthropic_api_key"):
        triage.triage(
            conn, profile_path=profile_path, prompts_dir=REPO_PROMPTS_DIR
        )
