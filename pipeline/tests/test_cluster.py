"""Tests for src.cluster: HDBSCAN integration, label calls, DB writes,
idempotency, fallback behavior.

`_hdbscan_labels` is monkeypatched in most tests so cluster shapes are
deterministic. One end-to-end test exercises real sklearn HDBSCAN against
synthetic embeddings that should clearly resolve into 3 groups.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from src import cluster, db, embed, llm, settings

REPO_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


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


def _near_axis(axis: int, *, noise: float = 0.01, seed: int = 0) -> npt.NDArray[np.float32]:
    """384-dim unit vector strongly pointed along `axis` with small noise."""
    v = np.zeros(embed.EMBEDDING_DIM, dtype=np.float32)
    v[axis] = 1.0
    if noise > 0:
        rng = np.random.default_rng(seed)
        v = v + rng.normal(0, noise, embed.EMBEDDING_DIM).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _insert_with_embedding(
    conn: sqlite3.Connection,
    arxiv_id: str,
    *,
    vec: npt.NDArray[np.float32] | None = None,
    title: str = "Paper Title",
    abstract: str = "Abstract sentence one. More content here.",
    published_at: str = "2024-01-15T00:00:00Z",
) -> None:
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, authors, abstract, primary_category,
                           categories, published_at, fetched_at, url_abs, url_pdf)
        VALUES (?, ?, '["A"]', ?, 'cs.AI', '["cs.AI"]', ?,
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
    if vec is not None:
        embed.store_embedding(conn, arxiv_id, vec)


def _success_label(label: str = "RAG for Code") -> llm.CallResult:
    return llm.CallResult(
        custom_id="",
        status="success",
        data={
            "label": label,
            "description": "Papers on retrieval-augmented code generation and tool use.",
            "subtopics": ["retrieval", "code-generation"],
        },
        error=None,
        raw_text=None,
        input_tokens=200,
        output_tokens=60,
        cached_input_tokens=0,
        cost_usd=0.003,
    )


def _schema_failed() -> llm.CallResult:
    return llm.CallResult(
        custom_id="",
        status="schema_failed",
        data=None,
        error="bad output",
        raw_text="garbage",
        input_tokens=150,
        output_tokens=20,
        cached_input_tokens=0,
        cost_usd=0.0008,
    )


# ---- first_sentence helper ------------------------------------------------


def test_first_sentence_terminates_at_period() -> None:
    assert cluster._first_sentence("Hello world. More text.") == "Hello world."


def test_first_sentence_handles_exclamation() -> None:
    assert cluster._first_sentence("Surprise! Another sentence.") == "Surprise!"


def test_first_sentence_fallback_on_long_run() -> None:
    long_text = "x" * 500
    out = cluster._first_sentence(long_text, max_chars=100)
    assert len(out) == 100


def test_first_sentence_empty() -> None:
    assert cluster._first_sentence("") == ""


# ---- render papers block ---------------------------------------------------


def test_render_papers_block_substitutes_loop() -> None:
    template = "Header\n{{#each PAPERS}}{{this.title}}{{/each}}\nFooter"
    out = cluster._render_papers_block(template, "1. A\n2. B")
    assert out == "Header\n1. A\n2. B\nFooter"


def test_render_papers_block_preserves_special_chars() -> None:
    # Title contains regex-replacement characters; lambda guards against them.
    template = "{{#each PAPERS}}{{x}}{{/each}}"
    content = r"backslash \1 and \g<group>"
    out = cluster._render_papers_block(template, content)
    assert out == content


# ---- no/few papers ---------------------------------------------------------


def test_no_papers_returns_zeros(conn: sqlite3.Connection) -> None:
    result = cluster.cluster(
        conn, week="2024-W01", since="2024-01-01", prompts_dir=REPO_PROMPTS_DIR
    )
    assert result.clusters_formed == 0
    assert result.cost_usd == 0.0


def test_few_papers_skips_clustering(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(0, seed=i)
        )
    # Spy: no calls happen when too-few
    called: list[Any] = []
    monkeypatch.setattr(cluster, "_hdbscan_labels", lambda *_, **__: called.append("hdb"))
    monkeypatch.setattr(llm, "call_direct", lambda **_: called.append("llm"))

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=5,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    assert result.clusters_formed == 0
    assert called == []


# ---- window filter --------------------------------------------------------


def test_only_papers_in_window_clustered(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _insert_with_embedding(
            conn,
            f"2401.in{i:03d}",
            vec=_near_axis(0, seed=i),
            published_at="2024-01-15T00:00:00Z",
        )
    for i in range(3):
        _insert_with_embedding(
            conn,
            f"2312.out{i:03d}",
            vec=_near_axis(1, seed=i + 100),
            published_at="2023-12-01T00:00:00Z",
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([0, 0, 0], dtype=np.int_),
    )
    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_label())

    result = cluster.cluster(
        conn,
        week="2024-W03",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    assert result.papers_clustered == 3

    # Verify paper_clusters only contains the in-window arxiv_ids
    mapped = conn.execute(
        "SELECT arxiv_id FROM paper_clusters WHERE week = '2024-W03'"
    ).fetchall()
    arxiv_ids = {str(r["arxiv_id"]) for r in mapped}
    assert all(aid.startswith("2401.in") for aid in arxiv_ids)


# ---- DB write shape -------------------------------------------------------


def test_clusters_persisted_with_correct_shape(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(4):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(i % 2, seed=i)
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([0, 0, 1, 1], dtype=np.int_),
    )

    sonnet_calls = 0

    def fake_direct(**_: Any) -> llm.CallResult:
        nonlocal sonnet_calls
        sonnet_calls += 1
        return _success_label(label=f"Topic {sonnet_calls}")

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    assert result.clusters_formed == 2
    assert result.label_succeeded == 2
    assert result.label_failed == 0
    assert sonnet_calls == 2

    rows = conn.execute(
        "SELECT * FROM clusters WHERE week = '2024-W01' ORDER BY cluster_id"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0]["label"] == "Topic 1"
    assert rows[0]["paper_count"] == 2
    assert json.loads(rows[0]["subtopics"]) == ["retrieval", "code-generation"]

    mapped = conn.execute(
        "SELECT * FROM paper_clusters WHERE week = '2024-W01'"
    ).fetchall()
    assert len(mapped) == 4


# ---- schema failure fallback ----------------------------------------------


def test_schema_failure_uses_fallback_label(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(0, seed=i)
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([0, 0, 0], dtype=np.int_),
    )
    monkeypatch.setattr(llm, "call_direct", lambda **_: _schema_failed())

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    assert result.clusters_formed == 1
    assert result.label_failed == 1
    assert result.label_succeeded == 0

    row = conn.execute(
        "SELECT label, description FROM clusters WHERE week='2024-W01'"
    ).fetchone()
    assert row["label"] == cluster.FALLBACK_LABEL
    assert row["description"] == cluster.FALLBACK_DESCRIPTION


# ---- noise exclusion ------------------------------------------------------


def test_noise_papers_excluded_from_paper_clusters(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(6):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(0, seed=i)
        )

    # 4 in cluster 0, 2 are noise (-1)
    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array(
            [0, 0, 0, 0, -1, -1], dtype=np.int_
        ),
    )
    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_label())

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    assert result.papers_clustered == 4
    assert result.noise_count == 2
    assert result.clusters_formed == 1

    mapped = conn.execute(
        "SELECT arxiv_id FROM paper_clusters WHERE week = '2024-W01'"
    ).fetchall()
    assert len(mapped) == 4


def test_all_noise_skips_sonnet_and_db_writes(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(6):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(0, seed=i)
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([-1] * 6, dtype=np.int_),
    )

    call_count = 0

    def fake_direct(**_: Any) -> llm.CallResult:
        nonlocal call_count
        call_count += 1
        return _success_label()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    assert result.clusters_formed == 0
    assert call_count == 0

    count = conn.execute(
        "SELECT COUNT(*) AS c FROM clusters WHERE week = '2024-W01'"
    ).fetchone()["c"]
    assert int(count) == 0


# ---- idempotency ----------------------------------------------------------


def test_rerun_overwrites_previous_week(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(0, seed=i)
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([0, 0, 0], dtype=np.int_),
    )
    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_label())

    cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )

    count_clusters = conn.execute(
        "SELECT COUNT(*) AS c FROM clusters WHERE week = '2024-W01'"
    ).fetchone()["c"]
    count_pc = conn.execute(
        "SELECT COUNT(*) AS c FROM paper_clusters WHERE week = '2024-W01'"
    ).fetchone()["c"]
    assert int(count_clusters) == 1
    assert int(count_pc) == 3


# ---- prompt plumbing ------------------------------------------------------


def test_cluster_size_placeholder_filled(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(3):
        _insert_with_embedding(
            conn,
            f"2401.{i:05d}",
            vec=_near_axis(0, seed=i),
            title=f"Paper {i}",
            abstract="First sentence here. Second sentence.",
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([0, 0, 0], dtype=np.int_),
    )

    captured: list[str] = []

    def fake_direct(**kw: Any) -> llm.CallResult:
        captured.append(kw["user"])
        return _success_label()

    monkeypatch.setattr(llm, "call_direct", fake_direct)

    cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    user_msg = captured[0]
    assert "Cluster size: 3 papers." in user_msg
    assert "Paper 0" in user_msg
    assert "First sentence here." in user_msg
    # The Handlebars loop syntax must be substituted, not present
    assert "{{#each PAPERS}}" not in user_msg


def test_cost_accumulates_across_clusters(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(4):
        _insert_with_embedding(
            conn, f"2401.{i:05d}", vec=_near_axis(i % 2, seed=i)
        )

    monkeypatch.setattr(
        cluster,
        "_hdbscan_labels",
        lambda emb, *, min_cluster_size: np.array([0, 0, 1, 1], dtype=np.int_),
    )
    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_label())

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=2,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    # Two clusters, each cost 0.003
    assert result.cost_usd == pytest.approx(0.006)


# ---- real HDBSCAN integration --------------------------------------------


def test_real_hdbscan_finds_obvious_clusters(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One end-to-end test against real sklearn HDBSCAN with synthetic
    embeddings that should form 3 tight clusters in 3 different axes."""
    for axis in range(3):
        for i in range(5):
            _insert_with_embedding(
                conn,
                f"2401.{axis}{i:04d}",
                vec=_near_axis(axis, noise=0.01, seed=axis * 100 + i),
            )

    monkeypatch.setattr(llm, "call_direct", lambda **_: _success_label())

    result = cluster.cluster(
        conn,
        week="2024-W01",
        since="2024-01-01",
        min_cluster_size=3,
        prompts_dir=REPO_PROMPTS_DIR,
    )
    # Allow some slack: HDBSCAN with min_cluster_size=3 should find at
    # least 2 clear clusters from this synthetic data.
    assert result.clusters_formed >= 2
    assert result.papers_clustered >= 10
