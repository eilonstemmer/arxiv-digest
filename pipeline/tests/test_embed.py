"""Tests for src.embed: serialize round-trip, cosine, batch embedding.

The autouse fake-model fixture monkeypatches `embed._model` so no test
ever triggers the real sentence_transformers import (heavy: torch +
transformers). Deterministic vectors come from MD5-seeded RNG keyed by
the text, so the same input always produces the same embedding.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from src import db, embed


class _FakeModel:
    """Deterministic stand-in for SentenceTransformer.

    Maps each text to a `(EMBEDDING_DIM,)` float32 vector via MD5-seeded
    RNG, then L2-normalizes. Same text always yields the same vector.
    """

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
            vec = rng.standard_normal(embed.EMBEDDING_DIM).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            out[i] = vec
        return out


@pytest.fixture(autouse=True)
def _fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embed, "_model", _FakeModel())


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


def _insert_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    title: str = "T",
    abstract: str = "A",
    published_at: str = "2024-01-01T00:00:00Z",
) -> None:
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, authors, abstract, primary_category,
                           categories, published_at, fetched_at, url_abs, url_pdf)
        VALUES (?, ?, '[]', ?, 'cs.AI', '["cs.AI"]', ?,
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


# ---- serialize / deserialize ----------------------------------------------


def test_serialize_round_trip() -> None:
    vec = np.arange(embed.EMBEDDING_DIM, dtype=np.float32)
    blob = embed.serialize(vec)
    assert len(blob) == embed.EMBEDDING_DIM * 4
    back = embed.deserialize(blob)
    np.testing.assert_array_equal(back, vec)


def test_serialize_casts_non_float32() -> None:
    vec64 = np.ones(embed.EMBEDDING_DIM, dtype=np.float64)
    blob = embed.serialize(vec64.astype(np.float32))
    assert len(blob) == embed.EMBEDDING_DIM * 4
    back = embed.deserialize(blob)
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, 1.0)


def test_serialize_rejects_wrong_shape() -> None:
    bad = np.zeros(10, dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        embed.serialize(bad)


def test_deserialize_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="bytes"):
        embed.deserialize(b"x" * 100)


def test_deserialize_returns_writable() -> None:
    vec = np.ones(embed.EMBEDDING_DIM, dtype=np.float32)
    back = embed.deserialize(embed.serialize(vec))
    back[0] = 99.0  # must not raise (frombuffer view would be read-only)
    assert back[0] == 99.0


# ---- embed_text / embed_texts ---------------------------------------------


def test_embed_text_shape_and_dtype() -> None:
    vec = embed.embed_text("hello")
    assert vec.shape == (embed.EMBEDDING_DIM,)
    assert vec.dtype == np.float32


def test_embed_texts_batch_shape() -> None:
    arr = embed.embed_texts(["a", "b", "c"])
    assert arr.shape == (3, embed.EMBEDDING_DIM)


def test_embed_texts_empty_input() -> None:
    arr = embed.embed_texts([])
    assert arr.shape == (0, embed.EMBEDDING_DIM)


def test_embed_text_deterministic() -> None:
    v1 = embed.embed_text("the same text")
    v2 = embed.embed_text("the same text")
    np.testing.assert_array_equal(v1, v2)


def test_embed_different_inputs_different_outputs() -> None:
    v1 = embed.embed_text("first")
    v2 = embed.embed_text("second")
    assert not np.array_equal(v1, v2)


# ---- paper_text -----------------------------------------------------------


def test_paper_text_concatenates() -> None:
    assert embed.paper_text("Hello", "World") == "Hello World"


def test_paper_text_strips_outer_whitespace() -> None:
    assert embed.paper_text("  Title  ", "  Abstract  ") == "Title Abstract"


def test_paper_text_empty_title() -> None:
    assert embed.paper_text("", "Abstract") == "Abstract"


# ---- cosine_similarity ----------------------------------------------------


def test_cosine_similarity_identical_vectors() -> None:
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert embed.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert embed.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([-1.0, 0.0], dtype=np.float32)
    assert embed.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector() -> None:
    zero = np.zeros(3, dtype=np.float32)
    other = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert embed.cosine_similarity(zero, other) == 0.0
    assert embed.cosine_similarity(other, zero) == 0.0


def test_cosine_similarity_arbitrary_magnitude() -> None:
    # 2*v should yield 1.0 with v (cosine is magnitude-invariant)
    a = np.array([3.0, 4.0], dtype=np.float32)
    b = np.array([6.0, 8.0], dtype=np.float32)
    assert embed.cosine_similarity(a, b) == pytest.approx(1.0)


# ---- cosine_similarity_batch ----------------------------------------------


def test_cosine_similarity_batch_basic() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    sims = embed.cosine_similarity_batch(q, corpus)
    np.testing.assert_allclose(sims, [1.0, 0.0, -1.0], atol=1e-6)


def test_cosine_similarity_batch_returns_float32() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array([[1.0, 0.0]], dtype=np.float32)
    sims = embed.cosine_similarity_batch(q, corpus)
    assert sims.dtype == np.float32


def test_cosine_similarity_batch_empty_corpus() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    sims = embed.cosine_similarity_batch(q, np.empty((0, 2), dtype=np.float32))
    assert sims.shape == (0,)


def test_cosine_similarity_batch_zero_query() -> None:
    q = np.zeros(2, dtype=np.float32)
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    sims = embed.cosine_similarity_batch(q, corpus)
    np.testing.assert_array_equal(sims, [0.0, 0.0])


def test_cosine_similarity_batch_zero_corpus_row() -> None:
    q = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    sims = embed.cosine_similarity_batch(q, corpus)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == 0.0


# ---- load_embedding / store_embedding -------------------------------------


def test_load_embedding_paper_missing(conn: sqlite3.Connection) -> None:
    assert embed.load_embedding(conn, "9999.99999") is None


def test_load_embedding_paper_without_embedding(conn: sqlite3.Connection) -> None:
    _insert_paper(conn, arxiv_id="2401.00001")
    assert embed.load_embedding(conn, "2401.00001") is None


def test_store_and_load_round_trip(conn: sqlite3.Connection) -> None:
    _insert_paper(conn, arxiv_id="2401.00001")
    vec = embed.embed_text("hello world")
    embed.store_embedding(conn, "2401.00001", vec)
    loaded = embed.load_embedding(conn, "2401.00001")
    assert loaded is not None
    np.testing.assert_array_equal(loaded, vec)


def test_store_embedding_records_model_name(conn: sqlite3.Connection) -> None:
    _insert_paper(conn, arxiv_id="2401.00001")
    embed.store_embedding(conn, "2401.00001", embed.embed_text("x"))
    row = conn.execute(
        "SELECT embedding_model FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert row["embedding_model"] == embed.MODEL_NAME


# ---- embed_papers_batch ---------------------------------------------------


def test_embed_papers_batch_no_rows(conn: sqlite3.Connection) -> None:
    assert embed.embed_papers_batch(conn) == 0


def test_embed_papers_batch_fills_all_unembedded(conn: sqlite3.Connection) -> None:
    for i in range(3):
        _insert_paper(
            conn,
            arxiv_id=f"2401.0000{i + 1}",
            title=f"Paper {i}",
            abstract=f"Abstract {i}",
        )
    assert embed.embed_papers_batch(conn, batch_size=2) == 3
    for i in range(3):
        loaded = embed.load_embedding(conn, f"2401.0000{i + 1}")
        assert loaded is not None
        assert loaded.shape == (embed.EMBEDDING_DIM,)


def test_embed_papers_batch_skips_already_embedded(conn: sqlite3.Connection) -> None:
    _insert_paper(conn, arxiv_id="2401.00001")
    _insert_paper(conn, arxiv_id="2401.00002")
    embed.embed_papers_batch(conn)
    # Second run finds nothing to do.
    assert embed.embed_papers_batch(conn) == 0


def test_embed_papers_batch_respects_limit(conn: sqlite3.Connection) -> None:
    for i in range(5):
        _insert_paper(
            conn,
            arxiv_id=f"2401.0000{i + 1}",
            title=f"P{i}",
            abstract=f"A{i}",
        )
    assert embed.embed_papers_batch(conn, batch_size=2, limit=3) == 3
    remaining = conn.execute(
        "SELECT COUNT(*) AS c FROM papers WHERE embedding IS NULL"
    ).fetchone()["c"]
    assert int(remaining) == 2


def test_embed_papers_batch_limit_zero(conn: sqlite3.Connection) -> None:
    _insert_paper(conn, arxiv_id="2401.00001")
    assert embed.embed_papers_batch(conn, limit=0) == 0


def test_embed_papers_batch_deterministic_per_text(conn: sqlite3.Connection) -> None:
    # Same title+abstract -> same embedding via the fake model.
    _insert_paper(conn, arxiv_id="2401.00001", title="X", abstract="Y")
    _insert_paper(conn, arxiv_id="2401.00002", title="X", abstract="Y")
    embed.embed_papers_batch(conn)
    v1 = embed.load_embedding(conn, "2401.00001")
    v2 = embed.load_embedding(conn, "2401.00002")
    assert v1 is not None
    assert v2 is not None
    np.testing.assert_array_equal(v1, v2)


def test_embed_papers_batch_orders_by_published_at_desc(
    conn: sqlite3.Connection,
) -> None:
    # The fetch order shouldn't matter to correctness, but the batch SQL
    # specifies ORDER BY published_at DESC -- regression-guard it.
    _insert_paper(
        conn, arxiv_id="2401.00001", title="old", published_at="2020-01-01T00:00:00Z"
    )
    _insert_paper(
        conn, arxiv_id="2401.00002", title="new", published_at="2024-01-01T00:00:00Z"
    )
    # limit=1 -> we should embed the newest first.
    embed.embed_papers_batch(conn, limit=1)
    assert embed.load_embedding(conn, "2401.00002") is not None
    assert embed.load_embedding(conn, "2401.00001") is None
