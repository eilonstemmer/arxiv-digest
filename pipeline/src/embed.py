"""sentence-transformers wrapper with BLOB storage and cosine similarity helpers.

Model: `sentence-transformers/all-MiniLM-L6-v2`. 384-dim float32, ~22M
parameters, CPU-friendly. Module-scope singleton, lazily loaded on first
use so tests that monkeypatch `embed._model` never trigger the heavy
torch + transformers import.

Embeddings are stored as raw float32 bytes in `papers.embedding` (1536
bytes each). The model name is recorded in `papers.embedding_model` so a
future model swap can identify stale rows.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import structlog

from . import db

log = structlog.get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DEFAULT_BATCH_SIZE = 64


class EmbedModel(Protocol):
    """Structural type for the parts of SentenceTransformer we use.

    Tests substitute a fake; production uses the real
    sentence_transformers.SentenceTransformer.
    """

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int = ...,
        show_progress_bar: bool = ...,
        convert_to_numpy: bool = ...,
    ) -> npt.NDArray[np.float32]: ...


_model: EmbedModel | None = None


def get_model() -> EmbedModel:
    """Return the cached SentenceTransformer model, loading on first call.

    The import of `sentence_transformers` is deliberately lazy: tests
    monkeypatch `embed._model` directly with a fake, avoiding the heavy
    torch + transformers import at test collection time.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log.info("embed.model.load", model=MODEL_NAME)
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> npt.NDArray[np.float32]:
    """Embed a single string. Returns shape `(EMBEDDING_DIM,)` float32."""
    vec: npt.NDArray[np.float32] = embed_texts([text])[0]
    return vec


def embed_texts(texts: list[str]) -> npt.NDArray[np.float32]:
    """Embed a batch of strings. Returns shape `(n, EMBEDDING_DIM)` float32.

    Always prefer this over calling `embed_text` in a loop -- the
    underlying model is dramatically faster batched than one-at-a-time.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
    model = get_model()
    raw: Any = model.encode(  # SentenceTransformer's return type is loose
        texts,
        batch_size=DEFAULT_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return np.asarray(raw, dtype=np.float32)


def serialize(vec: npt.NDArray[np.float32]) -> bytes:
    """Pack a `(EMBEDDING_DIM,)` float32 array into bytes for SQLite BLOB."""
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(
            f"expected shape ({EMBEDDING_DIM},), got {vec.shape}"
        )
    if vec.dtype != np.float32:
        vec = vec.astype(np.float32)
    return vec.tobytes()


def deserialize(blob: bytes) -> npt.NDArray[np.float32]:
    """Unpack bytes back into a `(EMBEDDING_DIM,)` float32 array."""
    expected = EMBEDDING_DIM * 4
    if len(blob) != expected:
        raise ValueError(
            f"expected {expected} bytes ({EMBEDDING_DIM} x float32), got {len(blob)}"
        )
    # frombuffer returns a non-writable view; copy() makes it mutable.
    arr: npt.NDArray[np.float32] = np.frombuffer(blob, dtype=np.float32).copy()
    return arr


def paper_text(title: str, abstract: str) -> str:
    """Canonical text for embedding a paper: `title + ' ' + abstract`.

    Used by both the embed step (storing the paper's vector) and the
    triage prefilter (rebuilding the same vector to score against the
    profile embedding). Keeping the concatenation in one place ensures
    the two stages remain consistent.
    """
    return f"{title.strip()} {abstract.strip()}".strip()


def cosine_similarity(
    a: npt.NDArray[np.float32], b: npt.NDArray[np.float32]
) -> float:
    """Cosine similarity between two 1-D vectors. Zero vectors map to 0.0."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_similarity_batch(
    query: npt.NDArray[np.float32],
    corpus: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Cosine similarity between query `(D,)` and each row of corpus
    `(N, D)`. Returns `(N,)` float32. Empty corpus returns shape `(0,)`.
    Zero query or zero-norm rows yield 0.0 at those positions."""
    if corpus.size == 0:
        return np.empty(0, dtype=np.float32)
    qnorm = float(np.linalg.norm(query))
    cnorms = np.linalg.norm(corpus, axis=1)
    if qnorm == 0.0:
        return np.zeros(corpus.shape[0], dtype=np.float32)
    safe_cnorms = np.where(cnorms == 0.0, 1.0, cnorms)
    sims: npt.NDArray[np.float32] = (
        corpus @ query / (safe_cnorms * qnorm)
    ).astype(np.float32)
    sims[cnorms == 0.0] = 0.0
    return sims


def load_embedding(
    conn: sqlite3.Connection, arxiv_id: str
) -> npt.NDArray[np.float32] | None:
    """Load a paper's embedding from the DB. Returns None if the paper
    does not exist or has no stored embedding."""
    row = conn.execute(
        "SELECT embedding FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    if row is None or row["embedding"] is None:
        return None
    return deserialize(bytes(row["embedding"]))


def store_embedding(
    conn: sqlite3.Connection,
    arxiv_id: str,
    vec: npt.NDArray[np.float32],
    *,
    model_name: str = MODEL_NAME,
) -> None:
    """Manually store one paper's embedding. `embed_papers_batch` is the
    normal entry point; this is for tests and one-off reprocessing."""
    conn.execute(
        "UPDATE papers SET embedding = ?, embedding_model = ? WHERE arxiv_id = ?",
        (serialize(vec), model_name, arxiv_id),
    )


def embed_papers_batch(
    conn: sqlite3.Connection,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> int:
    """Embed every paper that lacks an embedding. Returns count embedded.

    Reads up to `batch_size` unembedded rows at a time, embeds them in
    one model call (much faster than per-row), writes back in a single
    transaction per batch. `limit` caps total papers processed in one
    invocation (default unlimited).
    """
    total = 0
    while True:
        if limit is not None and total >= limit:
            break
        fetch_size = batch_size
        if limit is not None:
            fetch_size = min(batch_size, limit - total)
        rows = conn.execute(
            """
            SELECT arxiv_id, title, abstract
            FROM papers
            WHERE embedding IS NULL
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (fetch_size,),
        ).fetchall()
        if not rows:
            break

        texts = [paper_text(str(r["title"]), str(r["abstract"])) for r in rows]
        vectors = embed_texts(texts)

        with db.transaction(conn):
            for row, vec in zip(rows, vectors, strict=True):
                conn.execute(
                    "UPDATE papers SET embedding = ?, embedding_model = ? "
                    "WHERE arxiv_id = ?",
                    (serialize(vec), MODEL_NAME, str(row["arxiv_id"])),
                )
        total += len(rows)
        log.info("embed.batch.complete", count=len(rows), total=total)
    return total
