"""Cluster stage: HDBSCAN over MiniLM embeddings + Sonnet cluster labels.

For each weekly run, papers with `published_at >= since` are clustered
via HDBSCAN. Each cluster is labeled by a direct Sonnet call using
`prompts/cluster_label.md`. Outputs persist to `clusters` (one row per
cluster_id) and `paper_clusters` (one row per paper-in-cluster).

Sklearn's HDBSCAN is imported inside `_hdbscan_labels` so tests that
monkeypatch the labels function don't pay the sklearn import cost.

On Sonnet schema failure (after llm.call_direct's own one-shot retry),
we fall back to label='Mixed Topics' and a one-line description so the
cluster still lands in the DB and can be referenced by the trend stage.

Re-running for the same week is idempotent: the function deletes
existing `clusters` and `paper_clusters` rows for that week before
inserting fresh ones.
"""

from __future__ import annotations

import dataclasses
import json
import re
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import structlog

from . import db, embed, llm, settings

log = structlog.get_logger(__name__)

DEFAULT_MIN_CLUSTER_SIZE = 5
REPRESENTATIVES_PER_CLUSTER = 15  # medoid + 14 neighbors
FALLBACK_LABEL = "Mixed Topics"
FALLBACK_DESCRIPTION = "Cluster could not be auto-labeled."


@dataclasses.dataclass(frozen=True)
class ClusterResult:
    """Outcome counters for one cluster stage run."""

    week: str
    papers_clustered: int   # papers actually grouped (excludes HDBSCAN noise)
    noise_count: int        # papers HDBSCAN labeled -1
    clusters_formed: int
    label_succeeded: int
    label_failed: int       # Sonnet returned invalid JSON twice; fallback applied
    cost_usd: float


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _first_sentence(text: str, *, max_chars: int = 300) -> str:
    """Best-effort first-sentence extraction for the cluster-label prompt.

    Returns the text up to the first `.!?` if it fits within `max_chars`;
    otherwise the first `max_chars` chars. Empty strings pass through.
    """
    text = text.strip()
    if not text:
        return ""
    match = re.match(r"[^.!?]+[.!?]", text)
    if match and len(match.group()) <= max_chars:
        return match.group()
    return text[:max_chars].rstrip()


def _render_papers_block(template: str, papers_text: str) -> str:
    """Replace the prompt's `{{#each PAPERS}}...{{/each}}` block with the
    rendered papers list.

    The Handlebars-style loop is documentation in the prompt file. We
    substitute the literal rendered text via a lambda so backslashes and
    `\\g<...>` patterns inside titles or abstracts aren't reinterpreted
    by re.sub.
    """
    pattern = re.compile(r"\{\{#each PAPERS\}\}.*?\{\{/each\}\}", re.DOTALL)
    return pattern.sub(lambda _: papers_text, template, count=1)


def _hdbscan_labels(
    embeddings: npt.NDArray[np.float32], *, min_cluster_size: int
) -> npt.NDArray[np.int_]:
    """Run HDBSCAN on the embedding matrix. Returns one cluster label per
    row; -1 indicates noise.

    Lazy import: sklearn is only loaded when this function actually runs,
    so tests that monkeypatch us never trigger the import.
    """
    from sklearn.cluster import HDBSCAN

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
    labels: npt.NDArray[np.int_] = hdb.fit_predict(embeddings).astype(np.int_)
    return labels


def _label_cluster(
    *,
    representative_rows: list[sqlite3.Row],
    prompt: llm.Prompt,
    api_key: str,
    model: str,
) -> tuple[dict[str, Any], float]:
    """Call Sonnet to label one cluster.

    Returns `(label_data, cost_usd)`. On any failure path (API error,
    schema validation after retry), returns the fallback label so the
    cluster row still lands in the DB.
    """
    parts: list[str] = []
    for i, row in enumerate(representative_rows):
        title = str(row["title"]).strip()
        first = _first_sentence(str(row["abstract"]))
        parts.append(f"{i + 1}. **{title}**\n   {first}")
    papers_text = "\n".join(parts)

    user_msg = _render_papers_block(prompt.user_template, papers_text)
    user_msg = llm.fill_template(
        user_msg, {"CLUSTER_SIZE": str(len(representative_rows))}
    )

    result = llm.call_direct(
        model=model,
        system=prompt.system_template,
        user=user_msg,
        schema=prompt.schema,
        api_key=api_key,
    )
    if result.status == "success" and result.data is not None:
        return result.data, result.cost_usd

    log.warning(
        "cluster.label.fallback",
        error=result.error,
        status=result.status,
    )
    return (
        {
            "label": FALLBACK_LABEL,
            "description": FALLBACK_DESCRIPTION,
            "subtopics": [],
        },
        result.cost_usd,
    )


def cluster(
    conn: sqlite3.Connection,
    *,
    week: str,
    since: str,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    prompts_dir: Path | str | None = None,
) -> ClusterResult:
    """Cluster + label papers for one weekly digest.

    Inputs:
      * `week`: ISO week label, e.g. '2026-W20'. Used as the `clusters.week`
        primary-key component.
      * `since`: ISO datetime string; only papers with `published_at >= since`
        are considered.
      * `min_cluster_size`: HDBSCAN parameter; below this, papers are noise.

    Re-running for the same `week` overwrites previous `clusters` and
    `paper_clusters` rows for that week.
    """
    rows = conn.execute(
        """
        SELECT arxiv_id, title, abstract, embedding
          FROM papers
         WHERE published_at >= ? AND embedding IS NOT NULL
         ORDER BY published_at DESC
        """,
        (since,),
    ).fetchall()

    if len(rows) < min_cluster_size:
        log.info(
            "cluster.too_few_papers",
            count=len(rows),
            min_cluster_size=min_cluster_size,
        )
        return ClusterResult(
            week=week,
            papers_clustered=0,
            noise_count=0,
            clusters_formed=0,
            label_succeeded=0,
            label_failed=0,
            cost_usd=0.0,
        )

    embeddings = np.stack(
        [embed.deserialize(bytes(r["embedding"])) for r in rows]
    )
    arxiv_ids = [str(r["arxiv_id"]) for r in rows]

    labels = _hdbscan_labels(embeddings, min_cluster_size=min_cluster_size)

    by_cluster: dict[int, list[int]] = defaultdict(list)
    noise_count = 0
    for i, label in enumerate(labels):
        lbl = int(label)
        if lbl == -1:
            noise_count += 1
        else:
            by_cluster[lbl].append(i)

    clusters_formed = len(by_cluster)
    if clusters_formed == 0:
        log.info("cluster.all_noise", noise=noise_count, total=len(rows))
        return ClusterResult(
            week=week,
            papers_clustered=0,
            noise_count=noise_count,
            clusters_formed=0,
            label_succeeded=0,
            label_failed=0,
            cost_usd=0.0,
        )

    api_key = settings.anthropic_api_key(conn)
    model = settings.summarize_model(conn)  # Sonnet, same model class
    prompt = llm.load_prompt("cluster_label", prompts_dir=prompts_dir)

    total_cost = 0.0
    label_succeeded = 0
    label_failed = 0
    payloads: list[tuple[int, list[int], dict[str, Any]]] = []

    for cluster_id, indices in sorted(by_cluster.items()):
        cluster_embeddings = embeddings[indices]
        centroid = cluster_embeddings.mean(axis=0).astype(np.float32)
        sims = embed.cosine_similarity_batch(centroid, cluster_embeddings)
        top_indices = np.argsort(-sims)[:REPRESENTATIVES_PER_CLUSTER]
        representative_rows = [rows[indices[int(ti)]] for ti in top_indices]

        label_data, cost = _label_cluster(
            representative_rows=representative_rows,
            prompt=prompt,
            api_key=api_key,
            model=model,
        )
        total_cost += cost
        if (
            label_data["label"] == FALLBACK_LABEL
            and label_data["description"] == FALLBACK_DESCRIPTION
        ):
            label_failed += 1
        else:
            label_succeeded += 1
        payloads.append((cluster_id, indices, label_data))

    with db.transaction(conn):
        conn.execute(
            "DELETE FROM paper_clusters WHERE week = ?", (week,)
        )
        conn.execute("DELETE FROM clusters WHERE week = ?", (week,))
        for cluster_id, indices, label_data in payloads:
            conn.execute(
                """
                INSERT INTO clusters(week, cluster_id, label, description,
                                     subtopics, paper_count)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    week,
                    cluster_id,
                    label_data["label"],
                    label_data["description"],
                    json.dumps(label_data["subtopics"]),
                    len(indices),
                ),
            )
            for idx in indices:
                conn.execute(
                    """
                    INSERT INTO paper_clusters(arxiv_id, week, cluster_id)
                    VALUES (?, ?, ?)
                    """,
                    (arxiv_ids[idx], week, cluster_id),
                )

    papers_clustered = sum(len(idxs) for _, idxs, _ in payloads)
    log.info(
        "cluster.complete",
        week=week,
        clusters_formed=clusters_formed,
        papers_clustered=papers_clustered,
        noise_count=noise_count,
        label_succeeded=label_succeeded,
        label_failed=label_failed,
        cost_usd=round(total_cost, 4),
    )
    return ClusterResult(
        week=week,
        papers_clustered=papers_clustered,
        noise_count=noise_count,
        clusters_formed=clusters_formed,
        label_succeeded=label_succeeded,
        label_failed=label_failed,
        cost_usd=total_cost,
    )
