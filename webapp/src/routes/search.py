"""Route: GET /search?q=...

v1: FTS5 full-text search only. Semantic search (cosine over embeddings)
is deferred because the webapp deliberately avoids the heavy
sentence-transformers dependency. To add in v2: embed the query with
MiniLM, load paper embeddings from the DB BLOB column, compute cosine
similarity, merge and deduplicate with FTS hits.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from src.deps import DBDep, TemplatesDep

log = structlog.get_logger(__name__)

router = APIRouter()

_MAX_RESULTS = 50


@router.get("/search", response_class=HTMLResponse)
def search(
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
    q: str = "",
) -> HTMLResponse:
    results: list[dict[str, Any]] = []
    query = q.strip()

    if query:
        try:
            rows = conn.execute(
                """
                SELECT p.arxiv_id, p.title, p.authors, p.primary_category,
                       p.published_at, p.relevance_score, p.url_abs
                FROM papers_fts f
                JOIN papers p ON p.arxiv_id = f.arxiv_id
                WHERE papers_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (query, _MAX_RESULTS),
            ).fetchall()
            results = [dict(r) for r in rows]
        except Exception as exc:
            log.warning("search.fts_error", error=str(exc))
            results = []

    return templates.TemplateResponse(
        request,
        "search.html.j2",
        {"query": query, "results": results, "semantic_deferred": True},
    )
