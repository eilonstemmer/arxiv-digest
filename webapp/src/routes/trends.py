"""Route: GET /trends — shows latest weekly cluster_trends + trend_narrative."""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import HTMLResponse

from src.deps import DBDep, TemplatesDep

log = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/trends", response_class=HTMLResponse)
def trends(
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> HTMLResponse:
    digest_row = conn.execute(
        "SELECT * FROM digests WHERE kind = 'weekly' ORDER BY generated_at DESC LIMIT 1"
    ).fetchone()

    trend_narrative: dict[str, Any] | None = None
    cluster_trends_list: list[dict[str, Any]] = []
    digest_id: str | None = None
    generated_at: str | None = None

    if digest_row:
        digest_id = str(digest_row["digest_id"])
        generated_at = str(digest_row["generated_at"])

        raw_narrative = digest_row["trend_narrative"]
        if raw_narrative:
            try:
                trend_narrative = json.loads(raw_narrative)
            except (json.JSONDecodeError, TypeError):
                log.warning("trends.bad_narrative_json", digest_id=digest_id)

        trend_rows = conn.execute(
            """
            SELECT ct.*, c.label, c.description, c.paper_count
            FROM cluster_trends ct
            JOIN clusters c ON c.week = ct.week AND c.cluster_id = ct.cluster_id
            WHERE ct.week = ?
            ORDER BY c.paper_count DESC
            """,
            (digest_id,),
        ).fetchall()
        cluster_trends_list = [dict(r) for r in trend_rows]

    return templates.TemplateResponse(
        request,
        "trends.html.j2",
        {
            "digest_id": digest_id,
            "generated_at": generated_at,
            "trend_narrative": trend_narrative,
            "cluster_trends": cluster_trends_list,
        },
    )
