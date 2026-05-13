"""Routes: GET /papers/{arxiv_id}, POST /papers/{arxiv_id}/upvote."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.deps import DBDep, TemplatesDep

log = structlog.get_logger(__name__)

router = APIRouter()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@router.get("/papers/{arxiv_id}", response_class=HTMLResponse)
def get_paper(
    arxiv_id: str,
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> HTMLResponse:
    paper_row = conn.execute(
        "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    if paper_row is None:
        raise HTTPException(status_code=404, detail=f"Paper '{arxiv_id}' not found")

    summary_row = conn.execute(
        "SELECT * FROM summaries WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()

    paper: dict[str, Any] = dict(paper_row)
    summary: dict[str, Any] | None = dict(summary_row) if summary_row else None

    return templates.TemplateResponse(
        request,
        "paper.html.j2",
        {"paper": paper, "summary": summary},
    )


@router.post("/papers/{arxiv_id}/upvote")
def upvote_paper(
    arxiv_id: str,
    conn: DBDep,
) -> JSONResponse:
    paper_row = conn.execute(
        "SELECT arxiv_id FROM papers WHERE arxiv_id = ?", (arxiv_id,)
    ).fetchone()
    if paper_row is None:
        raise HTTPException(status_code=404, detail=f"Paper '{arxiv_id}' not found")

    now = _now_iso()
    conn.execute(
        "UPDATE papers SET upvoted = 1 WHERE arxiv_id = ?", (arxiv_id,)
    )
    conn.execute(
        """
        INSERT INTO feedback(arxiv_id, signal, notes, created_at)
        VALUES (?, 'upvote', NULL, ?)
        ON CONFLICT(arxiv_id, signal) DO UPDATE SET created_at = excluded.created_at
        """,
        (arxiv_id, now),
    )
    log.info("paper.upvoted", arxiv_id=arxiv_id)
    return JSONResponse(
        content={"status": "ok", "arxiv_id": arxiv_id},
        headers={"HX-Trigger": "upvoteDone"},
    )
