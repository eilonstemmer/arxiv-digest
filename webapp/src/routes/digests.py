"""Routes: GET /, GET /digests, GET /digests/{digest_id}, GET /healthz."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response

from src.deps import DBDep, TemplatesDep

log = structlog.get_logger(__name__)

router = APIRouter()


def _fetch_digest(conn: sqlite3.Connection, digest_id: str) -> sqlite3.Row | None:
    result: sqlite3.Row | None = conn.execute(
        "SELECT * FROM digests WHERE digest_id = ?",
        (digest_id,),
    ).fetchone()
    return result


def _latest_weekly(conn: sqlite3.Connection) -> sqlite3.Row | None:
    result: sqlite3.Row | None = conn.execute(
        "SELECT * FROM digests WHERE kind = 'weekly' ORDER BY generated_at DESC LIMIT 1"
    ).fetchone()
    return result


@router.get("/healthz", response_class=PlainTextResponse)
def healthz() -> str:
    return "ok"


@router.get("/", response_class=HTMLResponse, response_model=None)
def index(
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> Response:
    row = _latest_weekly(conn)
    if row is None:
        return templates.TemplateResponse(
            request,
            "digest_list.html.j2",
            {"digests": [], "no_digests": True},
        )
    return RedirectResponse(url=f"/digests/{row['digest_id']}", status_code=302)


@router.get("/digests", response_class=HTMLResponse)
def list_digests(
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> HTMLResponse:
    rows = conn.execute(
        "SELECT * FROM digests ORDER BY generated_at DESC"
    ).fetchall()
    digests_list: list[dict[str, Any]] = [dict(r) for r in rows]
    return templates.TemplateResponse(
        request,
        "digest_list.html.j2",
        {"digests": digests_list, "no_digests": False},
    )


@router.get("/digests/{digest_id}")
def view_digest(
    digest_id: str,
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> HTMLResponse:
    row = _fetch_digest(conn, digest_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Digest '{digest_id}' not found")

    html_path = Path(row["html_path"])
    if html_path.exists():
        digest_html: str | None = html_path.read_text(encoding="utf-8")
        return templates.TemplateResponse(
            request,
            "digest_view.html.j2",
            {"digest": dict(row), "digest_html": digest_html},
        )

    log.warning("digest.html_missing", digest_id=digest_id, html_path=str(html_path))
    return templates.TemplateResponse(
        request,
        "digest_view.html.j2",
        {"digest": dict(row), "digest_html": None},
        status_code=200,
    )
