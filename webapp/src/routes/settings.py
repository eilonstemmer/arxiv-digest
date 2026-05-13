"""Routes: GET /settings, POST /settings."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from src import db as wdb
from src.deps import DBDep, TemplatesDep
from src.settings_meta import SECRET_KEYS, SETTINGS, SETTINGS_BY_KEY

log = structlog.get_logger(__name__)

router = APIRouter()


def _db_path_str() -> str:
    return os.environ.get("DB_PATH", "/data/digest.db")


def _build_settings_context(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build the template context dict for the settings form."""
    current = wdb.list_settings(conn, redact_secrets=False)

    last_run_row = conn.execute(
        "SELECT * FROM pipeline_runs WHERE status = 'success' ORDER BY finished_at DESC LIMIT 1"
    ).fetchone()
    last_run: dict[str, Any] | None = dict(last_run_row) if last_run_row else None

    cost_row = conn.execute(
        """
        SELECT COALESCE(SUM(cost_usd), 0.0) AS total
        FROM pipeline_runs
        WHERE started_at >= date('now', 'weekday 1', '-7 days')
          AND status = 'success'
        """
    ).fetchone()
    cost_this_week: float = float(cost_row["total"]) if cost_row else 0.0

    db_path = Path(_db_path_str())
    try:
        db_size_bytes = db_path.stat().st_size
    except OSError:
        db_size_bytes = 0

    return {
        "settings_list": SETTINGS,
        "settings_by_key": SETTINGS_BY_KEY,
        "secret_keys": SECRET_KEYS,
        "current": current,
        "last_run": last_run,
        "cost_this_week": cost_this_week,
        "db_size_bytes": db_size_bytes,
        "db_path": str(db_path),
    }


@router.get("/settings", response_model=None)
def get_settings(
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> Response:
    accept = request.headers.get("accept", "")
    if "application/json" in accept:
        data = wdb.list_settings(conn, redact_secrets=True)
        return JSONResponse(content=data)

    ctx = _build_settings_context(conn)
    ctx["saved"] = False
    return templates.TemplateResponse(request, "settings.html.j2", ctx)


@router.post("/settings", response_class=HTMLResponse)
async def post_settings(
    request: Request,
    conn: DBDep,
    templates: TemplatesDep,
) -> HTMLResponse:
    form_data = await request.form()

    with wdb.transaction(conn):
        for meta in SETTINGS:
            raw_value = form_data.get(meta.key)
            value: str | None = str(raw_value) if raw_value is not None else None

            if meta.key in SECRET_KEYS:
                if not value:
                    continue
            else:
                if value is None:
                    value = ""
                if meta.field_type == "bool":
                    value = "true" if raw_value else "false"

            wdb.set_setting(conn, meta.key, value)

    log.info("settings.saved")

    ctx = _build_settings_context(conn)
    ctx["saved"] = True
    return templates.TemplateResponse(
        request,
        "partials/settings_form.html.j2",
        ctx,
    )
