"""FastAPI dependency factories for DB and templates.

Extracted from main.py to break the circular import:
  main.py imports routes/* which import DBDep/TemplatesDep from main.py.

Routes should import from here, not from src.main.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Annotated

from fastapi import Depends
from fastapi.templating import Jinja2Templates

from src import db as wdb

_TEMPLATES_DIR = Path(__file__).parent / "templates"


def get_templates() -> Jinja2Templates:
    return Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _get_db_path() -> Path:
    raw = os.environ.get("DB_PATH", "/data/digest.db")
    return Path(raw)


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency: open a DB connection, init schema, yield, close."""
    conn = wdb.connect(_get_db_path())
    wdb.init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


DBDep = Annotated[sqlite3.Connection, Depends(get_db)]
TemplatesDep = Annotated[Jinja2Templates, Depends(get_templates)]
