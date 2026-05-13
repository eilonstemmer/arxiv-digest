"""Database connection and low-level settings helpers for the webapp.

Duplicated from pipeline/src/db.py — the webapp is a separate deployable and
must not import from the pipeline package. Kept intentionally minimal: only
the pieces the webapp actually needs.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

SCHEMA_VERSION = 1

# Secret keys: always redacted in API responses; always stored with is_secret=1.
SECRET_SETTINGS: frozenset[str] = frozenset(
    {
        "anthropic_api_key",
        "smtp_password",
    }
)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT,
  is_secret INTEGER DEFAULT 0,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
  arxiv_id          TEXT PRIMARY KEY,
  title             TEXT NOT NULL,
  authors           TEXT NOT NULL,
  abstract          TEXT NOT NULL,
  primary_category  TEXT NOT NULL,
  categories        TEXT NOT NULL,
  published_at      TEXT NOT NULL,
  updated_at        TEXT,
  fetched_at        TEXT NOT NULL,
  url_abs           TEXT NOT NULL,
  url_pdf           TEXT NOT NULL,
  embedding         BLOB,
  embedding_model   TEXT,
  prefilter_score   REAL,
  relevance_score   REAL,
  relevance_reason  TEXT,
  triaged_at        TEXT,
  summarized        INTEGER DEFAULT 0,
  upvoted           INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_papers_published  ON papers(published_at);
CREATE INDEX IF NOT EXISTS idx_papers_relevance  ON papers(relevance_score DESC);
CREATE INDEX IF NOT EXISTS idx_papers_prefilter  ON papers(prefilter_score DESC);
CREATE INDEX IF NOT EXISTS idx_papers_category   ON papers(primary_category);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
  arxiv_id UNINDEXED, title, abstract,
  content='papers', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS papers_fts_ai AFTER INSERT ON papers BEGIN
  INSERT INTO papers_fts(rowid, arxiv_id, title, abstract)
    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_fts_ad AFTER DELETE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, arxiv_id, title, abstract)
    VALUES ('delete', old.rowid, old.arxiv_id, old.title, old.abstract);
END;

CREATE TRIGGER IF NOT EXISTS papers_fts_au AFTER UPDATE ON papers BEGIN
  INSERT INTO papers_fts(papers_fts, rowid, arxiv_id, title, abstract)
    VALUES ('delete', old.rowid, old.arxiv_id, old.title, old.abstract);
  INSERT INTO papers_fts(rowid, arxiv_id, title, abstract)
    VALUES (new.rowid, new.arxiv_id, new.title, new.abstract);
END;

CREATE TABLE IF NOT EXISTS summaries (
  arxiv_id            TEXT PRIMARY KEY REFERENCES papers(arxiv_id),
  problem             TEXT NOT NULL,
  method              TEXT NOT NULL,
  key_result          TEXT NOT NULL,
  why_it_matters      TEXT NOT NULL,
  cross_domain_hooks  TEXT NOT NULL,
  novelty_signal      TEXT NOT NULL,
  tags                TEXT NOT NULL,
  generated_at        TEXT NOT NULL,
  model               TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS clusters (
  week         TEXT NOT NULL,
  cluster_id   INTEGER NOT NULL,
  label        TEXT NOT NULL,
  description  TEXT NOT NULL,
  subtopics    TEXT NOT NULL,
  paper_count  INTEGER NOT NULL,
  PRIMARY KEY (week, cluster_id)
);

CREATE TABLE IF NOT EXISTS paper_clusters (
  arxiv_id    TEXT NOT NULL REFERENCES papers(arxiv_id),
  week        TEXT NOT NULL,
  cluster_id  INTEGER NOT NULL,
  PRIMARY KEY (arxiv_id, week),
  FOREIGN KEY (week, cluster_id) REFERENCES clusters(week, cluster_id)
);

CREATE TABLE IF NOT EXISTS cluster_trends (
  week           TEXT NOT NULL,
  cluster_id     INTEGER NOT NULL,
  delta_vs_w1    REAL,
  delta_vs_w4    REAL,
  delta_vs_w12   REAL,
  delta_vs_w52   REAL,
  is_new         INTEGER DEFAULT 0,
  velocity_class TEXT,
  PRIMARY KEY (week, cluster_id)
);

CREATE TABLE IF NOT EXISTS digests (
  digest_id        TEXT PRIMARY KEY,
  kind             TEXT NOT NULL,
  generated_at     TEXT NOT NULL,
  html_path        TEXT NOT NULL,
  paper_count      INTEGER NOT NULL,
  trend_narrative  TEXT,
  sent_at          TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
  arxiv_id   TEXT NOT NULL REFERENCES papers(arxiv_id),
  signal     TEXT NOT NULL,
  notes      TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (arxiv_id, signal)
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
  run_id              TEXT PRIMARY KEY,
  kind                TEXT NOT NULL,
  started_at          TEXT NOT NULL,
  finished_at         TEXT,
  status              TEXT NOT NULL,
  papers_ingested     INTEGER,
  papers_prefiltered  INTEGER,
  papers_triaged      INTEGER,
  papers_summarized   INTEGER,
  cost_usd            REAL,
  error               TEXT
);

CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _set_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection with standard pragmas. Parent dir is created if needed."""
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _set_pragmas(conn)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """BEGIN / COMMIT block; ROLLBACK on exception."""
    conn.execute("BEGIN")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def init_schema(conn: sqlite3.Connection) -> None:
    """Create all tables, indexes, triggers, and FTS5 virtual table. Idempotent."""
    conn.executescript(SCHEMA_SQL)
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO schema_version(version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
    log.info("schema.initialized", version=SCHEMA_VERSION)


def get_setting(conn: sqlite3.Connection, key: str) -> str | None:
    """Return the raw stored value for key, or None if absent."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    if row is None:
        return None
    value = row["value"]
    return None if value is None else str(value)


def set_setting(
    conn: sqlite3.Connection,
    key: str,
    value: str | None,
    *,
    is_secret: bool | None = None,
) -> None:
    """Upsert a setting row. Never logs value — it may be a secret."""
    if is_secret is None:
        is_secret = key in SECRET_SETTINGS
    conn.execute(
        """
        INSERT INTO settings(key, value, is_secret, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
          value = excluded.value,
          is_secret = excluded.is_secret,
          updated_at = excluded.updated_at
        """,
        (key, value, 1 if is_secret else 0, _now_iso()),
    )
    log.info(
        "setting.written",
        key=key,
        has_value=value is not None and value != "",
        is_secret=is_secret,
    )


def list_settings(
    conn: sqlite3.Connection,
    *,
    redact_secrets: bool = True,
) -> dict[str, str | None]:
    """Return all settings as {key: value}. Secrets redacted to '***' when non-empty."""
    out: dict[str, str | None] = {}
    for row in conn.execute("SELECT key, value, is_secret FROM settings"):
        key = str(row["key"])
        raw = row["value"]
        value: str | None = None if raw is None else str(raw)
        is_secret_row = bool(row["is_secret"]) or key in SECRET_SETTINGS
        if redact_secrets and is_secret_row and value not in (None, ""):
            out[key] = "***"
        else:
            out[key] = value
    return out
