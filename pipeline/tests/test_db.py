"""Tests for src.db: schema, connection, transactions, settings, FTS triggers."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import db


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


def _insert_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str = "2401.00001",
    title: str = "Adaptive Tool-Use Memory for Long-Horizon Agents",
    abstract: str = "We introduce an agent memory system for tool-use.",
) -> None:
    conn.execute(
        """
        INSERT INTO papers(arxiv_id, title, authors, abstract, primary_category,
                           categories, published_at, fetched_at, url_abs, url_pdf)
        VALUES (?, ?, '["A. Author"]', ?, 'cs.AI', '["cs.AI"]',
                '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00',
                ?, ?)
        """,
        (
            arxiv_id,
            title,
            abstract,
            f"https://arxiv.org/abs/{arxiv_id}",
            f"https://arxiv.org/pdf/{arxiv_id}",
        ),
    )


def test_connect_creates_parent_dir(tmp_path: Path) -> None:
    nested = tmp_path / "deeper" / "still" / "test.db"
    c = db.connect(nested)
    assert nested.parent.exists()
    c.close()


def test_init_schema_is_idempotent(tmp_path: Path) -> None:
    c = db.connect(tmp_path / "x.db")
    db.init_schema(c)
    db.init_schema(c)
    assert db.current_schema_version(c) == db.SCHEMA_VERSION
    c.close()


def test_pragmas_applied(conn: sqlite3.Connection) -> None:
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert fk == 1
    assert str(journal).lower() == "wal"


def test_all_expected_tables_exist(conn: sqlite3.Connection) -> None:
    expected = {
        "settings",
        "papers",
        "summaries",
        "clusters",
        "paper_clusters",
        "cluster_trends",
        "digests",
        "feedback",
        "pipeline_runs",
        "schema_version",
        "papers_fts",
    }
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    ).fetchall()
    actual = {str(r["name"]) for r in rows}
    missing = expected - actual
    assert not missing, f"missing tables: {missing}"


def test_papers_indexes_exist(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='papers'"
    ).fetchall()
    names = {str(r["name"]) for r in rows}
    expected = {
        "idx_papers_published",
        "idx_papers_relevance",
        "idx_papers_prefilter",
        "idx_papers_category",
    }
    assert expected.issubset(names), f"missing indexes: {expected - names}"


def test_fts_triggers_keep_index_in_sync(conn: sqlite3.Connection) -> None:
    # Original title contains "Adaptive" (unique to the title) and the
    # abstract contains "tool-use". After UPDATE we replace the title with
    # one containing "Quantization" but leave the abstract intact.
    _insert_paper(conn)

    matches = lambda term: [  # noqa: E731
        str(r["arxiv_id"])
        for r in conn.execute(
            "SELECT arxiv_id FROM papers_fts WHERE papers_fts MATCH ?",
            (term,),
        )
    ]

    assert "2401.00001" in matches("memory")
    assert "2401.00001" in matches("adaptive")
    assert "2401.00001" not in matches("quantization")

    conn.execute(
        "UPDATE papers SET title = ? WHERE arxiv_id = ?",
        ("Replaced Title About Quantization", "2401.00001"),
    )

    assert "2401.00001" in matches("quantization")
    # "adaptive" was unique to the old title -- the AU trigger must have
    # removed the stale index entry.
    assert "2401.00001" not in matches("adaptive")
    # Abstract was not changed; abstract-only terms still match.
    assert "2401.00001" in matches("memory")

    conn.execute("DELETE FROM papers WHERE arxiv_id = ?", ("2401.00001",))
    assert "2401.00001" not in matches("quantization")
    assert "2401.00001" not in matches("memory")


def test_get_setting_absent_returns_none(conn: sqlite3.Connection) -> None:
    assert db.get_setting(conn, "nope") is None


def test_set_and_get_setting(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "digest_top_n", "80")
    assert db.get_setting(conn, "digest_top_n") == "80"


def test_set_setting_upserts(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "digest_top_n", "80")
    db.set_setting(conn, "digest_top_n", "100")
    assert db.get_setting(conn, "digest_top_n") == "100"
    count = conn.execute(
        "SELECT COUNT(*) AS c FROM settings WHERE key = 'digest_top_n'"
    ).fetchone()["c"]
    assert int(count) == 1


def test_secret_keys_default_to_is_secret(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "anthropic_api_key", "fake-key-not-real")
    row = conn.execute(
        "SELECT is_secret FROM settings WHERE key = 'anthropic_api_key'"
    ).fetchone()
    assert int(row["is_secret"]) == 1


def test_non_secret_key_defaults_to_not_secret(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "digest_top_n", "80")
    row = conn.execute(
        "SELECT is_secret FROM settings WHERE key = 'digest_top_n'"
    ).fetchone()
    assert int(row["is_secret"]) == 0


def test_explicit_is_secret_override(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "smtp_host", "smtp.example.com", is_secret=True)
    row = conn.execute(
        "SELECT is_secret FROM settings WHERE key = 'smtp_host'"
    ).fetchone()
    assert int(row["is_secret"]) == 1


def test_delete_setting(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "k", "v")
    db.delete_setting(conn, "k")
    assert db.get_setting(conn, "k") is None


def test_delete_absent_setting_is_silent(conn: sqlite3.Connection) -> None:
    db.delete_setting(conn, "never_set")


def test_list_settings_redacts_secrets_by_default(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "anthropic_api_key", "real-looking-secret")
    db.set_setting(conn, "digest_top_n", "80")
    out = db.list_settings(conn)
    assert out["anthropic_api_key"] == "***"
    assert out["digest_top_n"] == "80"


def test_list_settings_unredacted(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "anthropic_api_key", "real-looking-secret")
    out = db.list_settings(conn, redact_secrets=False)
    assert out["anthropic_api_key"] == "real-looking-secret"


def test_list_settings_empty_secret_passes_through(conn: sqlite3.Connection) -> None:
    # Empty string is "explicitly cleared" — not a secret to redact.
    db.set_setting(conn, "smtp_password", "")
    out = db.list_settings(conn)
    assert out["smtp_password"] == ""


def test_transaction_rolls_back_on_exception(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "k", "before")
    with pytest.raises(RuntimeError, match="boom"), db.transaction(conn):
        db.set_setting(conn, "k", "during")
        raise RuntimeError("boom")
    assert db.get_setting(conn, "k") == "before"


def test_transaction_commits_on_success(conn: sqlite3.Connection) -> None:
    with db.transaction(conn):
        db.set_setting(conn, "k", "committed")
    assert db.get_setting(conn, "k") == "committed"


def test_migrate_on_fresh_db_initializes(tmp_path: Path) -> None:
    c = db.connect(tmp_path / "fresh.db")
    version = db.migrate(c)
    assert version == db.SCHEMA_VERSION
    c.close()


def test_migrate_on_current_db_is_noop(conn: sqlite3.Connection) -> None:
    version = db.migrate(conn)
    assert version == db.SCHEMA_VERSION


def test_foreign_keys_enforced(conn: sqlite3.Connection) -> None:
    # summaries.arxiv_id REFERENCES papers(arxiv_id). Insert without a
    # matching paper must fail.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO summaries(arxiv_id, problem, method, key_result,
                                  why_it_matters, cross_domain_hooks,
                                  novelty_signal, tags, generated_at, model)
            VALUES ('9999.99999', 'p', 'm', 'r', 'w', '[]', 'incremental',
                    '["x", "y"]', '2024-01-01', 'claude-sonnet-4-6')
            """
        )
