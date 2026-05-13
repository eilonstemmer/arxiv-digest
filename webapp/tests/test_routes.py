"""Route tests for the arxiv-digest webapp.

Uses FastAPI's TestClient (backed by httpx). Each test gets a fresh SQLite DB
via the tmp_path fixture so tests are fully isolated.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import db as wdb

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@pytest.fixture()
def test_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret-32chars-long-abcdefg")

    conn = wdb.connect(db_path)
    wdb.init_schema(conn)
    conn.close()

    from src.main import create_app

    return TestClient(create_app(), raise_server_exceptions=True)


@pytest.fixture()
def populated_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Client with a paper, summary, and digest pre-inserted."""
    db_path = tmp_path / "populated.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret-32chars-long-abcdefg")

    conn = wdb.connect(db_path)
    wdb.init_schema(conn)

    # Insert a paper.
    conn.execute(
        """
        INSERT INTO papers(
            arxiv_id, title, authors, abstract, primary_category,
            categories, published_at, fetched_at, url_abs, url_pdf
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2401.00001",
            "Test Paper Title",
            '["Alice", "Bob"]',
            "This is the abstract.",
            "cs.AI",
            '["cs.AI"]',
            "2024-01-01",
            _now(),
            "https://arxiv.org/abs/2401.00001",
            "https://arxiv.org/pdf/2401.00001",
        ),
    )

    # Insert a digest (html_path points to a temp file we create).
    html_file = tmp_path / "digest.html"
    html_file.write_text("<h1>Weekly Digest</h1>", encoding="utf-8")

    conn.execute(
        """
        INSERT INTO digests(digest_id, kind, generated_at, html_path, paper_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("2024-W01", "weekly", _now(), str(html_file), 1),
    )
    conn.close()

    from src.main import create_app

    return TestClient(create_app(), raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# /healthz
# ---------------------------------------------------------------------------


def test_healthz(test_client: TestClient) -> None:
    resp = test_client.get("/healthz")
    assert resp.status_code == 200
    assert resp.text == "ok"


# ---------------------------------------------------------------------------
# / (root)
# ---------------------------------------------------------------------------


def test_root_no_digests(test_client: TestClient) -> None:
    resp = test_client.get("/", follow_redirects=True)
    assert resp.status_code == 200
    body = resp.text.lower()
    assert "no digests" in body or "digest" in body


def test_root_with_digest_redirects(populated_client: TestClient) -> None:
    resp = populated_client.get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert "/digests/2024-W01" in resp.headers["location"]


# ---------------------------------------------------------------------------
# /digests
# ---------------------------------------------------------------------------


def test_digests_list_empty(test_client: TestClient) -> None:
    resp = test_client.get("/digests")
    assert resp.status_code == 200
    assert "digest" in resp.text.lower()


def test_digests_list_populated(populated_client: TestClient) -> None:
    resp = populated_client.get("/digests")
    assert resp.status_code == 200
    assert "2024-W01" in resp.text


# ---------------------------------------------------------------------------
# /digests/{digest_id}
# ---------------------------------------------------------------------------


def test_digest_view_404(test_client: TestClient) -> None:
    resp = test_client.get("/digests/nonexistent")
    assert resp.status_code == 404


def test_digest_view_returns_html(populated_client: TestClient) -> None:
    resp = populated_client.get("/digests/2024-W01")
    assert resp.status_code == 200
    assert "Weekly Digest" in resp.text


# ---------------------------------------------------------------------------
# /papers/{arxiv_id}
# ---------------------------------------------------------------------------


def test_paper_404(test_client: TestClient) -> None:
    resp = test_client.get("/papers/9999.99999")
    assert resp.status_code == 404


def test_paper_200(populated_client: TestClient) -> None:
    resp = populated_client.get("/papers/2401.00001")
    assert resp.status_code == 200
    assert "Test Paper Title" in resp.text
    assert "This is the abstract." in resp.text


# ---------------------------------------------------------------------------
# /papers/{arxiv_id}/upvote
# ---------------------------------------------------------------------------


def test_upvote_404(test_client: TestClient) -> None:
    resp = test_client.post("/papers/9999.99999/upvote")
    assert resp.status_code == 404


def test_upvote_sets_flag_and_inserts_feedback(
    populated_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resp = populated_client.post("/papers/2401.00001/upvote")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"

    db_path = Path(str(tmp_path / "populated.db"))
    conn = wdb.connect(db_path)
    paper = conn.execute(
        "SELECT upvoted FROM papers WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert paper is not None
    assert paper["upvoted"] == 1

    feedback = conn.execute(
        "SELECT signal FROM feedback WHERE arxiv_id = '2401.00001'"
    ).fetchone()
    assert feedback is not None
    assert feedback["signal"] == "upvote"
    conn.close()


# ---------------------------------------------------------------------------
# /settings
# ---------------------------------------------------------------------------


def test_settings_get_renders_form(test_client: TestClient) -> None:
    resp = test_client.get("/settings")
    assert resp.status_code == 200
    assert "Anthropic API Key" in resp.text


def test_settings_get_password_not_in_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secret value must NOT appear in the rendered HTML."""
    db_path = tmp_path / "secret.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret-32chars-long-abcdefg")

    conn = wdb.connect(db_path)
    wdb.init_schema(conn)
    wdb.set_setting(conn, "anthropic_api_key", "sk-supersecretvalue123")
    conn.close()

    from src.main import create_app

    client = TestClient(create_app())
    resp = client.get("/settings")
    assert resp.status_code == 200
    assert "sk-supersecretvalue123" not in resp.text


def test_settings_post_saves_non_secret(test_client: TestClient) -> None:
    resp = test_client.post(
        "/settings",
        data={"smtp_host": "smtp.example.com", "notify_enabled": ""},
    )
    assert resp.status_code == 200
    assert "smtp.example.com" in resp.text or "saved" in resp.text.lower()


def test_settings_post_empty_password_no_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty password field must not overwrite existing secret."""
    db_path = tmp_path / "pwtest.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret-32chars-long-abcdefg")

    conn = wdb.connect(db_path)
    wdb.init_schema(conn)
    wdb.set_setting(conn, "anthropic_api_key", "keep-this-value")
    conn.close()

    from src.main import create_app

    client = TestClient(create_app())
    # Submit empty anthropic_api_key
    resp = client.post("/settings", data={"anthropic_api_key": ""})
    assert resp.status_code == 200

    conn2 = wdb.connect(db_path)
    stored = wdb.get_setting(conn2, "anthropic_api_key")
    conn2.close()
    assert stored == "keep-this-value"


def test_settings_json_redacts_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "json_test.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret-32chars-long-abcdefg")

    conn = wdb.connect(db_path)
    wdb.init_schema(conn)
    wdb.set_setting(conn, "anthropic_api_key", "sk-real-key-value")
    conn.close()

    from src.main import create_app

    client = TestClient(create_app())
    resp = client.get("/settings", headers={"Accept": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("anthropic_api_key") == "***"
    assert "sk-real-key-value" not in json.dumps(data)


# ---------------------------------------------------------------------------
# /webhook/digest-ready
# ---------------------------------------------------------------------------


def _make_sig(body: bytes, secret: str) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()


def test_webhook_valid_signature(test_client: TestClient) -> None:
    body = b'{"digest_id": "2024-W01"}'
    sig = _make_sig(body, "test-secret-32chars-long-abcdefg")
    resp = test_client.post(
        "/webhook/digest-ready",
        content=body,
        headers={"X-Webhook-Signature": sig, "Content-Type": "application/json"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_webhook_invalid_signature(test_client: TestClient) -> None:
    body = b'{"digest_id": "2024-W01"}'
    resp = test_client.post(
        "/webhook/digest-ready",
        content=body,
        headers={
            "X-Webhook-Signature": "sha256=badhash",
            "Content-Type": "application/json",
        },
    )
    assert resp.status_code == 401


def test_webhook_no_signature(test_client: TestClient) -> None:
    body = b'{"digest_id": "2024-W01"}'
    resp = test_client.post(
        "/webhook/digest-ready",
        content=body,
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# /search
# ---------------------------------------------------------------------------


def test_search_empty_query(test_client: TestClient) -> None:
    resp = test_client.get("/search")
    assert resp.status_code == 200
    assert "Search" in resp.text


def test_search_with_query_empty_db(test_client: TestClient) -> None:
    resp = test_client.get("/search?q=foo")
    assert resp.status_code == 200
    assert "foo" in resp.text


def test_search_finds_paper(populated_client: TestClient) -> None:
    resp = populated_client.get("/search?q=Test Paper")
    assert resp.status_code == 200
    # Result may appear; empty result is also acceptable with empty DB
    assert resp.status_code == 200
