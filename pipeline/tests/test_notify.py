"""Tests for pipeline/src/notify.py.

Covers:
  - sign_payload determinism and sensitivity
  - send_webhook: happy path, retry on 5xx, fail-fast on 4xx, exhausted retries
  - Correct X-Webhook-Signature and Content-Type headers
  - send_email: STARTTLS branch, auth branch, non-auth branch, SMTPException
  - notify_for_digest: reads digest row, calls webhook, skips/sends email,
    updates sent_at, correct NotifyResult shapes
"""

from __future__ import annotations

import smtplib
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src import db, notify

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


def _insert_digest(
    conn: sqlite3.Connection,
    digest_id: str = "2026-W20",
    kind: str = "weekly",
    html_path: str = "/data/digests/2026-W20.html",
    generated_at: str = "2026-05-13T07:00:00+00:00",
) -> None:
    conn.execute(
        """INSERT INTO digests (digest_id, kind, generated_at, html_path, paper_count)
           VALUES (?, ?, ?, ?, ?)""",
        (digest_id, kind, generated_at, html_path, 42),
    )
    conn.commit()


def _set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    from datetime import UTC, datetime

    now = datetime.now(UTC).isoformat(timespec="seconds")
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, now),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# sign_payload
# ---------------------------------------------------------------------------


def test_sign_payload_same_input_same_output() -> None:
    payload = {"digest_id": "2026-W20", "kind": "weekly"}
    body1, sig1 = notify.sign_payload(payload, "secret")
    body2, sig2 = notify.sign_payload(payload, "secret")
    assert body1 == body2
    assert sig1 == sig2


def test_sign_payload_different_secret_differs() -> None:
    payload = {"digest_id": "2026-W20", "kind": "weekly"}
    _, sig1 = notify.sign_payload(payload, "secret-a")
    _, sig2 = notify.sign_payload(payload, "secret-b")
    assert sig1 != sig2


def test_sign_payload_different_payload_differs() -> None:
    _, sig1 = notify.sign_payload({"x": 1}, "secret")
    _, sig2 = notify.sign_payload({"x": 2}, "secret")
    assert sig1 != sig2


def test_sign_payload_signature_has_sha256_prefix() -> None:
    _, sig = notify.sign_payload({"x": 1}, "secret")
    assert sig.startswith("sha256=")
    # hex part is 64 characters (SHA-256)
    assert len(sig) == len("sha256=") + 64


def test_sign_payload_sort_keys_makes_order_independent() -> None:
    """Dict ordering must not affect the signature — sort_keys=True."""
    payload_ab: dict[str, Any] = {"a": 1, "b": 2}
    payload_ba: dict[str, Any] = {"b": 2, "a": 1}
    _, sig_ab = notify.sign_payload(payload_ab, "secret")
    _, sig_ba = notify.sign_payload(payload_ba, "secret")
    assert sig_ab == sig_ba


def test_sign_payload_body_is_compact_json() -> None:
    body, _ = notify.sign_payload({"a": 1}, "secret")
    # compact separators: no extra spaces
    assert b" " not in body
    assert body == b'{"a":1}'


# ---------------------------------------------------------------------------
# Fake urllib machinery
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal file-like object returned by urlopen."""

    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def read(self) -> bytes:
        return b""


def _make_http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://example.com",
        code=code,
        msg=f"Fake {code}",
        hdrs=MagicMock(),  # type: ignore[arg-type]
        fp=None,
    )


# ---------------------------------------------------------------------------
# send_webhook
# ---------------------------------------------------------------------------


def test_send_webhook_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        calls.append(req)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = notify.send_webhook(
        "http://webapp", "mysecret", {"digest_id": "2026-W20"}, backoff_seconds=0
    )
    assert ok is True
    assert err is None
    assert len(calls) == 1


def test_send_webhook_signature_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[urllib.request.Request] = []

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        captured.append(req)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    payload = {"digest_id": "2026-W20", "kind": "weekly"}
    notify.send_webhook("http://webapp", "s3cr3t", payload, backoff_seconds=0)

    req = captured[0]
    sig_header = req.get_header("X-webhook-signature")
    assert sig_header is not None
    assert sig_header.startswith("sha256=")
    # Verify it matches what sign_payload would produce
    _, expected_sig = notify.sign_payload(payload, "s3cr3t")
    assert sig_header == expected_sig


def test_send_webhook_content_type_header(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[urllib.request.Request] = []

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        captured.append(req)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.send_webhook(
        "http://webapp", "s3cr3t", {"x": 1}, backoff_seconds=0
    )
    req = captured[0]
    assert req.get_header("Content-type") == "application/json"


def test_send_webhook_url_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[urllib.request.Request] = []

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        captured.append(req)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.send_webhook("http://webapp:8080", "s", {}, backoff_seconds=0)
    assert captured[0].full_url == "http://webapp:8080/webhook/digest-ready"


def test_send_webhook_retry_on_5xx_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two 5xx failures then one success — 3 attempts total."""
    attempt_count = 0

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise _make_http_error(503)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = notify.send_webhook(
        "http://webapp", "s", {}, max_attempts=3, backoff_seconds=0
    )
    assert ok is True
    assert err is None
    assert attempt_count == 3


def test_send_webhook_fail_fast_on_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """4xx: only one attempt made, no retry."""
    attempt_count = 0

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        nonlocal attempt_count
        attempt_count += 1
        raise _make_http_error(422)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = notify.send_webhook(
        "http://webapp", "s", {}, max_attempts=3, backoff_seconds=0
    )
    assert ok is False
    assert err is not None
    assert "422" in err
    assert attempt_count == 1


def test_send_webhook_exhausts_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    attempt_count = 0

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        nonlocal attempt_count
        attempt_count += 1
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = notify.send_webhook(
        "http://webapp", "s", {}, max_attempts=3, backoff_seconds=0
    )
    assert ok is False
    assert err is not None
    assert "3" in err  # mentions attempt count
    assert attempt_count == 3


def test_send_webhook_5xx_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    attempt_count = 0

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        nonlocal attempt_count
        attempt_count += 1
        raise _make_http_error(500)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    ok, err = notify.send_webhook(
        "http://webapp", "s", {}, max_attempts=3, backoff_seconds=0
    )
    assert ok is False
    assert err is not None
    assert attempt_count == 3


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------


class FakeSMTP:
    """Test double for smtplib.SMTP."""

    starttls_called: bool
    login_called: bool
    login_args: tuple[str, str]
    sent_messages: list[Any]
    quit_called: bool

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        FakeSMTP.starttls_called = False
        FakeSMTP.login_called = False
        FakeSMTP.login_args = ("", "")
        FakeSMTP.sent_messages = []
        FakeSMTP.quit_called = False

    def starttls(self) -> None:
        FakeSMTP.starttls_called = True

    def login(self, username: str, password: str) -> None:
        FakeSMTP.login_called = True
        FakeSMTP.login_args = (username, password)

    def send_message(self, msg: Any) -> None:
        FakeSMTP.sent_messages.append(msg)

    def quit(self) -> None:
        FakeSMTP.quit_called = True


def test_send_email_starttls_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    ok, err = notify.send_email(
        html_body="<h1>hi</h1>",
        subject="Test Subject",
        recipient="user@example.com",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="user@example.com",
        smtp_password="pass",
        use_tls=True,
    )
    assert ok is True
    assert err is None
    assert FakeSMTP.starttls_called is True
    assert FakeSMTP.login_called is True
    assert len(FakeSMTP.sent_messages) == 1
    assert FakeSMTP.quit_called is True


def test_send_email_no_tls_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    ok, err = notify.send_email(
        html_body="<h1>hi</h1>",
        subject="Test Subject",
        recipient="user@example.com",
        smtp_host="smtp.example.com",
        smtp_port=25,
        smtp_username=None,
        smtp_password=None,
        use_tls=False,
    )
    assert ok is True
    assert err is None
    assert FakeSMTP.starttls_called is False
    assert FakeSMTP.login_called is False
    assert len(FakeSMTP.sent_messages) == 1


def test_send_email_auth_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    ok, _err = notify.send_email(
        html_body="<h1>hi</h1>",
        subject="Test",
        recipient="user@example.com",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username="myuser",
        smtp_password="mypass",
        use_tls=False,
    )
    assert ok is True
    assert FakeSMTP.login_called is True
    assert FakeSMTP.login_args == ("myuser", "mypass")


def test_send_email_non_auth_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """No username/password → login is not called."""
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    ok, _err = notify.send_email(
        html_body="<h1>hi</h1>",
        subject="Test",
        recipient="user@example.com",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username=None,
        smtp_password=None,
        use_tls=False,
    )
    assert ok is True
    assert FakeSMTP.login_called is False


def test_send_email_smtp_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenSMTP(FakeSMTP):
        def send_message(self, msg: Any) -> None:
            raise smtplib.SMTPException("connection refused")

    monkeypatch.setattr(smtplib, "SMTP", BrokenSMTP)
    ok, err = notify.send_email(
        html_body="<h1>hi</h1>",
        subject="Test",
        recipient="user@example.com",
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_username=None,
        smtp_password=None,
        use_tls=False,
    )
    assert ok is False
    assert err is not None
    assert "connection refused" in err


def test_send_email_os_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class OSErrorSMTP(FakeSMTP):
        def __init__(self, host: str, port: int) -> None:
            raise OSError("name resolution failed")

    monkeypatch.setattr(smtplib, "SMTP", OSErrorSMTP)
    ok, err = notify.send_email(
        html_body="<h1>hi</h1>",
        subject="Test",
        recipient="user@example.com",
        smtp_host="bad-host",
        smtp_port=587,
        smtp_username=None,
        smtp_password=None,
        use_tls=False,
    )
    assert ok is False
    assert err is not None


# ---------------------------------------------------------------------------
# notify_for_digest — end-to-end
# ---------------------------------------------------------------------------


def _make_html_file(tmp_path: Path, digest_id: str = "2026-W20") -> str:
    p = tmp_path / f"{digest_id}.html"
    p.write_text("<html><body>digest</body></html>", encoding="utf-8")
    return str(p)


def test_notify_for_digest_reads_row_and_calls_webhook(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    assert result.digest_id == "2026-W20"
    assert result.webhook_sent is True
    assert result.webhook_error is None


def test_notify_for_digest_notify_disabled_skips_email(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)
    # notify_enabled defaults to False — no extra setting needed

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse(200))

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    assert result.email_attempted is False
    assert result.email_sent is False
    assert result.email_error is None


def test_notify_for_digest_notify_enabled_sends_email_and_updates_sent_at(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)
    _set_setting(conn, "notify_enabled", "true")
    _set_setting(conn, "smtp_host", "smtp.example.com")
    _set_setting(conn, "notify_email", "user@example.com")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse(200))
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    assert result.email_attempted is True
    assert result.email_sent is True
    assert result.email_error is None

    # sent_at must be updated in DB
    row = conn.execute(
        "SELECT sent_at FROM digests WHERE digest_id = '2026-W20'"
    ).fetchone()
    assert row["sent_at"] is not None


def test_notify_for_digest_smtp_host_missing_skips_email(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)
    _set_setting(conn, "notify_enabled", "true")
    # smtp_host NOT set → email_attempted should be False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse(200))

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    assert result.email_attempted is False


def test_notify_for_digest_notify_email_missing_skips_email(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)
    _set_setting(conn, "notify_enabled", "true")
    _set_setting(conn, "smtp_host", "smtp.example.com")
    # notify_email NOT set → email_attempted should be False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse(200))

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    assert result.email_attempted is False


def test_notify_for_digest_webhook_failure_still_returns_result(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)

    def fake_urlopen(req: Any, **kwargs: Any) -> _FakeResponse:
        raise urllib.error.URLError("network down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
        # keep backoff_seconds default — we need to override send_webhook instead
    )
    # webhook failed but we still get a result
    assert result.webhook_sent is False
    assert result.webhook_error is not None


def test_notify_for_digest_missing_digest_raises(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(ValueError, match="not found"):
        notify.notify_for_digest(
            conn,
            digest_id="nonexistent",
            webapp_url="http://webapp",
            webhook_secret="secret",
        )


def test_notify_for_digest_email_subject_contains_digest_id(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)
    _set_setting(conn, "notify_enabled", "true")
    _set_setting(conn, "smtp_host", "smtp.example.com")
    _set_setting(conn, "notify_email", "user@example.com")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse(200))
    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)

    notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    msg = FakeSMTP.sent_messages[0]
    assert "2026-W20" in msg["Subject"]
    assert msg["Subject"] == "arxiv-digest: 2026-W20"


def test_notify_for_digest_sent_at_not_updated_on_email_failure(
    conn: sqlite3.Connection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    html_path = _make_html_file(tmp_path)
    _insert_digest(conn, html_path=html_path)
    _set_setting(conn, "notify_enabled", "true")
    _set_setting(conn, "smtp_host", "smtp.example.com")
    _set_setting(conn, "notify_email", "user@example.com")

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: _FakeResponse(200))

    class FailingSMTP(FakeSMTP):
        def send_message(self, msg: Any) -> None:
            raise smtplib.SMTPException("auth failed")

    monkeypatch.setattr(smtplib, "SMTP", FailingSMTP)

    result = notify.notify_for_digest(
        conn,
        digest_id="2026-W20",
        webapp_url="http://webapp",
        webhook_secret="secret",
    )
    assert result.email_sent is False
    assert result.email_error is not None

    row = conn.execute(
        "SELECT sent_at FROM digests WHERE digest_id = '2026-W20'"
    ).fetchone()
    assert row["sent_at"] is None
