"""Notification: signed webhook POST to webapp + optional SMTP email.

Step 11 of the pipeline (PROJECT_BRIEF.md §6):
  1. POST /webhook/digest-ready with HMAC-SHA256 signature.
  2. Optionally send HTML email via smtplib (STARTTLS or plain SMTP).
  3. UPDATE digests.sent_at on successful email send.

Only stdlib is used for HTTP and SMTP — no httpx, no requests.
Secrets (smtp_password, webhook_secret, HMAC values) are never logged.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
import smtplib
import sqlite3
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from email.message import EmailMessage
from typing import Any

import structlog

from . import settings

log = structlog.get_logger(__name__)

_WEBHOOK_USER_AGENT = "arxiv-digest-notify/0.1"


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NotifyResult:
    """Outcome of a single notify_for_digest call."""

    digest_id: str
    webhook_sent: bool
    webhook_error: str | None  # human-readable; None on success
    email_attempted: bool  # False when disabled / not configured
    email_sent: bool
    email_error: str | None


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


def sign_payload(payload: dict[str, Any], secret: str) -> tuple[bytes, str]:
    """Serialize *payload* to canonical JSON and compute HMAC-SHA256.

    Returns (body_bytes, "sha256=<hex>").  body_bytes is the exact byte
    string that was signed — caller must POST these bytes verbatim.
    ``sort_keys=True`` ensures dict ordering never changes the signature.
    """
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    mac = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return body, f"sha256={mac}"


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------


def send_webhook(
    webapp_url: str,
    webhook_secret: str,
    payload: dict[str, Any],
    *,
    max_attempts: int = 3,
    backoff_seconds: float = 5.0,
) -> tuple[bool, str | None]:
    """POST *payload* to ``<webapp_url>/webhook/digest-ready``.

    Returns ``(True, None)`` on success.  Retries up to *max_attempts* on
    HTTP 5xx or connection errors (with *backoff_seconds* sleep between
    attempts).  Fails immediately on HTTP 4xx — the webapp rejected us and
    retrying won't help.  Returns ``(False, error_message)`` on failure.

    Never logs webhook_secret or the HMAC value itself.
    """
    url = webapp_url.rstrip("/") + "/webhook/digest-ready"
    body, sig = sign_payload(payload, webhook_secret)

    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": _WEBHOOK_USER_AGENT,
            "X-Webhook-Signature": sig,
        },
    )

    last_error: str = "no attempts made"
    for attempt in range(1, max_attempts + 1):
        log.info(
            "notify.webhook.attempt",
            webhook_url=url,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req) as resp:
                status: int = resp.status
            elapsed = time.monotonic() - t0
            log.info(
                "notify.webhook.success",
                webhook_url=url,
                status=status,
                duration_s=round(elapsed, 3),
                attempt=attempt,
            )
            return True, None
        except urllib.error.HTTPError as exc:
            elapsed = time.monotonic() - t0
            status_code: int = exc.code
            log.warning(
                "notify.webhook.http_error",
                webhook_url=url,
                status=status_code,
                duration_s=round(elapsed, 3),
                attempt=attempt,
            )
            if 400 <= status_code < 500:
                # 4xx: fail immediately, retrying won't help.
                last_error = f"HTTP {status_code}: {exc.reason}"
                return False, last_error
            # 5xx: fall through to retry.
            last_error = f"HTTP {status_code}: {exc.reason}"
        except urllib.error.URLError as exc:
            elapsed = time.monotonic() - t0
            last_error = f"connection error: {exc.reason}"
            log.warning(
                "notify.webhook.url_error",
                webhook_url=url,
                error=last_error,
                duration_s=round(elapsed, 3),
                attempt=attempt,
            )
        except OSError as exc:
            elapsed = time.monotonic() - t0
            last_error = f"OS error: {exc}"
            log.warning(
                "notify.webhook.os_error",
                webhook_url=url,
                error=last_error,
                duration_s=round(elapsed, 3),
                attempt=attempt,
            )

        if attempt < max_attempts:
            log.info(
                "notify.webhook.backoff",
                webhook_url=url,
                backoff_seconds=backoff_seconds,
                attempt=attempt,
            )
            time.sleep(backoff_seconds)

    log.error(
        "notify.webhook.exhausted",
        webhook_url=url,
        attempts=max_attempts,
        last_error=last_error,
    )
    return False, f"failed after {max_attempts} attempts: {last_error}"


# ---------------------------------------------------------------------------
# SMTP email
# ---------------------------------------------------------------------------


def send_email(
    *,
    html_body: str,
    subject: str,
    recipient: str,
    smtp_host: str,
    smtp_port: int,
    smtp_username: str | None,
    smtp_password: str | None,
    use_tls: bool,
) -> tuple[bool, str | None]:
    """Send an HTML email via smtplib.

    Uses STARTTLS when *use_tls* is True.  Authenticates when both
    *smtp_username* and *smtp_password* are provided.  Never logs
    *smtp_password*.

    Returns ``(True, None)`` on success; ``(False, str(error))`` on failure.
    """
    msg = EmailMessage()
    msg["From"] = smtp_username or recipient
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content("This is an HTML digest. Please view in an HTML-capable client.")
    msg.add_alternative(html_body, subtype="html")

    t0 = time.monotonic()
    try:
        smtp: smtplib.SMTP = smtplib.SMTP(smtp_host, smtp_port)
        if use_tls:
            smtp.starttls()
        if smtp_username and smtp_password:
            smtp.login(smtp_username, smtp_password)
        smtp.send_message(msg)
        smtp.quit()
    except (smtplib.SMTPException, OSError) as exc:
        elapsed = time.monotonic() - t0
        error_str = str(exc)
        log.error(
            "notify.email.error",
            recipient=recipient,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            use_tls=use_tls,
            duration_s=round(elapsed, 3),
            error=error_str,
        )
        return False, error_str

    elapsed = time.monotonic() - t0
    log.info(
        "notify.email.sent",
        recipient=recipient,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        use_tls=use_tls,
        duration_s=round(elapsed, 3),
    )
    return True, None


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def notify_for_digest(
    conn: sqlite3.Connection,
    *,
    digest_id: str,
    webapp_url: str,
    webhook_secret: str,
) -> NotifyResult:
    """Full notify flow for one digest.

    1. Fetch the digest row from the DB.
    2. POST /webhook/digest-ready with HMAC-signed payload.
    3. If notify_enabled and SMTP is configured, send the HTML email and
       update ``digests.sent_at`` on success.

    Returns a :class:`NotifyResult` summarising both outcomes.
    """
    # ------------------------------------------------------------------
    # Load digest row
    # ------------------------------------------------------------------
    row = conn.execute(
        "SELECT digest_id, kind, html_path, generated_at FROM digests WHERE digest_id = ?",
        (digest_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"digest '{digest_id}' not found in database")

    # ------------------------------------------------------------------
    # Build and send webhook
    # ------------------------------------------------------------------
    payload: dict[str, Any] = {
        "digest_id": row["digest_id"],
        "kind": row["kind"],
        "html_path": row["html_path"],
        "generated_at": row["generated_at"],
    }

    webhook_sent, webhook_error = send_webhook(
        webapp_url,
        webhook_secret,
        payload,
    )

    # ------------------------------------------------------------------
    # Optional SMTP
    # ------------------------------------------------------------------
    email_attempted = False
    email_sent = False
    email_error: str | None = None

    notify_on = settings.notify_enabled(conn)
    if not notify_on:
        log.info("notify.email.skipped", reason="notify_enabled=false", digest_id=digest_id)
    else:
        host = settings.smtp_host(conn)
        recipient = settings.notify_email(conn)
        if not host or not recipient:
            log.warning(
                "notify.email.skipped",
                reason="smtp_host or notify_email not configured",
                digest_id=digest_id,
            )
        else:
            # Read the HTML file
            html_path: str = row["html_path"]
            try:
                with open(html_path, encoding="utf-8") as fh:
                    html_body = fh.read()
            except OSError as exc:
                email_attempted = True
                email_error = f"cannot read html_path '{html_path}': {exc}"
                log.error(
                    "notify.email.read_error",
                    html_path=html_path,
                    error=email_error,
                    digest_id=digest_id,
                )
                return NotifyResult(
                    digest_id=digest_id,
                    webhook_sent=webhook_sent,
                    webhook_error=webhook_error,
                    email_attempted=email_attempted,
                    email_sent=False,
                    email_error=email_error,
                )

            email_attempted = True
            email_sent, email_error = send_email(
                html_body=html_body,
                subject=f"arxiv-digest: {digest_id}",
                recipient=recipient,
                smtp_host=host,
                smtp_port=settings.smtp_port(conn),
                smtp_username=settings.smtp_username(conn),
                smtp_password=settings.smtp_password(conn),
                use_tls=settings.smtp_use_tls(conn),
            )

            if email_sent:
                now = datetime.now(UTC).isoformat(timespec="seconds")
                conn.execute(
                    "UPDATE digests SET sent_at = ? WHERE digest_id = ?",
                    (now, digest_id),
                )
                conn.commit()
                log.info(
                    "notify.digest.sent_at_updated",
                    digest_id=digest_id,
                    sent_at=now,
                )

    return NotifyResult(
        digest_id=digest_id,
        webhook_sent=webhook_sent,
        webhook_error=webhook_error,
        email_attempted=email_attempted,
        email_sent=email_sent,
        email_error=email_error,
    )
