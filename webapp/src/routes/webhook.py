"""Route: POST /webhook/digest-ready — HMAC-SHA256 verified internal callback."""

from __future__ import annotations

import hashlib
import hmac
import os

import structlog
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)

router = APIRouter()


def _get_webhook_secret() -> str:
    secret = os.environ.get("WEBHOOK_SECRET", "")
    if not secret:
        raise RuntimeError(
            "WEBHOOK_SECRET environment variable is not set. "
            "Generate with `openssl rand -hex 32` and add to .env."
        )
    return secret


def _verify_signature(body: bytes, received_sig: str, secret: str) -> bool:
    """Return True iff received_sig matches the HMAC-SHA256 of body."""
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"), body, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, received_sig)


@router.post("/webhook/digest-ready")
async def digest_ready(request: Request) -> JSONResponse:
    received_sig = request.headers.get("X-Webhook-Signature", "")
    if not received_sig:
        log.warning("webhook.missing_signature")
        raise HTTPException(status_code=401, detail="missing signature")

    body = await request.body()

    try:
        secret = _get_webhook_secret()
    except RuntimeError as exc:
        log.error("webhook.secret_unavailable", error=str(exc))
        raise HTTPException(status_code=500, detail="server misconfiguration") from exc

    if not _verify_signature(body, received_sig, secret):
        log.warning("webhook.invalid_signature")
        raise HTTPException(status_code=401, detail="invalid signature")

    log.info("webhook.digest_ready_received")
    return JSONResponse(content={"status": "ok"})
