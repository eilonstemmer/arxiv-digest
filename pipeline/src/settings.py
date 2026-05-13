"""Typed settings accessors with environment-variable fallback.

Resolution order (PROJECT_BRIEF.md section 4):
  1. Database `settings` table (via db.get_setting).
  2. Environment variable of the same name in uppercase, when (1) is absent
     or the stored value is the empty string ("explicitly cleared").
  3. The documented default (passed as `default=...`) for optional settings;
     otherwise raise MissingSettingError pointing the operator at /settings.

Use the named convenience getters in pipeline modules rather than the
generic helpers. They live near the bottom of this file and mirror the
PROJECT_BRIEF settings table row-for-row.
"""

from __future__ import annotations

import os
import sqlite3
from typing import overload

import structlog

from . import db

log = structlog.get_logger(__name__)

# Documented defaults (PROJECT_BRIEF.md section 4 table).
DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_USE_TLS = True
DEFAULT_NOTIFY_ENABLED = False
DEFAULT_DIGEST_TOP_N = 80
DEFAULT_TRIAGE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_SUMMARIZE_MODEL = "claude-sonnet-4-6"
DEFAULT_TRIAGE_PREFILTER_ENABLED = True
DEFAULT_TRIAGE_PREFILTER_KEEP_FRACTION = 0.4

# Operator-editable keys (shown on the Settings page).
USER_SETTING_KEYS: tuple[str, ...] = (
    "anthropic_api_key",
    "smtp_host",
    "smtp_port",
    "smtp_username",
    "smtp_password",
    "smtp_use_tls",
    "notify_email",
    "notify_enabled",
    "digest_top_n",
    "triage_model",
    "summarize_model",
    "triage_prefilter_enabled",
    "triage_prefilter_keep_fraction",
)

# Pipeline-managed cache keys. Never user-editable; never shown on the
# Settings page. Written by triage.py when profile.md changes.
MANAGED_SETTING_KEYS: tuple[str, ...] = (
    "profile_embedding_hash",
    "profile_embedding_b64",
)


class MissingSettingError(RuntimeError):
    """Raised when a required setting has no value in DB, env, or defaults."""

    def __init__(self, key: str) -> None:
        msg = (
            f"Required setting '{key}' is not set. "
            f"Visit http://localhost:8080/settings to configure it, "
            f"or export {key.upper()} in the environment."
        )
        super().__init__(msg)
        self.key = key


class SettingTypeError(ValueError):
    """Raised when a stored setting value cannot be parsed to its expected type."""

    def __init__(self, key: str, raw: str, type_name: str) -> None:
        msg = (
            f"Setting '{key}' = {raw!r} is not a valid {type_name}. "
            f"Fix it at http://localhost:8080/settings or correct the "
            f"{key.upper()} environment variable."
        )
        super().__init__(msg)
        self.key = key
        self.raw = raw


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _resolve_raw(conn: sqlite3.Connection, key: str) -> str | None:
    """Steps 1 + 2 of the resolution chain. Returns None when both DB and
    env say absent or empty.
    """
    value = db.get_setting(conn, key)
    if value is not None and value != "":
        return value
    env_value = os.environ.get(key.upper())
    if env_value is not None and env_value != "":
        return env_value
    return None


# ----- Generic typed accessors ---------------------------------------------
#
# The overloads narrow the return type to T when a `default: T` is provided,
# and to T | None otherwise. The single implementation handles both shapes.


@overload
def get_str(conn: sqlite3.Connection, key: str, *, default: str) -> str: ...
@overload
def get_str(conn: sqlite3.Connection, key: str) -> str | None: ...


def get_str(
    conn: sqlite3.Connection,
    key: str,
    *,
    default: str | None = None,
) -> str | None:
    raw = _resolve_raw(conn, key)
    return raw if raw is not None else default


def require_str(conn: sqlite3.Connection, key: str) -> str:
    raw = _resolve_raw(conn, key)
    if raw is None:
        raise MissingSettingError(key)
    return raw


@overload
def get_int(conn: sqlite3.Connection, key: str, *, default: int) -> int: ...
@overload
def get_int(conn: sqlite3.Connection, key: str) -> int | None: ...


def get_int(
    conn: sqlite3.Connection,
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    raw = _resolve_raw(conn, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SettingTypeError(key, raw, "int") from exc


def require_int(conn: sqlite3.Connection, key: str) -> int:
    val = get_int(conn, key)
    if val is None:
        raise MissingSettingError(key)
    return val


@overload
def get_float(conn: sqlite3.Connection, key: str, *, default: float) -> float: ...
@overload
def get_float(conn: sqlite3.Connection, key: str) -> float | None: ...


def get_float(
    conn: sqlite3.Connection,
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    raw = _resolve_raw(conn, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise SettingTypeError(key, raw, "float") from exc


def require_float(conn: sqlite3.Connection, key: str) -> float:
    val = get_float(conn, key)
    if val is None:
        raise MissingSettingError(key)
    return val


@overload
def get_bool(conn: sqlite3.Connection, key: str, *, default: bool) -> bool: ...
@overload
def get_bool(conn: sqlite3.Connection, key: str) -> bool | None: ...


def get_bool(
    conn: sqlite3.Connection,
    key: str,
    *,
    default: bool | None = None,
) -> bool | None:
    raw = _resolve_raw(conn, key)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    raise SettingTypeError(key, raw, "bool")


def require_bool(conn: sqlite3.Connection, key: str) -> bool:
    val = get_bool(conn, key)
    if val is None:
        raise MissingSettingError(key)
    return val


# ----- Named convenience getters --------------------------------------------
#
# One per documented setting (PROJECT_BRIEF.md section 4). Use these in
# pipeline modules rather than the generic helpers — call sites stay tight
# and the defaults live in one place.


def anthropic_api_key(conn: sqlite3.Connection) -> str:
    """Required. Raises MissingSettingError if absent in DB and env."""
    return require_str(conn, "anthropic_api_key")


def smtp_host(conn: sqlite3.Connection) -> str | None:
    return get_str(conn, "smtp_host")


def smtp_port(conn: sqlite3.Connection) -> int:
    return get_int(conn, "smtp_port", default=DEFAULT_SMTP_PORT)


def smtp_username(conn: sqlite3.Connection) -> str | None:
    return get_str(conn, "smtp_username")


def smtp_password(conn: sqlite3.Connection) -> str | None:
    return get_str(conn, "smtp_password")


def smtp_use_tls(conn: sqlite3.Connection) -> bool:
    return get_bool(conn, "smtp_use_tls", default=DEFAULT_SMTP_USE_TLS)


def notify_email(conn: sqlite3.Connection) -> str | None:
    return get_str(conn, "notify_email")


def notify_enabled(conn: sqlite3.Connection) -> bool:
    return get_bool(conn, "notify_enabled", default=DEFAULT_NOTIFY_ENABLED)


def digest_top_n(conn: sqlite3.Connection) -> int:
    return get_int(conn, "digest_top_n", default=DEFAULT_DIGEST_TOP_N)


def triage_model(conn: sqlite3.Connection) -> str:
    return get_str(conn, "triage_model", default=DEFAULT_TRIAGE_MODEL)


def summarize_model(conn: sqlite3.Connection) -> str:
    return get_str(conn, "summarize_model", default=DEFAULT_SUMMARIZE_MODEL)


def triage_prefilter_enabled(conn: sqlite3.Connection) -> bool:
    return get_bool(
        conn,
        "triage_prefilter_enabled",
        default=DEFAULT_TRIAGE_PREFILTER_ENABLED,
    )


def triage_prefilter_keep_fraction(conn: sqlite3.Connection) -> float:
    return get_float(
        conn,
        "triage_prefilter_keep_fraction",
        default=DEFAULT_TRIAGE_PREFILTER_KEEP_FRACTION,
    )


def profile_embedding_hash(conn: sqlite3.Connection) -> str | None:
    """Pipeline-managed. None until first triage run caches the profile."""
    return get_str(conn, "profile_embedding_hash")


def profile_embedding_b64(conn: sqlite3.Connection) -> str | None:
    """Pipeline-managed. None until first triage run caches the profile."""
    return get_str(conn, "profile_embedding_b64")
