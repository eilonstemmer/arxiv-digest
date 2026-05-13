"""Tests for src.settings: resolution order, type parsing, named getters."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from src import db, settings


@pytest.fixture
def conn(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    c = db.connect(tmp_path / "test.db")
    db.init_schema(c)
    yield c
    c.close()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wipe every settings-related env var so tests are isolated from the
    developer's shell. monkeypatch auto-restores after the test."""
    for key in (*settings.USER_SETTING_KEYS, *settings.MANAGED_SETTING_KEYS):
        monkeypatch.delenv(key.upper(), raising=False)


# ---- resolution order ------------------------------------------------------


def test_db_value_wins_over_env(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.set_setting(conn, "smtp_host", "from-db.example.com")
    monkeypatch.setenv("SMTP_HOST", "from-env.example.com")
    assert settings.smtp_host(conn) == "from-db.example.com"


def test_env_used_when_db_absent(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_HOST", "from-env.example.com")
    assert settings.smtp_host(conn) == "from-env.example.com"


def test_empty_db_string_falls_through_to_env(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    db.set_setting(conn, "smtp_host", "")
    monkeypatch.setenv("SMTP_HOST", "from-env.example.com")
    assert settings.smtp_host(conn) == "from-env.example.com"


def test_empty_env_falls_through_to_default(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SMTP_HOST", "")
    # smtp_host has no default -> returns None
    assert settings.smtp_host(conn) is None


def test_default_used_when_neither_set(conn: sqlite3.Connection) -> None:
    assert settings.digest_top_n(conn) == settings.DEFAULT_DIGEST_TOP_N


# ---- required + missing ----------------------------------------------------


def test_required_missing_raises(conn: sqlite3.Connection) -> None:
    with pytest.raises(settings.MissingSettingError, match="anthropic_api_key"):
        settings.anthropic_api_key(conn)


def test_required_present_in_db_returns_value(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "anthropic_api_key", "test-key-value")
    assert settings.anthropic_api_key(conn) == "test-key-value"


def test_required_present_in_env_returns_value(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key-value")
    assert settings.anthropic_api_key(conn) == "env-key-value"


def test_missing_error_message_mentions_settings_page_and_env(
    conn: sqlite3.Connection,
) -> None:
    with pytest.raises(settings.MissingSettingError) as exc_info:
        settings.anthropic_api_key(conn)
    msg = str(exc_info.value)
    assert "/settings" in msg
    assert "ANTHROPIC_API_KEY" in msg
    assert exc_info.value.key == "anthropic_api_key"


# ---- generic accessors -----------------------------------------------------


def test_get_str_no_default_returns_none(conn: sqlite3.Connection) -> None:
    assert settings.get_str(conn, "never_set") is None


def test_get_str_with_default_returns_default(conn: sqlite3.Connection) -> None:
    assert settings.get_str(conn, "never_set", default="fallback") == "fallback"


def test_get_str_with_default_returns_value_when_set(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "k", "stored")
    assert settings.get_str(conn, "k", default="fallback") == "stored"


def test_get_int_no_default_returns_none(conn: sqlite3.Connection) -> None:
    assert settings.get_int(conn, "never_set") is None


def test_get_int_parses_valid(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "digest_top_n", "120")
    assert settings.digest_top_n(conn) == 120


def test_get_int_raises_on_invalid(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "digest_top_n", "abc")
    with pytest.raises(settings.SettingTypeError, match="abc"):
        settings.digest_top_n(conn)


def test_setting_type_error_carries_metadata(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "digest_top_n", "abc")
    with pytest.raises(settings.SettingTypeError) as exc_info:
        settings.digest_top_n(conn)
    assert exc_info.value.key == "digest_top_n"
    assert exc_info.value.raw == "abc"


def test_get_float_parses(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "triage_prefilter_keep_fraction", "0.6")
    assert settings.triage_prefilter_keep_fraction(conn) == 0.6


def test_get_float_raises_on_invalid(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "triage_prefilter_keep_fraction", "not-a-float")
    with pytest.raises(settings.SettingTypeError):
        settings.triage_prefilter_keep_fraction(conn)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("  true  ", True),
    ],
)
def test_bool_parsing(conn: sqlite3.Connection, raw: str, expected: bool) -> None:
    db.set_setting(conn, "notify_enabled", raw)
    assert settings.notify_enabled(conn) is expected


def test_bool_raises_on_invalid(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "notify_enabled", "maybe")
    with pytest.raises(settings.SettingTypeError):
        settings.notify_enabled(conn)


def test_require_str_raises_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(settings.MissingSettingError):
        settings.require_str(conn, "anywhere")


def test_require_int_raises_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(settings.MissingSettingError):
        settings.require_int(conn, "anywhere")


def test_require_int_returns_value_when_set(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "anywhere", "42")
    assert settings.require_int(conn, "anywhere") == 42


def test_require_float_raises_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(settings.MissingSettingError):
        settings.require_float(conn, "anywhere")


def test_require_bool_raises_when_missing(conn: sqlite3.Connection) -> None:
    with pytest.raises(settings.MissingSettingError):
        settings.require_bool(conn, "anywhere")


# ---- defaults per documented setting ---------------------------------------


def test_smtp_port_default(conn: sqlite3.Connection) -> None:
    assert settings.smtp_port(conn) == 587


def test_smtp_use_tls_default(conn: sqlite3.Connection) -> None:
    assert settings.smtp_use_tls(conn) is True


def test_notify_enabled_default(conn: sqlite3.Connection) -> None:
    assert settings.notify_enabled(conn) is False


def test_digest_top_n_default(conn: sqlite3.Connection) -> None:
    assert settings.digest_top_n(conn) == 80


def test_triage_model_default(conn: sqlite3.Connection) -> None:
    assert settings.triage_model(conn) == "claude-haiku-4-5-20251001"


def test_summarize_model_default(conn: sqlite3.Connection) -> None:
    assert settings.summarize_model(conn) == "claude-sonnet-4-6"


def test_triage_prefilter_enabled_default(conn: sqlite3.Connection) -> None:
    assert settings.triage_prefilter_enabled(conn) is True


def test_triage_prefilter_keep_fraction_default(conn: sqlite3.Connection) -> None:
    assert settings.triage_prefilter_keep_fraction(conn) == 0.4


# ---- pipeline-managed cache keys -------------------------------------------


def test_profile_embedding_hash_none_when_not_cached(conn: sqlite3.Connection) -> None:
    assert settings.profile_embedding_hash(conn) is None


def test_profile_embedding_b64_none_when_not_cached(conn: sqlite3.Connection) -> None:
    assert settings.profile_embedding_b64(conn) is None


def test_profile_embedding_round_trip(conn: sqlite3.Connection) -> None:
    db.set_setting(conn, "profile_embedding_hash", "abc123")
    db.set_setting(conn, "profile_embedding_b64", "fakebase64==")
    assert settings.profile_embedding_hash(conn) == "abc123"
    assert settings.profile_embedding_b64(conn) == "fakebase64=="


# ---- key registry sanity ---------------------------------------------------


def test_managed_keys_disjoint_from_user_keys() -> None:
    assert not set(settings.MANAGED_SETTING_KEYS) & set(settings.USER_SETTING_KEYS)


def test_user_keys_match_documented_set() -> None:
    expected = {
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
    }
    assert set(settings.USER_SETTING_KEYS) == expected


def test_managed_keys_match_documented_set() -> None:
    assert set(settings.MANAGED_SETTING_KEYS) == {
        "profile_embedding_hash",
        "profile_embedding_b64",
    }
