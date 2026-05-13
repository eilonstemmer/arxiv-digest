"""Webapp-side knowledge of which settings exist, their types, labels, and defaults.

Mirrors PROJECT_BRIEF.md §4 settings table. Used by the Settings page to
render the form and validate/save submitted values.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

FieldType = Literal["text", "password", "number", "bool", "float"]


@dataclasses.dataclass(frozen=True)
class SettingMeta:
    key: str
    label: str
    field_type: FieldType
    default: str | None = None
    help_text: str | None = None
    section: str = "General"


# Ordered list of user-editable settings (PROJECT_BRIEF.md §4).
# pipeline-managed keys (profile_embedding_hash, profile_embedding_b64) are excluded.
SETTINGS: tuple[SettingMeta, ...] = (
    # ---- Anthropic ----
    SettingMeta(
        key="anthropic_api_key",
        label="Anthropic API Key",
        field_type="password",
        section="Anthropic",
        help_text="Required for all LLM calls. Get one at https://console.anthropic.com/",
    ),
    # ---- Email notification ----
    SettingMeta(
        key="smtp_host",
        label="SMTP Host",
        field_type="text",
        section="Email Notification",
        help_text="e.g. smtp.gmail.com",
    ),
    SettingMeta(
        key="smtp_port",
        label="SMTP Port",
        field_type="number",
        default="587",
        section="Email Notification",
    ),
    SettingMeta(
        key="smtp_username",
        label="SMTP Username",
        field_type="text",
        section="Email Notification",
        help_text="Usually the same as the notify email address.",
    ),
    SettingMeta(
        key="smtp_password",
        label="SMTP Password",
        field_type="password",
        section="Email Notification",
        help_text="App Password for Gmail — not your account password.",
    ),
    SettingMeta(
        key="smtp_use_tls",
        label="SMTP Use TLS",
        field_type="bool",
        default="true",
        section="Email Notification",
    ),
    SettingMeta(
        key="notify_email",
        label="Notify Email",
        field_type="text",
        section="Email Notification",
        help_text="Where to send the digest.",
    ),
    SettingMeta(
        key="notify_enabled",
        label="Notifications Enabled",
        field_type="bool",
        default="false",
        section="Email Notification",
        help_text="Master switch for email delivery.",
    ),
    # ---- Digest behaviour ----
    SettingMeta(
        key="digest_top_n",
        label="Top-N Papers",
        field_type="number",
        default="80",
        section="Digest Behavior",
        help_text="How many papers to fully summarize each week.",
    ),
    SettingMeta(
        key="triage_model",
        label="Triage Model",
        field_type="text",
        default="claude-haiku-4-5-20251001",
        section="Digest Behavior",
    ),
    SettingMeta(
        key="summarize_model",
        label="Summarize Model",
        field_type="text",
        default="claude-sonnet-4-6",
        section="Digest Behavior",
    ),
    SettingMeta(
        key="triage_prefilter_enabled",
        label="Prefilter Enabled",
        field_type="bool",
        default="true",
        section="Digest Behavior",
        help_text="Local cosine-similarity prefilter before Haiku. Saves ~$0.25/week.",
    ),
    SettingMeta(
        key="triage_prefilter_keep_fraction",
        label="Prefilter Keep Fraction",
        field_type="float",
        default="0.4",
        section="Digest Behavior",
        help_text=(
            "Higher = more thorough but costlier triage. "
            "Lower = cheaper but may miss off-vocabulary matches. "
            "Default 0.4 sends ~800/2,000 weekly papers to Haiku."
        ),
    ),
)

# Quick lookup by key.
SETTINGS_BY_KEY: dict[str, SettingMeta] = {s.key: s for s in SETTINGS}

# Secret keys (password fields).
SECRET_KEYS: frozenset[str] = frozenset(
    s.key for s in SETTINGS if s.field_type == "password"
)

# All user-editable keys in order.
USER_SETTING_KEYS: tuple[str, ...] = tuple(s.key for s in SETTINGS)
