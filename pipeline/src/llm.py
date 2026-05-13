"""LLM substrate: Anthropic direct + Batch API with prompt caching, JSON Schema
validation, and retry-once-then-skip.

This module is the only place in the pipeline that talks to Anthropic.
Modules 6 (triage), 7 (cluster), 8 (summarize), and 9 (trend_narrative)
all consume `call_direct` / `run_batch` and never import the SDK directly.

Tests monkeypatch the four `_anthropic_*` internal functions so they never
make real network calls. The `LLM_MODE=replay` env var (set by CI) acts as
a belt-and-suspenders guard: production code paths raise `LLMReplayError`
if they're reached without a monkeypatch.

Prompt + schema loading parses `prompts/<name>.md` (split on `## System
prompt` and `## User template` headers) and `prompts/<name>.schema.json`.
Both files must exist or `PromptNotFoundError` is raised.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import anthropic
import jsonschema
import structlog

log = structlog.get_logger(__name__)

# Set to "replay" in CI to forbid live API calls without an explicit
# monkeypatch. Production sets nothing (defaults to "live").
LLM_MODE = os.environ.get("LLM_MODE", "live")
DEFAULT_PROMPTS_DIR = Path(os.environ.get("PROMPTS_DIR", "/prompts"))
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.0
DEFAULT_POLL_INTERVAL = 30.0
DEFAULT_BATCH_TIMEOUT = 24 * 60 * 60  # 24h

# Per-million-token prices from anthropic.com/pricing (May 2026).
# (input, output, cache_read).
_PRICING: dict[str, tuple[float, float, float]] = {
    "claude-haiku-4-5-20251001": (1.00, 5.00, 0.10),
    "claude-sonnet-4-6": (3.00, 15.00, 0.30),
}


# ---------------------------------------------------------------------------
# Dataclasses + exceptions
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Prompt:
    """Loaded prompt template + JSON Schema for output validation."""

    name: str
    system_template: str
    user_template: str
    schema: dict[str, Any]


CallStatus = Literal["success", "schema_failed", "api_failed"]


@dataclasses.dataclass(frozen=True)
class CallResult:
    """Result of a single LLM call (from direct API or one batch item).

    `data` is the parsed-and-validated JSON object on success, None on any
    failure. Token counts and cost are populated regardless of status so
    pipeline_runs.cost_usd reflects all spend including failed attempts.
    """

    custom_id: str
    status: CallStatus
    data: dict[str, Any] | None
    error: str | None
    raw_text: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    cost_usd: float


@dataclasses.dataclass(frozen=True)
class BatchRequest:
    """One item to submit in a batch. `custom_id` is opaque to the API and
    used to correlate results back to inputs (e.g., arxiv_id)."""

    custom_id: str
    user_message: str


@dataclasses.dataclass(frozen=True)
class _RawApiResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int


@dataclasses.dataclass(frozen=True)
class _RawBatchEntry:
    custom_id: str
    status: str  # "succeeded" | "errored" | "canceled" | "expired"
    text: str | None
    error: str | None
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int


class LLMReplayError(RuntimeError):
    """Raised when production code attempts a real API call while
    LLM_MODE=replay -- a sign that a test forgot to monkeypatch."""


class PromptNotFoundError(FileNotFoundError):
    """The requested prompt's .md or .schema.json file is missing."""


# ---------------------------------------------------------------------------
# Prompt loading + template filling
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(
    r"^##\s+(?P<name>System prompt|User template)\s*$\n+"
    r"(?P<body>.*?)(?=\n^##\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)


def load_prompt(name: str, *, prompts_dir: Path | str | None = None) -> Prompt:
    """Load `<prompts_dir>/<name>.md` + `<name>.schema.json` and return a Prompt.

    The markdown file must contain a `## System prompt` section; the
    `## User template` section is optional (the trend-narrative prompt
    has one, the others have one too, but the parser tolerates missing).

    `prompts_dir` defaults to the `PROMPTS_DIR` env var (or `/prompts`
    inside containers). Tests pass an explicit path.
    """
    base = Path(prompts_dir) if prompts_dir is not None else DEFAULT_PROMPTS_DIR
    md_path = base / f"{name}.md"
    schema_path = base / f"{name}.schema.json"
    if not md_path.exists():
        raise PromptNotFoundError(f"prompt file not found: {md_path}")
    if not schema_path.exists():
        raise PromptNotFoundError(f"schema file not found: {schema_path}")

    text = md_path.read_text(encoding="utf-8")
    sections: dict[str, str] = {}
    for match in _SECTION_RE.finditer(text):
        body = match.group("body").strip()
        # Strip trailing horizontal rule plus its leading blank line.
        body = re.sub(r"\n+---\s*$", "", body).strip()
        sections[match.group("name")] = body

    if "System prompt" not in sections:
        raise ValueError(f"{md_path}: missing '## System prompt' section")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if not isinstance(schema, dict):
        raise ValueError(f"{schema_path}: schema must be a JSON object")

    return Prompt(
        name=name,
        system_template=sections["System prompt"],
        user_template=sections.get("User template", ""),
        schema=schema,
    )


def fill_template(template: str, mapping: dict[str, str]) -> str:
    """Replace `{{KEY}}` placeholders with mapped values.

    Simple string replacement. Handlebars-style `{{#each ...}}` loops in
    the prompt files are documentation only; the calling module is
    responsible for rendering its variable-length data into a plain
    string and substituting via this helper.
    """
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value)
    return out


# ---------------------------------------------------------------------------
# Validation + cost
# ---------------------------------------------------------------------------


def validate_response(data: Any, schema: dict[str, Any]) -> None:
    """Validate `data` against `schema` (Draft 2020-12). Raises ValidationError.

    On failure, the raised error's message combines up to the first five
    path/message pairs so the retry prompt can show the model exactly
    what to fix.
    """
    validator = jsonschema.Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    if not errors:
        return
    summary = "; ".join(
        f"{'.'.join(str(p) for p in e.absolute_path)}: {e.message}"
        for e in errors[:5]
    )
    raise jsonschema.ValidationError(summary)


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
    is_batch: bool = False,
) -> float:
    """Estimate USD spend for one call. Returns 0.0 for unknown models.

    `cached_input_tokens` is billed at the lower `cache_read` rate; the
    remaining uncached portion of `input_tokens` is billed at the
    standard input rate. Batch API halves both input and output rates;
    the cache_read rate is unchanged.
    """
    pricing = _PRICING.get(model)
    if pricing is None:
        return 0.0
    input_per_m, output_per_m, cache_read_per_m = pricing
    if is_batch:
        input_per_m *= 0.5
        output_per_m *= 0.5
    uncached = max(0, input_tokens - cached_input_tokens)
    return (
        uncached * input_per_m / 1_000_000
        + cached_input_tokens * cache_read_per_m / 1_000_000
        + output_tokens * output_per_m / 1_000_000
    )


def _parse_and_validate(
    text: str, schema: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """Parse `text` as JSON, strip an optional markdown code fence, validate.

    Returns `(data, None)` on success or `(None, error_message)` on
    any failure. Never raises.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        if first_nl > 0:
            cleaned = cleaned[first_nl + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].rstrip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(data, dict):
        return None, f"expected JSON object, got {type(data).__name__}"
    try:
        validate_response(data, schema)
    except jsonschema.ValidationError as exc:
        return None, f"schema validation failed: {exc.message}"
    return data, None


# ---------------------------------------------------------------------------
# Anthropic SDK boundary (monkeypatched in tests)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _get_client(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


def _check_replay_mode(operation: str) -> None:
    if LLM_MODE == "replay":
        raise LLMReplayError(
            f"Live API call attempted in replay mode (operation={operation}). "
            "Tests must monkeypatch the _anthropic_* internals."
        )


def _build_system_blocks(system: str, *, cache: bool) -> list[dict[str, Any]]:
    block: dict[str, Any] = {"type": "text", "text": system}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return [block]


def _anthropic_messages_create(
    *,
    api_key: str,
    model: str,
    system_blocks: list[dict[str, Any]],
    user_message: str,
    max_tokens: int,
    temperature: float,
) -> _RawApiResponse:
    _check_replay_mode("messages.create")
    client = _get_client(api_key)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_blocks,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": user_message}],
    )
    text = "".join(
        getattr(block, "text", "")
        for block in response.content
        if getattr(block, "type", None) == "text"
    )
    cached = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    return _RawApiResponse(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cached_input_tokens=int(cached),
    )


def _anthropic_batches_create(
    *,
    api_key: str,
    model: str,
    system_blocks: list[dict[str, Any]],
    requests: list[BatchRequest],
    max_tokens: int,
    temperature: float,
) -> str:
    _check_replay_mode("batches.create")
    client = _get_client(api_key)
    sdk_requests = [
        {
            "custom_id": r.custom_id,
            "params": {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_blocks,
                "messages": [{"role": "user", "content": r.user_message}],
            },
        }
        for r in requests
    ]
    batch = client.messages.batches.create(requests=sdk_requests)  # type: ignore[arg-type]
    return str(batch.id)


def _anthropic_batches_status(*, api_key: str, batch_id: str) -> str:
    _check_replay_mode("batches.retrieve")
    client = _get_client(api_key)
    batch = client.messages.batches.retrieve(batch_id)
    return str(batch.processing_status)


def _anthropic_batches_results(
    *, api_key: str, batch_id: str
) -> list[_RawBatchEntry]:
    _check_replay_mode("batches.results")
    client = _get_client(api_key)
    out: list[_RawBatchEntry] = []
    for entry in client.messages.batches.results(batch_id):
        custom_id = str(entry.custom_id)
        result_type = str(getattr(entry.result, "type", "errored"))
        if result_type == "succeeded":
            msg = entry.result.message  # type: ignore[union-attr]
            text = "".join(
                getattr(block, "text", "")
                for block in msg.content
                if getattr(block, "type", None) == "text"
            )
            cached = getattr(msg.usage, "cache_read_input_tokens", 0) or 0
            out.append(
                _RawBatchEntry(
                    custom_id=custom_id,
                    status="succeeded",
                    text=text,
                    error=None,
                    input_tokens=int(msg.usage.input_tokens),
                    output_tokens=int(msg.usage.output_tokens),
                    cached_input_tokens=int(cached),
                )
            )
        else:
            out.append(
                _RawBatchEntry(
                    custom_id=custom_id,
                    status=result_type,
                    text=None,
                    error=str(entry.result),
                    input_tokens=0,
                    output_tokens=0,
                    cached_input_tokens=0,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Public: direct call
# ---------------------------------------------------------------------------


def call_direct(
    *,
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    cache_system: bool = True,
    custom_id: str = "",
    allow_retry: bool = True,
) -> CallResult:
    """One Anthropic Messages call with JSON Schema validation.

    On validation failure, if `allow_retry=True` (default), makes one
    corrective call with the validator's error appended to the user
    message. If that also fails (or `allow_retry=False`), returns
    `status="schema_failed"` with the second attempt's raw text.

    Token counts and cost include both attempts so cost tracking sees
    the full spend even on retry-then-skip outcomes.
    """
    system_blocks = _build_system_blocks(system, cache=cache_system)

    first = _attempt_direct(
        api_key=api_key,
        model=model,
        system_blocks=system_blocks,
        user=user,
        schema=schema,
        max_tokens=max_tokens,
        temperature=temperature,
        custom_id=custom_id,
    )
    if first.status != "schema_failed" or not allow_retry:
        return first

    log.warning(
        "llm.direct.validation_failed", custom_id=custom_id, retrying=True
    )
    retry_user = _retry_user_message(user, first.error or "")
    second = _attempt_direct(
        api_key=api_key,
        model=model,
        system_blocks=system_blocks,
        user=retry_user,
        schema=schema,
        max_tokens=max_tokens,
        temperature=temperature,
        custom_id=custom_id,
    )
    # Combine token counts so cost_usd reflects total spend.
    return CallResult(
        custom_id=custom_id,
        status=second.status,
        data=second.data,
        error=second.error,
        raw_text=second.raw_text,
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        cached_input_tokens=first.cached_input_tokens + second.cached_input_tokens,
        cost_usd=first.cost_usd + second.cost_usd,
    )


def _attempt_direct(
    *,
    api_key: str,
    model: str,
    system_blocks: list[dict[str, Any]],
    user: str,
    schema: dict[str, Any],
    max_tokens: int,
    temperature: float,
    custom_id: str,
) -> CallResult:
    try:
        raw = _anthropic_messages_create(
            api_key=api_key,
            model=model,
            system_blocks=system_blocks,
            user_message=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except anthropic.APIError as exc:
        log.error("llm.direct.api_error", error=str(exc), custom_id=custom_id)
        return CallResult(
            custom_id=custom_id,
            status="api_failed",
            data=None,
            error=str(exc),
            raw_text=None,
            input_tokens=0,
            output_tokens=0,
            cached_input_tokens=0,
            cost_usd=0.0,
        )
    cost = estimate_cost(
        model,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cached_input_tokens=raw.cached_input_tokens,
    )
    data, error = _parse_and_validate(raw.text, schema)
    status: CallStatus = "success" if data is not None else "schema_failed"
    return CallResult(
        custom_id=custom_id,
        status=status,
        data=data,
        error=error,
        raw_text=raw.text,
        input_tokens=raw.input_tokens,
        output_tokens=raw.output_tokens,
        cached_input_tokens=raw.cached_input_tokens,
        cost_usd=cost,
    )


def _retry_user_message(original: str, error: str) -> str:
    return (
        f"{original}\n\n"
        f"Your previous response was invalid: {error}\n"
        "Return ONLY the JSON object matching the schema. "
        "No preamble, no markdown code fence, no commentary."
    )


# ---------------------------------------------------------------------------
# Public: batch
# ---------------------------------------------------------------------------


def run_batch(
    *,
    model: str,
    system: str,
    requests: list[BatchRequest],
    schema: dict[str, Any],
    api_key: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    cache_system: bool = True,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL,
    timeout_seconds: float = DEFAULT_BATCH_TIMEOUT,
) -> list[CallResult]:
    """Submit a batch, poll until finished, validate each result, retry
    schema failures once via direct API, return all results.

    Output order mirrors `requests`. Per-item retries are exactly one
    direct call (via `call_direct(allow_retry=False)`) so the total
    attempts per item never exceed two (one batch + one direct).
    """
    if not requests:
        return []

    system_blocks = _build_system_blocks(system, cache=cache_system)
    batch_id = _anthropic_batches_create(
        api_key=api_key,
        model=model,
        system_blocks=system_blocks,
        requests=requests,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    log.info("llm.batch.submitted", batch_id=batch_id, count=len(requests))

    deadline = time.monotonic() + timeout_seconds
    while True:
        status = _anthropic_batches_status(api_key=api_key, batch_id=batch_id)
        log.info("llm.batch.poll", batch_id=batch_id, status=status)
        if status == "ended":
            break
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"batch {batch_id} did not finish within {timeout_seconds}s "
                f"(last status: {status})"
            )
        time.sleep(poll_interval_seconds)

    raw_entries = _anthropic_batches_results(api_key=api_key, batch_id=batch_id)
    log.info(
        "llm.batch.results_fetched", batch_id=batch_id, count=len(raw_entries)
    )
    return _finalize_batch_results(
        raw_entries=raw_entries,
        requests=requests,
        model=model,
        system=system,
        schema=schema,
        api_key=api_key,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_system=cache_system,
    )


def _finalize_batch_results(
    *,
    raw_entries: list[_RawBatchEntry],
    requests: list[BatchRequest],
    model: str,
    system: str,
    schema: dict[str, Any],
    api_key: str,
    max_tokens: int,
    temperature: float,
    cache_system: bool,
) -> list[CallResult]:
    requests_by_id = {r.custom_id: r for r in requests}
    out: list[CallResult] = []
    for entry in raw_entries:
        cost = estimate_cost(
            model,
            input_tokens=entry.input_tokens,
            output_tokens=entry.output_tokens,
            cached_input_tokens=entry.cached_input_tokens,
            is_batch=True,
        )
        if entry.status != "succeeded" or entry.text is None:
            out.append(
                CallResult(
                    custom_id=entry.custom_id,
                    status="api_failed",
                    data=None,
                    error=entry.error or f"batch entry status={entry.status}",
                    raw_text=entry.text,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    cached_input_tokens=entry.cached_input_tokens,
                    cost_usd=cost,
                )
            )
            continue

        data, error = _parse_and_validate(entry.text, schema)
        if data is not None:
            out.append(
                CallResult(
                    custom_id=entry.custom_id,
                    status="success",
                    data=data,
                    error=None,
                    raw_text=entry.text,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    cached_input_tokens=entry.cached_input_tokens,
                    cost_usd=cost,
                )
            )
            continue

        original = requests_by_id.get(entry.custom_id)
        if original is None:
            log.error(
                "llm.batch.unknown_custom_id",
                batch_custom_id=entry.custom_id,
            )
            out.append(
                CallResult(
                    custom_id=entry.custom_id,
                    status="schema_failed",
                    data=None,
                    error=error,
                    raw_text=entry.text,
                    input_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    cached_input_tokens=entry.cached_input_tokens,
                    cost_usd=cost,
                )
            )
            continue

        log.warning(
            "llm.batch.retrying", custom_id=entry.custom_id, error=error
        )
        retry = call_direct(
            model=model,
            system=system,
            user=_retry_user_message(original.user_message, error or ""),
            schema=schema,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            cache_system=cache_system,
            custom_id=entry.custom_id,
            allow_retry=False,
        )
        out.append(
            CallResult(
                custom_id=entry.custom_id,
                status=retry.status,
                data=retry.data,
                error=retry.error,
                raw_text=retry.raw_text,
                input_tokens=entry.input_tokens + retry.input_tokens,
                output_tokens=entry.output_tokens + retry.output_tokens,
                cached_input_tokens=(
                    entry.cached_input_tokens + retry.cached_input_tokens
                ),
                cost_usd=cost + retry.cost_usd,
            )
        )
    return out
