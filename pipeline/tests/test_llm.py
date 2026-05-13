"""Tests for src.llm: prompt loading, template fill, validation, cost,
parse-and-validate, call_direct, run_batch.

The four `_anthropic_*` internals are monkeypatched in every test that
exercises a call path. Production code stays behind the LLM_MODE=replay
guard so a missed monkeypatch fails loudly rather than silently making
a real API call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src import llm

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


# ---- load_prompt -----------------------------------------------------------


def test_load_prompt_triage_real_file() -> None:
    p = llm.load_prompt("triage", prompts_dir=PROMPTS_DIR)
    assert p.name == "triage"
    assert "research-paper triage" in p.system_template
    assert "{{PROFILE_MARKDOWN}}" in p.system_template
    assert "{{TITLE}}" in p.user_template
    assert "{{ABSTRACT}}" in p.user_template
    # Schema validity
    assert p.schema["$schema"].startswith("https://json-schema.org/draft/2020-12")
    assert "relevance_score" in p.schema["properties"]


def test_load_prompt_summarize_real_file() -> None:
    p = llm.load_prompt("summarize", prompts_dir=PROMPTS_DIR)
    assert "research synthesis" in p.system_template
    assert "cross_domain_hooks" in p.schema["properties"]


def test_load_prompt_cluster_label_real_file() -> None:
    p = llm.load_prompt("cluster_label", prompts_dir=PROMPTS_DIR)
    assert "{{CLUSTER_SIZE}}" in p.user_template
    assert "label" in p.schema["properties"]


def test_load_prompt_trend_narrative_real_file() -> None:
    p = llm.load_prompt("trend_narrative", prompts_dir=PROMPTS_DIR)
    assert "movements" in p.schema["properties"]


def test_load_prompt_missing_md_raises(tmp_path: Path) -> None:
    with pytest.raises(llm.PromptNotFoundError, match="not found"):
        llm.load_prompt("nope", prompts_dir=tmp_path)


def test_load_prompt_missing_schema_raises(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("## System prompt\nhello\n", encoding="utf-8")
    with pytest.raises(llm.PromptNotFoundError, match="schema"):
        llm.load_prompt("x", prompts_dir=tmp_path)


def test_load_prompt_missing_system_section_raises(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text("just text, no headers\n", encoding="utf-8")
    (tmp_path / "x.schema.json").write_text('{"type": "object"}', encoding="utf-8")
    with pytest.raises(ValueError, match="System prompt"):
        llm.load_prompt("x", prompts_dir=tmp_path)


def test_load_prompt_strips_trailing_horizontal_rule(tmp_path: Path) -> None:
    (tmp_path / "x.md").write_text(
        "## System prompt\n\nBody text.\n\n---\n\n## User template\n\nUser body.\n",
        encoding="utf-8",
    )
    (tmp_path / "x.schema.json").write_text('{"type": "object"}', encoding="utf-8")
    p = llm.load_prompt("x", prompts_dir=tmp_path)
    assert p.system_template == "Body text."
    assert p.user_template == "User body."


def test_load_prompt_preserves_nested_subheadings(tmp_path: Path) -> None:
    """Regression: section termination must only fire on the two section
    markers, not on arbitrary `##` lines inside a section body.

    Was: prompts/summarize.md's `## This paper` / `## Cross-domain candidates`
    sub-headings prematurely truncated the user template body.
    """
    (tmp_path / "x.md").write_text(
        "## System prompt\n\nSystem body.\n\n"
        "## User template\n\n"
        "## Inner sub-heading\n"
        "first chunk\n\n"
        "## Another sub-heading\n"
        "second chunk\n",
        encoding="utf-8",
    )
    (tmp_path / "x.schema.json").write_text('{"type": "object"}', encoding="utf-8")
    p = llm.load_prompt("x", prompts_dir=tmp_path)
    assert "## Inner sub-heading" in p.user_template
    assert "first chunk" in p.user_template
    assert "## Another sub-heading" in p.user_template
    assert "second chunk" in p.user_template


def test_load_prompt_real_summarize_user_template_has_candidates_section() -> None:
    """The real summarize.md user template has sub-headings; verify they're
    preserved in the parsed user_template (regression for nested ## bug)."""
    p = llm.load_prompt("summarize", prompts_dir=PROMPTS_DIR)
    assert "## This paper" in p.user_template
    assert "## Cross-domain candidates" in p.user_template
    assert "{{#each CANDIDATES}}" in p.user_template
    assert "{{#if NO_CANDIDATES}}" in p.user_template


# ---- fill_template ---------------------------------------------------------


def test_fill_template_simple() -> None:
    out = llm.fill_template("Hello {{NAME}}", {"NAME": "world"})
    assert out == "Hello world"


def test_fill_template_multiple_keys() -> None:
    out = llm.fill_template(
        "{{A}} and {{B}}", {"A": "alpha", "B": "beta"}
    )
    assert out == "alpha and beta"


def test_fill_template_repeated_key() -> None:
    out = llm.fill_template("{{X}}, {{X}}, {{X}}", {"X": "boom"})
    assert out == "boom, boom, boom"


def test_fill_template_missing_key_left_alone() -> None:
    # Unknown placeholders pass through unchanged (caller's choice to
    # validate completeness, or to render Handlebars loops separately).
    out = llm.fill_template("{{KNOWN}} {{UNKNOWN}}", {"KNOWN": "x"})
    assert out == "x {{UNKNOWN}}"


# ---- validate_response -----------------------------------------------------


_SIMPLE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["score", "label"],
    "properties": {
        "score": {"type": "number", "minimum": 0, "maximum": 10},
        "label": {"type": "string", "minLength": 1, "maxLength": 50},
    },
}


def test_validate_response_passes() -> None:
    llm.validate_response({"score": 7.5, "label": "ok"}, _SIMPLE_SCHEMA)


def test_validate_response_missing_required() -> None:
    import jsonschema

    with pytest.raises(jsonschema.ValidationError, match="score"):
        llm.validate_response({"label": "ok"}, _SIMPLE_SCHEMA)


def test_validate_response_out_of_range() -> None:
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        llm.validate_response({"score": 99, "label": "ok"}, _SIMPLE_SCHEMA)


def test_validate_response_additional_property_rejected() -> None:
    import jsonschema

    with pytest.raises(jsonschema.ValidationError):
        llm.validate_response(
            {"score": 1, "label": "ok", "extra": "x"}, _SIMPLE_SCHEMA
        )


# ---- estimate_cost ---------------------------------------------------------


def test_estimate_cost_haiku_direct() -> None:
    # 1M input, 1M output -> 1.00 + 5.00 = 6.00
    cost = llm.estimate_cost(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(6.00)


def test_estimate_cost_sonnet_batch_halved() -> None:
    cost = llm.estimate_cost(
        "claude-sonnet-4-6",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        is_batch=True,
    )
    # 3.00 + 15.00 halved -> 9.00
    assert cost == pytest.approx(9.00)


def test_estimate_cost_cached_input_cheaper() -> None:
    # 1M tokens cached vs 1M uncached for Haiku:
    # uncached: 1.00 / M -> $1.00. cached: 0.10 / M -> $0.10.
    cached = llm.estimate_cost(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=0,
        cached_input_tokens=1_000_000,
    )
    uncached = llm.estimate_cost(
        "claude-haiku-4-5-20251001",
        input_tokens=1_000_000,
        output_tokens=0,
    )
    assert cached < uncached
    assert cached == pytest.approx(0.10)


def test_estimate_cost_unknown_model_zero() -> None:
    assert llm.estimate_cost(
        "made-up-model", input_tokens=1000, output_tokens=1000
    ) == 0.0


# ---- _parse_and_validate ---------------------------------------------------


def test_parse_and_validate_success() -> None:
    data, err = llm._parse_and_validate(
        '{"score": 7.5, "label": "ok"}', _SIMPLE_SCHEMA
    )
    assert data == {"score": 7.5, "label": "ok"}
    assert err is None


def test_parse_and_validate_strips_markdown_fence() -> None:
    text = '```json\n{"score": 1, "label": "x"}\n```'
    data, err = llm._parse_and_validate(text, _SIMPLE_SCHEMA)
    assert data is not None
    assert err is None


def test_parse_and_validate_bad_json() -> None:
    data, err = llm._parse_and_validate("not json", _SIMPLE_SCHEMA)
    assert data is None
    assert err is not None
    assert "JSON parse error" in err


def test_parse_and_validate_array_at_top_level_rejected() -> None:
    data, err = llm._parse_and_validate("[1,2,3]", _SIMPLE_SCHEMA)
    assert data is None
    assert err is not None
    assert "object" in err


def test_parse_and_validate_schema_fail() -> None:
    data, err = llm._parse_and_validate('{"score": 99, "label": "x"}', _SIMPLE_SCHEMA)
    assert data is None
    assert err is not None
    assert "schema" in err


# ---- _build_system_blocks --------------------------------------------------


def test_build_system_blocks_with_cache() -> None:
    blocks = llm._build_system_blocks("hello", cache=True)
    assert blocks == [
        {
            "type": "text",
            "text": "hello",
            "cache_control": {"type": "ephemeral"},
        }
    ]


def test_build_system_blocks_without_cache() -> None:
    blocks = llm._build_system_blocks("hello", cache=False)
    assert blocks == [{"type": "text", "text": "hello"}]


# ---- call_direct (mocked) --------------------------------------------------


def _fake_response(text: str, *, input_tok: int = 100, output_tok: int = 50) -> llm._RawApiResponse:
    return llm._RawApiResponse(
        text=text,
        input_tokens=input_tok,
        output_tokens=output_tok,
        cached_input_tokens=0,
    )


def test_call_direct_success_first_try(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        calls.append(kwargs)
        return _fake_response('{"score": 7, "label": "good"}')

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)
    result = llm.call_direct(
        model="claude-haiku-4-5-20251001",
        system="sys",
        user="user",
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert result.status == "success"
    assert result.data == {"score": 7, "label": "good"}
    assert result.input_tokens == 100
    assert result.output_tokens == 50
    assert result.cost_usd > 0
    assert len(calls) == 1


def test_call_direct_retry_on_schema_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _fake_response('{"score": 999, "label": "bad"}'),  # out of range
            _fake_response('{"score": 5, "label": "ok"}'),
        ]
    )
    calls: list[str] = []

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        calls.append(kwargs["user_message"])
        return next(responses)

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)
    result = llm.call_direct(
        model="claude-haiku-4-5-20251001",
        system="sys",
        user="original user msg",
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert result.status == "success"
    assert result.data == {"score": 5, "label": "ok"}
    assert len(calls) == 2
    # Retry message should contain corrective text
    assert "previous response was invalid" in calls[1]
    # Token counts include both attempts
    assert result.input_tokens == 200
    assert result.output_tokens == 100


def test_call_direct_skip_after_two_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            _fake_response('not valid json'),
            _fake_response('still not json'),
        ]
    )

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        return next(responses)

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)
    result = llm.call_direct(
        model="claude-haiku-4-5-20251001",
        system="sys",
        user="user",
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert result.status == "schema_failed"
    assert result.data is None
    assert result.error is not None
    assert result.raw_text == "still not json"


def test_call_direct_no_retry_when_allow_retry_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = 0

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        nonlocal call_count
        call_count += 1
        return _fake_response("not json")

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)
    result = llm.call_direct(
        model="claude-haiku-4-5-20251001",
        system="sys",
        user="user",
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
        allow_retry=False,
    )
    assert call_count == 1
    assert result.status == "schema_failed"


def test_call_direct_api_error_returns_api_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anthropic

    class _FakeAPIError(anthropic.APIError):
        def __init__(self) -> None:
            self.message = "boom"

        def __str__(self) -> str:
            return "boom"

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        raise _FakeAPIError()

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)
    result = llm.call_direct(
        model="claude-haiku-4-5-20251001",
        system="sys",
        user="user",
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert result.status == "api_failed"
    assert result.error == "boom"
    assert result.cost_usd == 0.0


def test_call_direct_replay_mode_blocks_real_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "LLM_MODE", "replay")
    with pytest.raises(llm.LLMReplayError, match=r"messages\.create"):
        llm.call_direct(
            model="claude-haiku-4-5-20251001",
            system="sys",
            user="user",
            schema=_SIMPLE_SCHEMA,
            api_key="test-key",
        )


# ---- run_batch (mocked) ----------------------------------------------------


def _fake_batch_entry(
    custom_id: str,
    text: str,
    *,
    status: str = "succeeded",
    input_tok: int = 100,
    output_tok: int = 50,
) -> llm._RawBatchEntry:
    return llm._RawBatchEntry(
        custom_id=custom_id,
        status=status,
        text=text if status == "succeeded" else None,
        error=None if status == "succeeded" else f"status={status}",
        input_tokens=input_tok,
        output_tokens=output_tok,
        cached_input_tokens=0,
    )


def _wire_batch_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entries: list[llm._RawBatchEntry],
    statuses: list[str] | None = None,
    batch_id: str = "msg-batch-test-123",
) -> None:
    statuses = statuses or ["ended"]
    status_iter = iter(statuses)

    monkeypatch.setattr(
        llm,
        "_anthropic_batches_create",
        lambda **kw: batch_id,
    )

    def fake_status(*, api_key: str, batch_id: str) -> str:
        return next(status_iter)

    monkeypatch.setattr(llm, "_anthropic_batches_status", fake_status)
    monkeypatch.setattr(
        llm, "_anthropic_batches_results", lambda **kw: entries
    )


def test_run_batch_all_success(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [
        _fake_batch_entry("a", '{"score": 5, "label": "x"}'),
        _fake_batch_entry("b", '{"score": 7, "label": "y"}'),
    ]
    _wire_batch_mocks(monkeypatch, entries=entries)
    monkeypatch.setattr("time.sleep", lambda _: None)

    requests = [
        llm.BatchRequest(custom_id="a", user_message="m1"),
        llm.BatchRequest(custom_id="b", user_message="m2"),
    ]
    results = llm.run_batch(
        model="claude-haiku-4-5-20251001",
        system="sys",
        requests=requests,
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert len(results) == 2
    assert {r.custom_id for r in results} == {"a", "b"}
    assert all(r.status == "success" for r in results)


def test_run_batch_empty_requests_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Should not even call the SDK.
    def boom(**kw: Any) -> Any:
        raise AssertionError("must not be called")

    monkeypatch.setattr(llm, "_anthropic_batches_create", boom)
    assert (
        llm.run_batch(
            model="claude-haiku-4-5-20251001",
            system="sys",
            requests=[],
            schema=_SIMPLE_SCHEMA,
            api_key="test-key",
        )
        == []
    )


def test_run_batch_errored_entry_returns_api_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [
        _fake_batch_entry("a", "", status="errored"),
        _fake_batch_entry("b", '{"score": 1, "label": "ok"}'),
    ]
    _wire_batch_mocks(monkeypatch, entries=entries)
    monkeypatch.setattr("time.sleep", lambda _: None)

    requests = [
        llm.BatchRequest(custom_id="a", user_message="m1"),
        llm.BatchRequest(custom_id="b", user_message="m2"),
    ]
    results = llm.run_batch(
        model="claude-haiku-4-5-20251001",
        system="sys",
        requests=requests,
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    by_id = {r.custom_id: r for r in results}
    assert by_id["a"].status == "api_failed"
    assert by_id["b"].status == "success"


def test_run_batch_schema_fail_triggers_one_direct_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [_fake_batch_entry("a", '{"score": 999, "label": "bad"}')]
    _wire_batch_mocks(monkeypatch, entries=entries)
    monkeypatch.setattr("time.sleep", lambda _: None)

    # The retry goes through _anthropic_messages_create
    retry_calls: list[str] = []

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        retry_calls.append(kwargs["user_message"])
        return _fake_response('{"score": 4, "label": "fixed"}')

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)

    results = llm.run_batch(
        model="claude-haiku-4-5-20251001",
        system="sys",
        requests=[llm.BatchRequest(custom_id="a", user_message="m1")],
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert len(results) == 1
    assert results[0].status == "success"
    assert results[0].data == {"score": 4, "label": "fixed"}
    assert len(retry_calls) == 1
    assert "previous response was invalid" in retry_calls[0]


def test_run_batch_schema_fail_retry_also_fails_returns_schema_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = [_fake_batch_entry("a", '{"score": 999, "label": "bad"}')]
    _wire_batch_mocks(monkeypatch, entries=entries)
    monkeypatch.setattr("time.sleep", lambda _: None)

    def fake_create(**kwargs: Any) -> llm._RawApiResponse:
        return _fake_response("still not json")

    monkeypatch.setattr(llm, "_anthropic_messages_create", fake_create)

    results = llm.run_batch(
        model="claude-haiku-4-5-20251001",
        system="sys",
        requests=[llm.BatchRequest(custom_id="a", user_message="m1")],
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
    )
    assert results[0].status == "schema_failed"


def test_run_batch_polls_until_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    entries = [_fake_batch_entry("a", '{"score": 1, "label": "x"}')]
    statuses = ["in_progress", "in_progress", "ended"]
    _wire_batch_mocks(monkeypatch, entries=entries, statuses=statuses)
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    results = llm.run_batch(
        model="claude-haiku-4-5-20251001",
        system="sys",
        requests=[llm.BatchRequest(custom_id="a", user_message="m1")],
        schema=_SIMPLE_SCHEMA,
        api_key="test-key",
        poll_interval_seconds=0.01,
    )
    assert len(results) == 1
    # Slept twice while waiting for "ended" to come up.
    assert len(sleeps) == 2


def test_run_batch_timeout_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _wire_batch_mocks(
        monkeypatch,
        entries=[],
        statuses=["in_progress", "in_progress", "in_progress"],
    )
    monkeypatch.setattr("time.sleep", lambda _: None)
    # Deterministic clock: first call (deadline init) = 0; subsequent calls
    # jump past the deadline so the loop bails on iteration 1.
    clock = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr("time.monotonic", lambda: next(clock))

    with pytest.raises(TimeoutError, match="did not finish"):
        llm.run_batch(
            model="claude-haiku-4-5-20251001",
            system="sys",
            requests=[llm.BatchRequest(custom_id="a", user_message="m1")],
            schema=_SIMPLE_SCHEMA,
            api_key="test-key",
            poll_interval_seconds=0.0,
            timeout_seconds=1.0,
        )


def test_run_batch_replay_mode_blocks_create(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "LLM_MODE", "replay")
    with pytest.raises(llm.LLMReplayError, match=r"batches\.create"):
        llm.run_batch(
            model="claude-haiku-4-5-20251001",
            system="sys",
            requests=[llm.BatchRequest(custom_id="a", user_message="m1")],
            schema=_SIMPLE_SCHEMA,
            api_key="test-key",
        )
