"""The Gemini client: reported usage, a cache that does not fit, honest None costs.

Mocked at the SDK boundary, `client.models.generate_content`, with responses as
plain dicts, for the reasons `doubles.py` gives: mocking further in would test
the wrapper against itself, and hand-built SDK objects encode a belief about a
class hierarchy rather than the hierarchy.

The Gemini-shaped stubs live here rather than in `doubles.py` because the one
that matters, the error, is shaped differently from the shared HTTP stub: this
SDK puts the status on `code`, which is exactly the difference the client has
to handle.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from doubles import RecordingCreate, RecordingSleep

from rlm0.ports import LMClient
from rlm0.providers import ProviderDependencyError, ProviderResponseError, RetryPolicy
from rlm0.providers.gemini_client import GeminiClient, gemini_retryable


class StubGenAISDK:
    """Shaped like `genai.Client()`: a `.models.generate_content`."""

    def __init__(self, results: Sequence[object]) -> None:
        self.create = RecordingCreate(results)
        self.models = _Namespace(generate_content=self.create)


class _Namespace:
    def __init__(self, **attributes: Any) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class StubGenAIError(Exception):
    """An `google.genai.errors.APIError`: the status is on `code`, not `status_code`."""

    def __init__(
        self, code: int, headers: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(f"stub genai error code={code}")
        self.code = code
        if headers is not None:
            self.response = _Namespace(headers=dict(headers))


def gemini_response(
    *,
    text: str = "answer",
    model_version: str = "gemini-2.5-pro",
    prompt_tokens: int = 1000,
    candidates_tokens: int = 50,
    thoughts_tokens: int | None = None,
    cached_tokens: int | None = None,
    finish_reason: str = "STOP",
    thought_text: str | None = None,
) -> dict[str, Any]:
    """A `generateContent` response body.

    The optional fields default to absent rather than zero, because a response
    that omits them entirely is the case the counters have to survive.
    """
    usage: dict[str, Any] = {
        "prompt_token_count": prompt_tokens,
        "candidates_token_count": candidates_tokens,
    }
    if thoughts_tokens is not None:
        usage["thoughts_token_count"] = thoughts_tokens
    if cached_tokens is not None:
        usage["cached_content_token_count"] = cached_tokens
    parts: list[dict[str, Any]] = []
    if thought_text is not None:
        parts.append({"text": thought_text, "thought": True})
    parts.append({"text": text})
    return {
        "model_version": model_version,
        "candidates": [
            {"finish_reason": finish_reason, "content": {"parts": parts}}
        ],
        "usage_metadata": usage,
    }


def _client(results: list[object], **kwargs: Any) -> tuple[GeminiClient, Any]:
    sdk = StubGenAISDK(results)
    return GeminiClient(client=sdk, **kwargs), sdk


def test_satisfies_the_port() -> None:
    client, _ = _client([gemini_response()])
    assert isinstance(client, LMClient)


def test_usage_is_read_from_the_provider_not_derived_from_length() -> None:
    client, _ = _client([gemini_response(prompt_tokens=11, candidates_tokens=2)])
    response = client.complete(
        system="s" * 40_000,
        messages=[{"role": "user", "content": "u" * 150_000}],
        model="gemini-2.5-pro",
        max_tokens=256,
    )
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 2


def test_cached_tokens_are_split_out_of_the_prompt_count() -> None:
    """`prompt_token_count` is inclusive of the cached part on this API.

    Carrying both across whole would count the cached prefix twice in
    `billed_input`, which is the total a budget is checked against.
    """
    client, _ = _client([gemini_response(prompt_tokens=1000, cached_tokens=768)])
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.usage.cache_read_tokens == 768
    assert response.usage.input_tokens == 232
    assert response.usage.billed_input == 1000
    assert response.cached_prefix is True


def test_a_time_billed_cache_never_lands_in_cache_write_tokens() -> None:
    """The point of the whole module.

    Gemini bills explicit cached content by storage duration, and `TokenUsage`
    has no clock. Writing the cached count into `cache_write_tokens` would give
    every per-token price table something wrong to multiply, so the field stays
    empty and the storage cost stays unrepresented rather than guessed.
    """
    client, _ = _client(
        [gemini_response(prompt_tokens=5000, cached_tokens=4096)],
        cached_content="cachedContents/abc123",
    )
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.usage.cache_write_tokens == 0
    assert response.cost_usd is None


def test_an_explicit_cache_resource_is_referenced_and_readable_back() -> None:
    client, sdk = _client(
        [gemini_response()], cached_content="cachedContents/abc123"
    )
    client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    config = sdk.create.last_request["config"]
    assert config["cached_content"] == "cachedContents/abc123"
    assert client.cached_content == "cachedContents/abc123"


def test_no_cache_resource_is_created_on_the_callers_behalf() -> None:
    """Creating one would start a meter that nothing in this package stops."""
    client, sdk = _client([gemini_response()])
    client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert "cached_content" not in sdk.create.last_request["config"]
    assert client.cached_content is None


def test_cache_fields_stay_zero_when_the_response_omits_them() -> None:
    client, _ = _client([gemini_response(prompt_tokens=1000)])
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
        cache_prefix=True,
    )
    assert response.usage.cache_read_tokens == 0
    assert response.usage.cache_write_tokens == 0
    assert response.usage.input_tokens == 1000
    assert response.cached_prefix is False


def test_cache_prefix_is_ignored_rather_than_emulated() -> None:
    """Neither Gemini cache is a per-request breakpoint, so nothing changes.

    Reshaping the request on the strength of a flag the provider cannot honour
    would make the flag useless as a diagnostic everywhere else.
    """
    client, sdk = _client([gemini_response(), gemini_response()])
    messages = [
        {"role": "user", "content": "shared"},
        {"role": "user", "content": "tail"},
    ]
    for flag in (False, True):
        client.complete(
            system="instructions",
            messages=messages,
            model="gemini-2.5-pro",
            max_tokens=64,
            cache_prefix=flag,
        )
    assert sdk.create.requests[0] == sdk.create.requests[1]
    assert "cache_control" not in str(sdk.create.requests[1])


def test_thought_tokens_are_billed_as_output_and_counted_as_output() -> None:
    """They are reported apart from the candidate count and billed at its rate.

    Dropping them understates the bill by exactly the part that grows when the
    model is asked to think harder.
    """
    client, _ = _client(
        [gemini_response(candidates_tokens=50, thoughts_tokens=400)]
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.usage.output_tokens == 450


def test_thought_parts_are_not_read_as_the_answer() -> None:
    client, _ = _client(
        [gemini_response(text="42", thought_text="let me reconsider, maybe 41")]
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.text == "42"


def test_the_system_prompt_travels_as_an_instruction_not_a_turn() -> None:
    client, sdk = _client([gemini_response()])
    client.complete(
        system="be careful",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    request = sdk.create.last_request
    assert request["config"]["system_instruction"] == "be careful"
    assert len(request["contents"]) == 1
    assert request["config"]["max_output_tokens"] == 64


def test_the_assistant_role_is_renamed_and_the_order_is_kept() -> None:
    """Gemini spells it `model`, and both caches match a prefix from token one."""
    client, sdk = _client([gemini_response()])
    client.complete(
        system="",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    contents = sdk.create.last_request["contents"]
    assert [turn["role"] for turn in contents] == ["user", "model", "user"]
    assert [turn["parts"][0]["text"] for turn in contents] == [
        "first",
        "second",
        "third",
    ]


def test_a_max_tokens_finish_reason_is_normalised_so_truncation_is_visible() -> None:
    """`LMResponse.truncated` keys on `max_tokens`, and Gemini says `MAX_TOKENS`.

    Carrying the provider's spelling across would report every truncated
    sub-call as a complete one, and a truncated answer is indistinguishable
    from a short one everywhere downstream.
    """
    client, _ = _client([gemini_response(finish_reason="MAX_TOKENS")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=8,
    )
    assert response.stop_reason == "max_tokens"
    assert response.truncated is True


def test_an_enum_shaped_finish_reason_is_normalised_too() -> None:
    """The SDK hands back an enum, whose repr carries the class name."""

    class _FinishReason:
        name = "MAX_TOKENS"

        def __str__(self) -> str:  # pragma: no cover - only the name is read
            return "FinishReason.MAX_TOKENS"

    body = gemini_response()
    body["candidates"][0]["finish_reason"] = _FinishReason()
    client, _ = _client([body])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=8,
    )
    assert response.truncated is True


def test_a_stop_finish_reason_is_not_truncation() -> None:
    client, _ = _client([gemini_response(finish_reason="STOP")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.stop_reason == "stop"
    assert response.truncated is False


def test_every_gemini_model_is_unpriced_rather_than_priced_from_a_guess() -> None:
    """No entry is shipped, so no run is reported cheaper than it was.

    A per-token table would also understate any run using explicit caching, by
    the storage bill, which the response does not report at all.
    """
    client, _ = _client([gemini_response(model_version="gemini-2.5-pro")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.cost_usd is None
    assert "gemini-2.5-pro" in client.prices.unpriced_models


def test_the_model_version_the_provider_answered_with_is_reported() -> None:
    client, _ = _client([gemini_response(model_version="gemini-2.5-pro-002")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.model == "gemini-2.5-pro-002"


def test_a_retried_call_does_not_double_count_usage() -> None:
    """The status arrives on `code`, and the delay hint on the response headers."""
    sleep = RecordingSleep()
    client, sdk = _client(
        [
            StubGenAIError(429, {"retry-after": "3"}),
            gemini_response(prompt_tokens=500, candidates_tokens=40),
        ],
        retry_policy=RetryPolicy(max_attempts=3),
        sleep=sleep,
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert sdk.create.call_count == 2
    assert sleep.delays == [3.0]
    assert response.usage.total == 540


def test_a_status_on_code_is_classified_where_the_shared_rule_sees_nothing() -> None:
    """The whole reason this client passes its own classifier.

    The shared one reads `status_code`, finds nothing, and would treat a 429 as
    fatal, ending a sweep hours in.
    """
    assert gemini_retryable(StubGenAIError(429)) is True
    assert gemini_retryable(StubGenAIError(503)) is True
    assert gemini_retryable(StubGenAIError(400)) is False
    assert gemini_retryable(StubGenAIError(401)) is False


def test_a_transport_failure_with_no_status_still_falls_to_the_shared_rule() -> None:
    assert gemini_retryable(TimeoutError("connection reset")) is True
    assert gemini_retryable(ValueError("not a transport problem")) is False


def test_a_permanent_failure_is_not_retried() -> None:
    sleep = RecordingSleep()
    client, sdk = _client(
        [StubGenAIError(400)],
        retry_policy=RetryPolicy(max_attempts=3),
        sleep=sleep,
    )
    with pytest.raises(StubGenAIError):
        client.complete(
            system="",
            messages=[{"role": "user", "content": "q"}],
            model="gemini-2.5-pro",
            max_tokens=64,
        )
    assert sdk.create.call_count == 1
    assert sleep.delays == []


def test_a_response_without_usage_is_refused() -> None:
    client, _ = _client([{"model_version": "gemini-2.5-pro", "candidates": []}])
    with pytest.raises(ProviderResponseError, match="no usage_metadata"):
        client.complete(
            system="",
            messages=[{"role": "user", "content": "q"}],
            model="gemini-2.5-pro",
            max_tokens=64,
        )


def test_a_response_without_candidates_yields_empty_text_rather_than_raising() -> None:
    """A blocked or empty answer is a result the run layer handles, not a crash."""
    client, _ = _client(
        [
            {
                "model_version": "gemini-2.5-pro",
                "candidates": [],
                "usage_metadata": {
                    "prompt_token_count": 10,
                    "candidates_token_count": 0,
                },
            }
        ]
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gemini-2.5-pro",
        max_tokens=64,
    )
    assert response.text == ""
    assert response.usage.input_tokens == 10


def _genai_installed() -> bool:
    """Whether the SDK is importable, without exploding when it is not.

    `find_spec` on a dotted name imports the parent package first and raises
    when it is absent, so the plain call would fail on exactly the machines
    this predicate exists to identify.
    """
    try:
        return importlib.util.find_spec("google.genai") is not None
    except ImportError:
        return False


@pytest.mark.skipif(
    _genai_installed(),
    reason="the SDK is installed here, so the missing-dependency path cannot run",
)
def test_a_missing_sdk_says_what_to_install() -> None:
    """And says the distribution name, which is not the module name."""
    with pytest.raises(ProviderDependencyError) as caught:
        GeminiClient()
    assert "pip install google-genai" in str(caught.value)
