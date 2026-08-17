"""The OpenAI client: reported usage, ignored cache flag, explicit parameter shapes."""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

import pytest
from doubles import (
    RecordingSleep,
    StubHTTPError,
    StubOpenAISDK,
    openai_response,
)

from rlm0.ports import LMClient
from rlm0.providers import (
    CHAT_STYLE,
    REASONING_STYLE,
    OpenAIClient,
    ProviderDependencyError,
    ProviderResponseError,
    RetryPolicy,
    style_for,
)


def _client(results: list[object], **kwargs: Any) -> tuple[OpenAIClient, Any]:
    sdk = StubOpenAISDK(results)
    return OpenAIClient(client=sdk, **kwargs), sdk


def test_satisfies_the_port() -> None:
    client, _ = _client([openai_response()])
    assert isinstance(client, LMClient)


def test_usage_is_read_from_the_provider_not_derived_from_length() -> None:
    client, _ = _client([openai_response(prompt_tokens=11, completion_tokens=2)])
    response = client.complete(
        system="s" * 40_000,
        messages=[{"role": "user", "content": "u" * 150_000}],
        model="gpt-4o",
        max_tokens=256,
    )
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 2


def test_cached_tokens_are_split_out_of_prompt_tokens() -> None:
    """`prompt_tokens` is inclusive of the cached part on this API.

    Carrying both across whole would count the cached prefix twice in
    `billed_input`, which is the total a budget is checked against.
    """
    client, _ = _client([openai_response(prompt_tokens=1000, cached_tokens=768)])
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=64,
    )
    assert response.usage.cache_read_tokens == 768
    assert response.usage.input_tokens == 232
    assert response.usage.billed_input == 1000
    assert response.cached_prefix is True


def test_cache_fields_stay_zero_when_the_response_omits_them() -> None:
    client, _ = _client([openai_response(prompt_tokens=1000)])
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=64,
        cache_prefix=True,
    )
    assert response.usage.cache_read_tokens == 0
    assert response.usage.cache_write_tokens == 0
    assert response.usage.input_tokens == 1000
    assert response.cached_prefix is False


def test_cache_prefix_is_ignored_rather_than_emulated() -> None:
    """OpenAI offers no way to mark a block, so nothing about the request changes.

    Emulating the flag by reshaping the request would make it useless as a
    diagnostic, because a caller could no longer tell which providers actually
    honour it.
    """
    client, sdk = _client([openai_response(), openai_response()])
    messages = [
        {"role": "user", "content": "shared"},
        {"role": "user", "content": "tail"},
    ]
    for flag in (False, True):
        client.complete(
            system="instructions",
            messages=messages,
            model="gpt-4o",
            max_tokens=64,
            cache_prefix=flag,
        )
    assert sdk.create.requests[0] == sdk.create.requests[1]
    assert "cache_control" not in str(sdk.create.requests[1])


def test_a_chat_model_gets_max_tokens() -> None:
    client, sdk = _client([openai_response()])
    client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=321,
    )
    request = sdk.create.last_request
    assert request["max_tokens"] == 321
    assert "max_completion_tokens" not in request


def test_a_reasoning_model_gets_max_completion_tokens() -> None:
    """Looked up in a registry, not sniffed out of the model name."""
    client, sdk = _client([openai_response(model="o3")])
    client.complete(
        system="be careful",
        messages=[{"role": "user", "content": "q"}],
        model="o3",
        max_tokens=321,
    )
    request = sdk.create.last_request
    assert request["max_completion_tokens"] == 321
    assert "max_tokens" not in request
    assert request["messages"][0]["role"] == "developer"


def test_temperature_is_dropped_for_a_model_that_rejects_it() -> None:
    client, sdk = _client([openai_response(model="o3")], temperature=0.7)
    client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="o3",
        max_tokens=64,
    )
    assert "temperature" not in sdk.create.last_request


def test_temperature_is_sent_to_a_model_that_accepts_it() -> None:
    client, sdk = _client([openai_response()], temperature=0.7)
    client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=64,
    )
    assert sdk.create.last_request["temperature"] == 0.7


def test_dropping_a_temperature_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client, _ = _client(
        [openai_response(model="o3"), openai_response(model="o3")], temperature=0.7
    )
    with caplog.at_level(logging.WARNING, logger="rlm0.providers.openai_client"):
        for _ in range(2):
            client.complete(
                system="",
                messages=[{"role": "user", "content": "q"}],
                model="o3",
                max_tokens=64,
            )
    assert len([r for r in caplog.records if "temperature" in r.getMessage()]) == 1


def test_an_unregistered_model_falls_back_to_the_chat_shape() -> None:
    """Documented behaviour: unknown means default, and a wrong guess errors loudly.

    A substring rule would instead misclassify silently, and a future model
    whose name does not contain the magic token would be sent the wrong shape
    with nobody the wiser until the provider rejected it.
    """
    assert style_for("some-model-released-next-year") is CHAT_STYLE
    assert style_for("o3") is REASONING_STYLE


def test_a_caller_can_register_a_new_reasoning_model() -> None:
    styles = {"house-reasoner": REASONING_STYLE}
    client, sdk = _client([openai_response()], parameter_styles=styles)
    client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="house-reasoner",
        max_tokens=64,
    )
    assert sdk.create.last_request["max_completion_tokens"] == 64


def test_a_length_finish_reason_sets_truncated() -> None:
    client, _ = _client([openai_response(finish_reason="length")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=8,
    )
    assert response.stop_reason == "length"
    assert response.truncated is True


def test_an_unknown_model_costs_none_rather_than_zero() -> None:
    client, _ = _client([openai_response(model="gpt-nextgen")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-nextgen",
        max_tokens=64,
    )
    assert response.cost_usd is None
    assert "gpt-nextgen" in client.prices.unpriced_models


def test_a_known_model_is_priced_from_reported_usage() -> None:
    client, _ = _client(
        [openai_response(prompt_tokens=1_000_000, completion_tokens=0)]
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=64,
    )
    assert response.cost_usd == pytest.approx(2.50)


def test_a_retried_call_does_not_double_count_usage() -> None:
    sleep = RecordingSleep()
    client, sdk = _client(
        [
            StubHTTPError(429, {"retry-after": "3"}),
            openai_response(prompt_tokens=500, completion_tokens=40),
        ],
        retry_policy=RetryPolicy(max_attempts=3),
        sleep=sleep,
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="gpt-4o",
        max_tokens=64,
    )
    assert sdk.create.call_count == 2
    assert sleep.delays == [3.0]
    assert response.usage.total == 540


def test_a_response_without_usage_is_refused() -> None:
    client, _ = _client([{"model": "gpt-4o", "choices": []}])
    with pytest.raises(ProviderResponseError, match="no usage object"):
        client.complete(
            system="",
            messages=[{"role": "user", "content": "q"}],
            model="gpt-4o",
            max_tokens=64,
        )


@pytest.mark.skipif(
    importlib.util.find_spec("openai") is not None,
    reason="the SDK is installed here, so the missing-dependency path cannot run",
)
def test_a_missing_sdk_says_what_to_install() -> None:
    with pytest.raises(ProviderDependencyError) as caught:
        OpenAIClient()
    assert "pip install" in str(caught.value)
