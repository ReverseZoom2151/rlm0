"""The Anthropic client: real usage, real cache markers, honest absence of price."""

from __future__ import annotations

import importlib.util
from typing import Any

import pytest
from doubles import (
    RecordingSleep,
    StubAnthropicSDK,
    StubHTTPError,
    anthropic_response,
)

from rlm0.ports import LMClient
from rlm0.providers import (
    AnthropicClient,
    ModelPrice,
    PriceTable,
    ProviderDependencyError,
    ProviderResponseError,
    RetryPolicy,
)


def _client(results: list[object], **kwargs: Any) -> tuple[AnthropicClient, Any]:
    sdk = StubAnthropicSDK(results)
    return AnthropicClient(client=sdk, **kwargs), sdk


def test_satisfies_the_port() -> None:
    client, _ = _client([anthropic_response()])
    assert isinstance(client, LMClient)


def test_usage_is_read_from_the_provider_not_derived_from_length() -> None:
    """The distinguishing test: a very long prompt with a small reported count.

    Any implementation that estimates from string length cannot pass this,
    because the only source for 17 is the response body.
    """
    client, _ = _client([anthropic_response(input_tokens=17, output_tokens=3)])
    response = client.complete(
        system="s" * 50_000,
        messages=[{"role": "user", "content": "u" * 200_000}],
        model="claude-sonnet-4-6",
        max_tokens=1024,
    )
    assert response.usage.input_tokens == 17
    assert response.usage.output_tokens == 3


def test_cache_fields_are_populated_when_the_response_reports_them() -> None:
    client, _ = _client(
        [anthropic_response(input_tokens=100, cache_read=8000, cache_write=200)]
    )
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=64,
        cache_prefix=True,
    )
    assert response.usage.cache_read_tokens == 8000
    assert response.usage.cache_write_tokens == 200
    assert response.cached_prefix is True
    assert response.usage.billed_input == 8300


def test_cache_fields_stay_zero_when_the_response_omits_them() -> None:
    """Asking for a cache and getting one are different events.

    `cached_prefix` follows what the provider reported, so a request that set
    the flag and got no cache read back still reports False. That gap is the
    bug this diagnostic exists to make visible.
    """
    client, _ = _client([anthropic_response(input_tokens=100)])
    response = client.complete(
        system="shared",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=64,
        cache_prefix=True,
    )
    assert response.usage.cache_read_tokens == 0
    assert response.usage.cache_write_tokens == 0
    assert response.cached_prefix is False


def test_cache_prefix_marks_the_system_block_and_the_stable_head() -> None:
    client, sdk = _client([anthropic_response()])
    client.complete(
        system="instructions",
        messages=[
            {"role": "user", "content": "shared context"},
            {"role": "assistant", "content": "acknowledged"},
            {"role": "user", "content": "slice specific to this child"},
        ],
        model="claude-sonnet-4-6",
        max_tokens=64,
        cache_prefix=True,
    )
    request = sdk.create.last_request
    system = request["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"}

    messages = request["messages"]
    # The breakpoint sits at the end of the stable head, which is everything
    # except the final message. In a fan-out the final message is the only part
    # that differs between children, so this is the shared/varying boundary.
    assert "cache_control" not in str(messages[0])
    assert messages[1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert messages[2]["content"] == "slice specific to this child"


def test_no_cache_markers_are_sent_when_the_flag_is_off() -> None:
    client, sdk = _client([anthropic_response()])
    client.complete(
        system="instructions",
        messages=[
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ],
        model="claude-sonnet-4-6",
        max_tokens=64,
    )
    assert "cache_control" not in str(sdk.create.last_request)


def test_a_single_message_still_caches_the_system_block() -> None:
    client, sdk = _client([anthropic_response()])
    client.complete(
        system="instructions",
        messages=[{"role": "user", "content": "only"}],
        model="claude-sonnet-4-6",
        max_tokens=64,
        cache_prefix=True,
    )
    request = sdk.create.last_request
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert request["messages"][0]["content"] == "only"


def test_a_length_stop_sets_truncated() -> None:
    """A silently truncated sub-call answer looks exactly like a short one."""
    client, _ = _client([anthropic_response(stop_reason="max_tokens")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=8,
    )
    assert response.stop_reason == "max_tokens"
    assert response.truncated is True


def test_a_normal_stop_is_not_truncated() -> None:
    client, _ = _client([anthropic_response(stop_reason="end_turn")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=1024,
    )
    assert response.truncated is False


def test_an_unknown_model_costs_none_rather_than_zero() -> None:
    client, _ = _client([anthropic_response(model="claude-experimental-9")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-experimental-9",
        max_tokens=64,
    )
    assert response.cost_usd is None
    assert "claude-experimental-9" in client.prices.unpriced_models


def test_a_known_model_is_priced_from_reported_usage() -> None:
    client, _ = _client(
        [
            anthropic_response(
                model="claude-sonnet-4-6", input_tokens=1_000_000, output_tokens=0
            )
        ]
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=64,
    )
    assert response.cost_usd == pytest.approx(3.00)


def test_cost_follows_the_model_the_provider_says_it_served() -> None:
    """A gateway that silently reroutes should not be priced as what we asked for."""
    client, _ = _client([anthropic_response(model="claude-haiku-4-5")])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-opus-5",
        max_tokens=64,
    )
    assert response.model == "claude-haiku-4-5"


def test_a_retried_call_does_not_double_count_usage() -> None:
    """Only the response that finally arrived exists to be counted.

    The failed attempt raised, so it produced no usage to accumulate. This is
    the structural reason retry and honest accounting do not conflict, and the
    test pins it so a future refactor that sums across attempts fails here.
    """
    sleep = RecordingSleep()
    client, sdk = _client(
        [
            StubHTTPError(429, {"retry-after": "2"}),
            anthropic_response(input_tokens=1000, output_tokens=100),
        ],
        retry_policy=RetryPolicy(max_attempts=3),
        sleep=sleep,
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=64,
    )
    assert sdk.create.call_count == 2
    assert sleep.delays == [2.0]
    assert response.usage.input_tokens == 1000
    assert response.usage.output_tokens == 100
    assert response.usage.total == 1100


def test_a_response_without_usage_is_refused() -> None:
    """Zero tokens would make the call free in every total that ever sums it."""
    client, _ = _client([{"model": "claude-sonnet-4-6", "content": []}])
    with pytest.raises(ProviderResponseError, match="no usage object"):
        client.complete(
            system="",
            messages=[{"role": "user", "content": "q"}],
            model="claude-sonnet-4-6",
            max_tokens=64,
        )


def test_text_is_joined_across_blocks_and_skips_non_text() -> None:
    body = {
        "model": "claude-sonnet-4-6",
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "not part of the answer"},
            {"type": "text", "text": "first "},
            {"type": "text", "text": "second"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    client, _ = _client([body])
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="claude-sonnet-4-6",
        max_tokens=64,
    )
    assert response.text == "first second"


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is not None,
    reason="the SDK is installed here, so the missing-dependency path cannot run",
)
def test_a_missing_sdk_says_what_to_install() -> None:
    with pytest.raises(ProviderDependencyError) as caught:
        AnthropicClient()
    message = str(caught.value)
    assert "pip install" in message
    assert "anthropic" in message


def test_a_price_table_can_be_supplied() -> None:
    table = PriceTable().with_overrides(
        {"house": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0)}
    )
    client, _ = _client(
        [anthropic_response(model="house", input_tokens=1_000_000, output_tokens=0)],
        prices=table,
    )
    response = client.complete(
        system="",
        messages=[{"role": "user", "content": "q"}],
        model="house",
        max_tokens=64,
    )
    assert response.cost_usd == pytest.approx(1.0)
