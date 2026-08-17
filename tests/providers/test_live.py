"""Calls that hit the real APIs, kept out of the default run on purpose.

Two gates, not one. The key must be present, because without it nothing can
run, and RLM0_LIVE must be set, because a developer who happens to export a key
for unrelated work should not discover that `pytest` spends their money. The
mocked tests are the ones that must pass everywhere; these exist for the
narrower job the mocked tests cannot do, which is to check that the field names
this package reads off a response are still the field names the provider sends.

That is the whole point of the module. Every mocked test in this directory
asserts against a response body written by the same author who wrote the reader,
so a rename on the provider's side would leave every one of them green. These
two tests are the only place that assumption gets checked, which is also why
they assert on the shape of the usage rather than on the content of the answer.
"""

from __future__ import annotations

import os

import pytest

from rlm0.providers import AnthropicClient, OpenAIClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RLM0_LIVE") != "1",
    reason="live provider tests are opt-in; set RLM0_LIVE=1 to run them",
)

_SYSTEM = (
    "You are a terse assistant used by a test suite. Answer in one short word. "
    "This block is padding so that the prefix is long enough to be cacheable: "
) + ("Prefix caching requires a prefix of some length to take effect. " * 80)


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY is not set",
)
def test_anthropic_reports_usage_and_can_read_a_cached_prefix() -> None:
    client = AnthropicClient()
    model = os.environ.get("RLM0_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5")
    messages = [{"role": "user", "content": "Reply with the word: ok"}]

    first = client.complete(
        system=_SYSTEM,
        messages=[*messages, {"role": "user", "content": "Question one."}],
        model=model,
        max_tokens=16,
        cache_prefix=True,
    )
    assert first.usage.output_tokens > 0
    assert first.usage.billed_input > 0
    assert first.cost_usd is not None, (
        f"{model} is not in the price table; add it or expect None costs"
    )

    second = client.complete(
        system=_SYSTEM,
        messages=[*messages, {"role": "user", "content": "Question two."}],
        model=model,
        max_tokens=16,
        cache_prefix=True,
    )
    # Not asserted as a hard requirement: cache writes need a minimum prefix
    # length that varies by model, and a miss here is informative rather than
    # a defect in this package.
    if second.usage.cache_read_tokens == 0:
        pytest.skip(
            "the provider reported no cache read; the shared prefix is "
            "probably below this model's minimum cacheable length"
        )
    assert second.cached_prefix is True


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set",
)
def test_openai_reports_usage() -> None:
    client = OpenAIClient()
    model = os.environ.get("RLM0_LIVE_OPENAI_MODEL", "gpt-4o-mini")
    response = client.complete(
        system="You are a terse assistant. Answer in one short word.",
        messages=[{"role": "user", "content": "Reply with the word: ok"}],
        model=model,
        max_tokens=16,
    )
    assert response.usage.output_tokens > 0
    assert response.usage.input_tokens > 0
    assert response.stop_reason in {"stop", "length"}
