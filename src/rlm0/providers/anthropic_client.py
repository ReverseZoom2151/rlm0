"""Anthropic, with prefix caching wired up rather than mentioned.

Prefix caching is the reason this layer is worth writing carefully. Every
sub-call in an RLM fan-out shares a prefix by construction: the same system
block, the same instructions, the same framing, differing only in the slice of
context at the end. Reported savings on long-horizon agentic workloads run up
to eighty percent, and essentially nobody in this field wires it. There is a
live bug of exactly this shape filed against Anthropic's own SDK, where
subagents ship with caching disabled.

WHAT CALLERS MUST NOT DO
------------------------
A cache hit is a byte-exact prefix match. The provider walks the request from
the first token and stops at the first difference, so anything before a
breakpoint is load-bearing and anything variable in front of it is fatal. The
failure is silent: the call succeeds, the answer is fine, and the only symptom
is a bill.

Concretely, and in rough order of how often each one actually happens:

- Do not interpolate anything variable into `system`. A timestamp, a request
  id, a run id, a "current date", a user name, or a randomly ordered set
  rendered into the system text invalidates the prefix on every single call.
  This is the one that gets people. Put variable text in the last message.
- Grow the conversation monotonically. Appending invalidates nothing, because
  everything before the new turn is unchanged.
- Do not edit or re-order history. Rewriting an earlier turn, summarising in
  place, dropping a message from the middle, or sorting messages moves the
  first differing byte backwards and discards everything after it.
- Do not vary the model between calls that are meant to share a prefix. Caches
  are per model, so a fan-out that routes some children to a cheaper model
  gets no sharing between the two groups. That can still be the right trade;
  it just is not free.
- Keep the shared part first and the varying part last. In a fan-out, the
  slice of context each child is given is the varying part, so it belongs in
  the final message, after the breakpoint.
- Serialise deterministically. `json.dumps` without `sort_keys` and anything
  built by iterating a set produce a different prefix on each process.

WHERE THE BREAKPOINTS GO
------------------------
Two, when `cache_prefix` is set. One on the system block, and one on the last
message of the stable head, meaning everything except the final message. That
placement is chosen for the fan-out shape specifically: the children share
system plus everything up to the last message, and differ in the last message
alone, so the breakpoint sits exactly on the boundary between what is shared
and what is not. Putting it on the final message instead, which is the usual
multi-turn advice, would write a distinct cache entry per child and read none
of them.

`cached_prefix` and the cache token counts come from what the provider actually
reported, never from the fact that we asked. Asking and receiving are different
events, and the gap between them is the bug this diagnostic exists to catch.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from rlm0.ports import LMResponse
from rlm0.providers.errors import ProviderResponseError
from rlm0.providers.payload import field, int_field, str_field
from rlm0.providers.pricing import PriceTable
from rlm0.providers.retry import RetryPolicy, call_with_retry
from rlm0.providers.sdk import load_sdk
from rlm0.run import TokenUsage

__all__ = ["AnthropicClient"]

_EPHEMERAL = {"type": "ephemeral"}


class AnthropicClient:
    """An `LMClient` over the Anthropic Messages API.

    `client` exists so the SDK object can be supplied from outside, which is
    how the tests here run with no SDK installed and no network. It is also the
    seam a caller uses to pass an already-configured client, for instance one
    pointed at a gateway or carrying custom headers.
    """

    __slots__ = ("_client", "_extra", "_prices", "_retry", "_sleep")

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        prices: PriceTable | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        extra_request: dict[str, object] | None = None,
    ) -> None:
        if client is None:
            sdk = load_sdk("anthropic", "anthropic")
            client = (
                sdk.Anthropic(api_key=api_key) if api_key else sdk.Anthropic()
            )
        self._client = client
        self._prices = PriceTable() if prices is None else prices
        self._retry = RetryPolicy() if retry_policy is None else retry_policy
        self._sleep = sleep
        self._extra: dict[str, object] = dict(extra_request or {})

    @property
    def prices(self) -> PriceTable:
        """The table in use, so a caller can read back what it failed to price."""
        return self._prices

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        cache_prefix: bool = False,
    ) -> LMResponse:
        request = self.build_request(
            system=system,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            cache_prefix=cache_prefix,
        )
        started = time.perf_counter()
        raw: object = call_with_retry(
            lambda: self._client.messages.create(**request),
            policy=self._retry,
            sleep=self._sleep,
        )
        # Wall clock deliberately includes any backoff slept through. The run
        # layer bounds real elapsed time, and time spent waiting on a rate
        # limit is time the run spent.
        elapsed = time.perf_counter() - started

        usage = _usage_from(raw)
        answered_model = str_field(raw, "model", model)
        return LMResponse(
            text=_text_from(raw),
            usage=usage,
            model=answered_model,
            wall_clock_s=elapsed,
            cost_usd=self._prices.cost_usd(answered_model, usage),
            # Reported, not requested. See the module docstring.
            cached_prefix=usage.cache_read_tokens > 0,
            stop_reason=str_field(raw, "stop_reason", ""),
        )

    def build_request(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        cache_prefix: bool,
    ) -> dict[str, object]:
        """The request body, exposed so a test can assert where the markers land.

        Public because cache-control placement is the substance of this class
        and asserting it through a mock's call args is how that substance stays
        correct across edits.
        """
        request: dict[str, object] = dict(self._extra)
        request["model"] = model
        request["max_tokens"] = max_tokens
        request["messages"] = _messages_for(messages, cache_prefix=cache_prefix)
        if system:
            request["system"] = _system_for(system, cache_prefix=cache_prefix)
        return request


def _system_for(system: str, *, cache_prefix: bool) -> object:
    """The system field: a plain string, or one cacheable block."""
    if not cache_prefix:
        return system
    return [{"type": "text", "text": system, "cache_control": dict(_EPHEMERAL)}]


def _messages_for(
    messages: Sequence[dict[str, str]], *, cache_prefix: bool
) -> list[dict[str, object]]:
    """Messages, with a breakpoint on the last message of the stable head.

    The head is everything but the final message, so the marker goes on index
    -2. With fewer than two messages there is no head to cache and the system
    block carries the only breakpoint.
    """
    head_index = len(messages) - 2 if cache_prefix and len(messages) >= 2 else -1
    built: list[dict[str, object]] = []
    for index, message in enumerate(messages):
        role = message.get("role", "user")
        content = message.get("content", "")
        if index == head_index:
            built.append(
                {
                    "role": role,
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": dict(_EPHEMERAL),
                        }
                    ],
                }
            )
        else:
            built.append({"role": role, "content": content})
    return built


def _usage_from(raw: object) -> TokenUsage:
    """Token counts as the provider reported them.

    Anthropic reports `input_tokens` net of cache activity, so the three input
    figures are disjoint and are carried across unchanged. No arithmetic is
    performed on them and nothing is derived from the length of any string.
    """
    usage = field(raw, "usage", None)
    if usage is None:
        raise ProviderResponseError(
            "the Anthropic response carried no usage object, so this call "
            "cannot be accounted for. Reporting it as zero tokens would make "
            "it free in every downstream total."
        )
    return TokenUsage(
        input_tokens=int_field(usage, "input_tokens"),
        output_tokens=int_field(usage, "output_tokens"),
        cache_read_tokens=int_field(usage, "cache_read_input_tokens"),
        cache_write_tokens=int_field(usage, "cache_creation_input_tokens"),
    )


def _text_from(raw: object) -> str:
    """Every text block, concatenated. Thinking and tool blocks are skipped."""
    content = field(raw, "content", None)
    if isinstance(content, str):
        return content
    if not isinstance(content, Sequence):
        return ""
    parts: list[str] = []
    for block in content:
        if str_field(block, "type", "") == "text":
            parts.append(str_field(block, "text", ""))
    return "".join(parts)
