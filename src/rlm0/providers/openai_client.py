"""OpenAI chat completions, with the parameter differences handled explicitly.

Two things here differ from the Anthropic client, and both are properties of
the provider rather than choices.

CACHING IS NOT REQUESTABLE
--------------------------
OpenAI caches prompt prefixes automatically and offers no way to mark a block
cacheable, so `cache_prefix` is ignored. It is not emulated, and no request is
reshaped on the strength of it, because a client that pretended to honour the
flag would make the flag useless as a diagnostic everywhere else.

What is not ignored is the reported number. `prompt_tokens_details.cached_tokens`
is a real measurement of a real cache read, so it populates
`cache_read_tokens`, and `cached_prefix` follows it. `cache_write_tokens` stays
zero because there is no write to report and none is billed. The distinction
worth holding onto is that this layer never invents a cache field, but it does
report one the provider hands over unasked.

Note the arithmetic: OpenAI's `prompt_tokens` is inclusive of cached tokens,
where Anthropic's `input_tokens` is not. Carrying both straight across would
double count the cached portion in `TokenUsage.billed_input`, so the cached
count is subtracted out here and only here.

REASONING MODELS TAKE A DIFFERENT REQUEST
-----------------------------------------
`max_tokens` versus `max_completion_tokens`, and temperature being rejected
outright. Which shape a model wants is looked up in an explicit registry rather
than inferred from its name; `params.py` carries the argument for why.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from rlm0.ports import LMResponse
from rlm0.providers.errors import ProviderResponseError
from rlm0.providers.params import ChatParameterStyle, style_for
from rlm0.providers.payload import field, int_field, str_field
from rlm0.providers.pricing import PriceTable
from rlm0.providers.retry import RetryPolicy, call_with_retry
from rlm0.providers.sdk import load_sdk
from rlm0.run import TokenUsage

__all__ = ["OpenAIClient"]

_log = logging.getLogger(__name__)


class OpenAIClient:
    """An `LMClient` over the OpenAI chat completions API.

    `client` is injectable for the same reasons as on the Anthropic client, and
    additionally because the same protocol is spoken by several other services;
    a caller can pass an SDK instance pointed elsewhere. The price table will
    then not cover those model names, which is the correct outcome rather than
    a limitation: costs come back None instead of invented.
    """

    __slots__ = (
        "_client",
        "_extra",
        "_prices",
        "_retry",
        "_sleep",
        "_styles",
        "_temperature",
        "_warned_temperature",
    )

    def __init__(
        self,
        *,
        client: Any | None = None,
        api_key: str | None = None,
        prices: PriceTable | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        temperature: float | None = None,
        parameter_styles: Mapping[str, ChatParameterStyle] | None = None,
        extra_request: dict[str, object] | None = None,
    ) -> None:
        if client is None:
            sdk = load_sdk("openai", "openai")
            client = sdk.OpenAI(api_key=api_key) if api_key else sdk.OpenAI()
        self._client = client
        self._prices = PriceTable() if prices is None else prices
        self._retry = RetryPolicy() if retry_policy is None else retry_policy
        self._sleep = sleep
        self._temperature = temperature
        self._styles = parameter_styles
        self._warned_temperature: set[str] = set()
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
        # cache_prefix is accepted and ignored: see the module docstring. It is
        # read here only so that the signature is honest about doing nothing
        # with it rather than appearing to.
        del cache_prefix
        request = self.build_request(
            system=system,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        raw: object = call_with_retry(
            lambda: self._client.chat.completions.create(**request),
            policy=self._retry,
            sleep=self._sleep,
        )
        elapsed = time.perf_counter() - started

        usage = _usage_from(raw)
        answered_model = str_field(raw, "model", model)
        choice = _first_choice(raw)
        return LMResponse(
            text=_text_from(choice),
            usage=usage,
            model=answered_model,
            wall_clock_s=elapsed,
            cost_usd=self._prices.cost_usd(answered_model, usage),
            cached_prefix=usage.cache_read_tokens > 0,
            stop_reason=str_field(choice, "finish_reason", ""),
        )

    def build_request(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
    ) -> dict[str, object]:
        """The request body, with the output cap under whichever name applies."""
        style = style_for(model, self._styles)
        built: list[dict[str, str]] = []
        if system:
            built.append({"role": style.system_role, "content": system})
        for message in messages:
            built.append(
                {
                    "role": message.get("role", "user"),
                    "content": message.get("content", ""),
                }
            )
        request: dict[str, object] = dict(self._extra)
        request["model"] = model
        request["messages"] = built
        request[style.max_tokens_field] = max_tokens
        if self._temperature is not None:
            if style.supports_temperature:
                request["temperature"] = self._temperature
            elif model not in self._warned_temperature:
                self._warned_temperature.add(model)
                _log.warning(
                    "model %r does not accept a temperature, so the configured "
                    "value %.2f is being dropped from the request rather than "
                    "sent and rejected",
                    model,
                    self._temperature,
                )
        return request


def _first_choice(raw: object) -> object:
    """The first choice, or None. Only one completion is ever requested."""
    choices = field(raw, "choices", None)
    if isinstance(choices, Sequence) and not isinstance(choices, str) and choices:
        first: object = choices[0]
        return first
    return None


def _text_from(choice: object) -> str:
    """The assistant text, or empty when the model produced none."""
    message = field(choice, "message", None)
    return str_field(message, "content", "")


def _usage_from(raw: object) -> TokenUsage:
    """Token counts as reported, with the cached portion split out of the input.

    `prompt_tokens` includes cached tokens on this API, so leaving it whole
    while also recording `cache_read_tokens` would count the cached prefix
    twice in every total that sums the two.
    """
    usage = field(raw, "usage", None)
    if usage is None:
        raise ProviderResponseError(
            "the OpenAI response carried no usage object, so this call cannot "
            "be accounted for. Reporting it as zero tokens would make it free "
            "in every downstream total."
        )
    prompt_tokens = int_field(usage, "prompt_tokens")
    details = field(usage, "prompt_tokens_details", None)
    cached = int_field(details, "cached_tokens") if details is not None else 0
    cached = min(cached, prompt_tokens)
    return TokenUsage(
        input_tokens=prompt_tokens - cached,
        output_tokens=int_field(usage, "completion_tokens"),
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )
