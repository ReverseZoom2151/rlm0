"""Gemini, where the cache is a resource with a clock on it rather than a token count.

The token accounting here is the same discipline as the other two clients:
every number comes from `usage_metadata` on the response, nothing is derived
from the length of a string, and a model no price table covers costs None
rather than zero. What is genuinely different is the cache, and it is different
in a way that does not fit the fields `TokenUsage` has.

TWO CACHES, NEITHER OF THEM A BREAKPOINT
----------------------------------------
Gemini has implicit caching, which is automatic on the 2.5 family and cannot be
requested, and explicit caching, which is a `cachedContents` resource created
out of band and then referenced by name on later requests. Neither is a
cache_control marker placed inside a request, so there is nothing for
`cache_prefix` to do. It is ignored rather than emulated, for the reason given
at length in `openai_client`: a client that reshaped a request on the strength
of a flag it cannot honour would make the flag useless as a diagnostic
everywhere else.

What is not ignored is the reported number. `cached_content_token_count` is a
real measurement of a real cache read, so it populates `cache_read_tokens` and
`cached_prefix` follows it, whichever of the two caches produced it. Note that
implicit caching only engages above a per-model minimum prefix length, so a
short shared prefix reporting zero cached tokens is the API working as
documented rather than a bug in the fan-out.

The arithmetic matches OpenAI rather than Anthropic: `prompt_token_count` is
inclusive of the cached portion, so the cached count is subtracted out of
`input_tokens` here, or `billed_input` would count the cached prefix twice.

WHY `cache_write_tokens` IS ALWAYS ZERO AND `cost_usd` IS ALWAYS NONE
--------------------------------------------------------------------
Anthropic bills a cache write once, per token, at a multiple of the input rate.
OpenAI does not bill a write at all. Gemini bills explicit cached content by
*storage time*: dollars per million tokens per hour, for as long as the
resource lives, whether or not anything reads it. That is a rate against a
duration, and `TokenUsage` has four token counts and no clock.

So there is no honest value to put in `cache_write_tokens`. Putting the cached
token count there would be worse than leaving it empty, because any per-token
price table would then multiply it by a write rate and produce a number that is
confidently wrong in an unknown direction. It stays zero, and the storage cost
is not representable in this type at all.

The consequence is that this client has no price table entries and every call
it makes reports `cost_usd=None`, recorded in `PriceTable.unpriced_models`. That
is deliberate on two counts. `pricing.py` already refuses to ship an entry its
author cannot state without hedging, because a stale price is wrong in the
direction of looking cheap and never announces itself. And here even a correct
per-token table would understate any run that used explicit caching, by exactly
the storage bill, which nothing in the response reports. A caller who knows the
current per-token rates can supply them through `PriceTable.with_overrides` and
should read the resulting totals as a floor rather than a total.

Creating and deleting the cached-content resource is left to the caller for the
same reason. A client that created one on the caller's behalf would be starting
a meter that nothing in this package stops.

THE REST OF THE SHAPE
---------------------
The system prompt travels in `config.system_instruction` rather than as a turn,
the assistant role is spelled `model`, the output cap is `max_output_tokens`,
and the finish reason is an enum spelled `MAX_TOKENS`. That last one is worth
naming: `LMResponse.truncated` keys on `max_tokens` and `length`, so carrying
the provider's spelling across unchanged would leave truncation invisible,
which is precisely the failure that property exists to catch.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

from rlm0.ports import LMResponse
from rlm0.providers.errors import ProviderDependencyError, ProviderResponseError
from rlm0.providers.payload import field, int_field, str_field
from rlm0.providers.pricing import PriceTable
from rlm0.providers.retry import (
    RETRYABLE_STATUS_CODES,
    RetryPolicy,
    call_with_retry,
    is_retryable,
)
from rlm0.run import TokenUsage

__all__ = ["GeminiClient", "gemini_retryable", "load_genai"]

_MODULE: Final = "google.genai"
_DISTRIBUTION: Final = "google-genai"

_ROLES: Final[Mapping[str, str]] = {
    "assistant": "model",
    "model": "model",
    "system": "user",
    "user": "user",
}
"""How the roles this project speaks map onto the two Gemini accepts.

`system` maps to `user` rather than being dropped, because a caller that put
instructions in a message meant them to arrive. The system prompt proper goes
in `system_instruction` and never through here.
"""


def load_genai() -> Any:
    """Import the Gemini SDK, or say what to install.

    Not `providers.sdk.load_sdk`, which builds its pip command out of the
    module name. Here the module is `google.genai` and the distribution is
    `google-genai`, so that message would print a command that does not work,
    and an install instruction that fails is worse than none.
    """
    try:
        return importlib.import_module(_MODULE)
    except ImportError as exc:
        raise ProviderDependencyError(
            f"the {_MODULE!r} module is required for this client but is not "
            f"installed. Install it with:\n"
            f"    pip install {_DISTRIBUTION}\n"
            f"Note that the module and the distribution are spelled "
            f"differently; {_MODULE!r} is not a package name."
        ) from exc


def gemini_retryable(exc: BaseException) -> bool:
    """Whether a Gemini SDK failure is worth trying again.

    The shared classifier reads the status off `status_code`, directly or on a
    response. This SDK puts it on `code` instead, so the shared classifier sees
    no status at all and falls through to a class-name check that
    `google.genai.errors.APIError` does not match. The visible symptom would be
    a 429 treated as fatal, which is the exact failure `retry.py` exists to
    prevent, hours into a sweep.

    The `Retry-After` header is still honoured, because the SDK carries the
    response on the exception and the shared `retry_after_seconds` reads
    headers from there. Google additionally duplicates the hint inside the
    error body as a `RetryInfo` detail; that form is not parsed here, and when
    no header is present the policy's own backoff applies.
    """
    code = field(exc, "code", None)
    if isinstance(code, int) and not isinstance(code, bool):
        return code in RETRYABLE_STATUS_CODES
    return is_retryable(exc)


class GeminiClient:
    """An `LMClient` over the Gemini `generate_content` API.

    `client` is injectable for the same reasons as on the other two clients:
    it is how these tests run with no SDK, no key and no network, and it is the
    seam for a caller who has already configured a client, for instance one
    pointed at Vertex.

    `cached_content` names an explicit `cachedContents` resource to reference
    on every call. It is a constructor argument rather than a per-call one
    because `LMClient.complete` has a fixed signature, and a per-client value
    is the right granularity anyway: the resource is created against a
    particular prefix, and a client that switched between them mid-fan-out
    would be reading a cache written for a different prompt.
    """

    __slots__ = (
        "_cached_content",
        "_client",
        "_extra",
        "_prices",
        "_retry",
        "_sleep",
        "_temperature",
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
        cached_content: str | None = None,
        extra_config: dict[str, object] | None = None,
    ) -> None:
        if client is None:
            sdk = load_genai()
            client = sdk.Client(api_key=api_key) if api_key else sdk.Client()
        self._client = client
        self._prices = PriceTable() if prices is None else prices
        self._retry = RetryPolicy() if retry_policy is None else retry_policy
        self._sleep = sleep
        self._temperature = temperature
        self._cached_content = cached_content
        self._extra: dict[str, object] = dict(extra_config or {})

    @property
    def prices(self) -> PriceTable:
        """The table in use, so a caller can read back what it failed to price."""
        return self._prices

    @property
    def cached_content(self) -> str | None:
        """The explicit cache resource referenced, if any.

        Readable so that a caller can record which resource a run was billed
        storage for. That charge does not appear in `cost_usd` and cannot, so
        the name is the only trace of it this layer can offer.
        """
        return self._cached_content

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        cache_prefix: bool = False,
    ) -> LMResponse:
        # Accepted and ignored: neither Gemini cache is requestable per block.
        # Read here so the signature is honest about doing nothing with it
        # rather than appearing to. See the module docstring.
        del cache_prefix
        request = self.build_request(
            system=system,
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        started = time.perf_counter()
        raw: object = call_with_retry(
            lambda: self._client.models.generate_content(**request),
            policy=self._retry,
            sleep=self._sleep,
            retryable=gemini_retryable,
        )
        # Wall clock includes any backoff slept through, as on the other two
        # clients: time spent waiting on a rate limit is time the run spent.
        elapsed = time.perf_counter() - started

        usage = _usage_from(raw)
        answered_model = str_field(raw, "model_version", model)
        candidate = _first_candidate(raw)
        return LMResponse(
            text=_text_from(candidate),
            usage=usage,
            model=answered_model,
            wall_clock_s=elapsed,
            # None for every Gemini model today, on purpose. The module
            # docstring says why, and `unpriced_models` records it.
            cost_usd=self._prices.cost_usd(answered_model, usage),
            # Reported, not requested.
            cached_prefix=usage.cache_read_tokens > 0,
            stop_reason=_stop_reason_from(candidate),
        )

    def build_request(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
    ) -> dict[str, object]:
        """The call kwargs, exposed so a test can assert where everything landed.

        Plain dicts rather than the SDK's typed config objects, so that nothing
        in this package has a static dependency on the SDK and so that a test
        double can be a dict. The SDK accepts a mapping wherever it accepts a
        `GenerateContentConfig`.
        """
        config: dict[str, object] = dict(self._extra)
        config["max_output_tokens"] = max_tokens
        if system:
            config["system_instruction"] = system
        if self._temperature is not None:
            config["temperature"] = self._temperature
        if self._cached_content is not None:
            config["cached_content"] = self._cached_content
        return {
            "model": model,
            "contents": _contents_for(messages),
            "config": config,
        }


def _contents_for(messages: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """Turns of `{role, parts}`, in the order given and with nothing reordered.

    Order is load bearing even though no breakpoint is placed here: both Gemini
    caches match on a prefix from the first token, so the same rules apply as
    in `anthropic_client`. Keep the shared part first, append rather than edit,
    and put the slice that differs per child in the final turn.
    """
    return [
        {
            "role": _ROLES.get(message.get("role", "user"), "user"),
            "parts": [{"text": message.get("content", "")}],
        }
        for message in messages
    ]


def _first_candidate(raw: object) -> object:
    """The first candidate, or None. Only one is ever requested."""
    candidates = field(raw, "candidates", None)
    if (
        isinstance(candidates, Sequence)
        and not isinstance(candidates, str)
        and candidates
    ):
        first: object = candidates[0]
        return first
    return None


def _text_from(candidate: object) -> str:
    """Every answer part, concatenated. Thought parts are skipped.

    A thought summary arrives as a part like any other, distinguished only by a
    `thought` flag. Concatenating it into the answer would feed the model's
    reasoning about the task back in as though it were the result, which the
    parse layer above would then try to read an answer out of.
    """
    content = field(candidate, "content", None)
    parts = field(content, "parts", None)
    if not isinstance(parts, Sequence) or isinstance(parts, str):
        return ""
    collected: list[str] = []
    for part in parts:
        if field(part, "thought", None) is True:
            continue
        collected.append(str_field(part, "text", ""))
    return "".join(collected)


def _stop_reason_from(candidate: object) -> str:
    """The finish reason, normalised to the spelling `LMResponse` understands.

    Gemini answers with an enum whose value is `MAX_TOKENS`, and depending on
    how it is stringified it can arrive as `FinishReason.MAX_TOKENS`. Neither
    is in the set `LMResponse.truncated` checks, so carrying it across
    unchanged would report every truncated sub-call as a complete one. A
    truncated answer looks exactly like a short answer to everything
    downstream, which is the whole reason that property exists.
    """
    raw = field(candidate, "finish_reason", None)
    if raw is None:
        return ""
    name = field(raw, "name", None)
    text = name if isinstance(name, str) else str(raw)
    return text.rsplit(".", 1)[-1].strip().lower()


def _usage_from(raw: object) -> TokenUsage:
    """Token counts as reported, with two adjustments and no estimates.

    The cached count is subtracted out of the prompt count, because
    `prompt_token_count` is inclusive of it and carrying both across whole
    would double count the cached prefix in `billed_input`.

    Thought tokens are added to the output count. They are reported separately
    from `candidates_token_count` and are billed at the output rate, so a total
    that omits them understates the bill by exactly the part of it that grows
    when a model is asked to think harder.

    `cache_write_tokens` stays zero. Gemini bills explicit cached content by
    storage time rather than per written token, so there is no token count that
    corresponds to a write, and inventing one would give every price table
    something wrong to multiply.
    """
    usage = field(raw, "usage_metadata", None)
    if usage is None:
        raise ProviderResponseError(
            "the Gemini response carried no usage_metadata, so this call "
            "cannot be accounted for. Reporting it as zero tokens would make "
            "it free in every downstream total."
        )
    prompt_tokens = int_field(usage, "prompt_token_count")
    cached = min(int_field(usage, "cached_content_token_count"), prompt_tokens)
    output_tokens = int_field(usage, "candidates_token_count") + int_field(
        usage, "thoughts_token_count"
    )
    return TokenUsage(
        input_tokens=prompt_tokens - cached,
        output_tokens=output_tokens,
        cache_read_tokens=cached,
        cache_write_tokens=0,
    )
