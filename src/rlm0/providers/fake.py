"""A scripted client, so the rest of the project is testable without a key.

Everything above this layer, the REPL loop, the budget, the depth policy, the
run record, is only testable at all if a model call can be made to return a
known answer with known usage. Without that, the tests that matter, the ones
about escalation and cost ceilings and whether recursion helped, either do not
get written or get written against a live API and quietly stop running.

The numbers this client reports are fabricated. That is the one thing about it
that must stay obvious, because every other client in this package exists to
report numbers that are not. `FakeClient` is not a measurement and never
pretends to be one: it is a fixture that produces plausibly shaped usage so
that arithmetic downstream can be checked.

The prefix-cache simulation is here for one reason: the interesting bug in a
recursive workload is a cache read count that stays at zero across a fan-out,
and a fixture that always reports zero cache reads cannot exercise the code
that notices. So when asked to, this client tracks which prefixes it has seen
and moves tokens from input to cache-read on a repeat, the way a real provider
would. It splits the count by character share of the prefix, which is exactly
the length-derived estimate that would be dishonest in a real client and is
merely arbitrary in a fake.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rlm0.ports import LMResponse
from rlm0.providers.pricing import DEFAULT_PRICES, ModelPrice, PriceTable
from rlm0.run import TokenUsage

__all__ = [
    "FAKE_PRICES",
    "FakeCall",
    "FakeClient",
    "FakeReply",
    "ScriptExhaustedError",
]


class ScriptExhaustedError(RuntimeError):
    """The fake ran out of scripted replies and had no default to fall back on.

    An error rather than a repeat of the last reply, because a test that makes
    more calls than it scripted has found something, and silently serving it
    the previous answer is how that finding gets lost.
    """


# Fictional prices for the fictional models below, so that a test of the budget
# layer has non-None costs to add up. Round numbers on purpose: nothing here
# should ever be mistaken for a quote. One dollar per million in, ten out.
FAKE_PRICES: dict[str, ModelPrice] = {
    "fake-model": ModelPrice(
        input_usd_per_mtok=1.0,
        output_usd_per_mtok=10.0,
        cache_read_usd_per_mtok=0.1,
        cache_write_usd_per_mtok=1.25,
    ),
    "fake-model-small": ModelPrice(
        input_usd_per_mtok=0.1,
        output_usd_per_mtok=1.0,
        cache_read_usd_per_mtok=0.01,
        cache_write_usd_per_mtok=0.125,
    ),
}


@dataclass(frozen=True, slots=True)
class FakeReply:
    """One scripted answer, with the usage it should claim to have cost."""

    text: str
    input_tokens: int = 1000
    output_tokens: int = 100
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = "end_turn"
    model: str | None = None
    wall_clock_s: float = 0.0


@dataclass(frozen=True, slots=True)
class FakeCall:
    """What the client was asked for, recorded so a test can assert on it."""

    system: str
    messages: tuple[tuple[str, str], ...]
    model: str
    max_tokens: int
    cache_prefix: bool


@dataclass
class FakeClient:
    """An `LMClient` that returns scripted replies and records what it was asked.

    A script entry that is an exception is raised instead of returned, which is
    how the layers above get tested against provider failures without anyone
    having to construct a rate limit.
    """

    replies: Sequence[FakeReply | Exception] = ()
    model: str = "fake-model"
    default_reply: FakeReply | None = None
    simulate_prefix_cache: bool = True
    prices: PriceTable | None = None
    calls: list[FakeCall] = field(default_factory=list)

    _index: int = field(default=0, init=False, repr=False)
    _seen_prefixes: set[str] = field(default_factory=set, init=False, repr=False)
    _table: PriceTable = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._table = (
            PriceTable({**DEFAULT_PRICES, **FAKE_PRICES})
            if self.prices is None
            else self.prices
        )

    @property
    def price_table(self) -> PriceTable:
        """The table actually in use, including the fictional fake-model entries."""
        return self._table

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        cache_prefix: bool = False,
    ) -> LMResponse:
        self.calls.append(
            FakeCall(
                system=system,
                messages=tuple(
                    (m.get("role", "user"), m.get("content", "")) for m in messages
                ),
                model=model,
                max_tokens=max_tokens,
                cache_prefix=cache_prefix,
            )
        )
        reply = self._next_reply()
        usage = self._usage_for(
            reply, system=system, messages=messages, cache_prefix=cache_prefix
        )
        answered_model = reply.model or model
        return LMResponse(
            text=reply.text,
            usage=usage,
            model=answered_model,
            wall_clock_s=reply.wall_clock_s,
            cost_usd=self._table.cost_usd(answered_model, usage),
            cached_prefix=usage.cache_read_tokens > 0,
            stop_reason=reply.stop_reason,
        )

    def _next_reply(self) -> FakeReply:
        if self._index < len(self.replies):
            entry = self.replies[self._index]
            self._index += 1
            if isinstance(entry, Exception):
                raise entry
            return entry
        if self.default_reply is not None:
            return self.default_reply
        raise ScriptExhaustedError(
            f"FakeClient was called {len(self.calls)} times but only "
            f"{len(self.replies)} replies were scripted, and no default_reply "
            f"was given"
        )

    def _usage_for(
        self,
        reply: FakeReply,
        *,
        system: str,
        messages: list[dict[str, str]],
        cache_prefix: bool,
    ) -> TokenUsage:
        explicit_cache = reply.cache_read_tokens or reply.cache_write_tokens
        if not (cache_prefix and self.simulate_prefix_cache) or explicit_cache:
            return TokenUsage(
                input_tokens=reply.input_tokens,
                output_tokens=reply.output_tokens,
                cache_read_tokens=reply.cache_read_tokens,
                cache_write_tokens=reply.cache_write_tokens,
            )

        prefix = _prefix_key(system, messages)
        prefix_tokens = _prefix_share(system, messages, reply.input_tokens)
        remainder = reply.input_tokens - prefix_tokens
        if prefix in self._seen_prefixes:
            return TokenUsage(
                input_tokens=remainder,
                output_tokens=reply.output_tokens,
                cache_read_tokens=prefix_tokens,
            )
        self._seen_prefixes.add(prefix)
        return TokenUsage(
            input_tokens=remainder,
            output_tokens=reply.output_tokens,
            cache_write_tokens=prefix_tokens,
        )


def _prefix_key(system: str, messages: list[dict[str, str]]) -> str:
    """The part a real provider would match on: everything except the tail."""
    head = "\x00".join(
        f"{m.get('role', 'user')}\x01{m.get('content', '')}" for m in messages[:-1]
    )
    return f"{system}\x02{head}"


def _prefix_share(
    system: str, messages: list[dict[str, str]], input_tokens: int
) -> int:
    """How many of the claimed input tokens to attribute to the shared prefix.

    Character share, which is a length estimate and would be indefensible in a
    real client. Here it only has to be monotonic and stable so that the same
    request twice produces the same split.
    """
    head_chars = len(_prefix_key(system, messages))
    tail_chars = len(messages[-1].get("content", "")) if messages else 0
    total = head_chars + tail_chars
    if total == 0:
        return 0
    return min(input_tokens, int(input_tokens * head_chars / total))
