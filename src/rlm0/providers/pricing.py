"""What a call costs, and an explicit refusal to guess when the model is unknown.

The rule this module exists to enforce is that an unpriced model produces None
and never zero. A survey of existing implementations of this idea found several
whose cost ceilings could never fire, because calls the price table did not
cover accumulated as 0.0 and the running total therefore never moved. That is
strictly worse than having no ceiling, because a ceiling that is present and
never fires is the one people trust.

Cache reads and writes are priced separately rather than folded into input,
because they really are billed at different rates and because a recursive
workload is mostly cache reads when it is working properly. Folding them
together would make the one number that shows whether prefix caching is
working invisible in the one place people look, which is the bill.

Prices are stated per million tokens and are dated in the comment above each
block. They are data, not truth: providers change them, this file will drift,
and the override path exists so that a caller who knows better does not have
to fork the package to say so.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from rlm0.run import TokenUsage

__all__ = [
    "ANTHROPIC_PRICES",
    "DEFAULT_PRICES",
    "OPENAI_PRICES",
    "ModelPrice",
    "PriceTable",
]

_log = logging.getLogger(__name__)

_PER_MILLION = 1_000_000.0


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """USD per million tokens, by the category the provider bills them under."""

    input_usd_per_mtok: float
    output_usd_per_mtok: float
    cache_read_usd_per_mtok: float = 0.0
    cache_write_usd_per_mtok: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "input_usd_per_mtok",
            "output_usd_per_mtok",
            "cache_read_usd_per_mtok",
            "cache_write_usd_per_mtok",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @classmethod
    def anthropic(cls, input_usd: float, output_usd: float) -> ModelPrice:
        """Anthropic's published cache multipliers applied to a base price.

        Reads bill at a tenth of input and five-minute writes at 1.25x, and
        those multipliers have held across model generations while the base
        prices have not. Deriving them means a price update is one number per
        model rather than four, which is the difference between a table that
        gets updated and one that quietly rots.
        """
        return cls(
            input_usd_per_mtok=input_usd,
            output_usd_per_mtok=output_usd,
            cache_read_usd_per_mtok=input_usd * 0.10,
            cache_write_usd_per_mtok=input_usd * 1.25,
        )

    @classmethod
    def openai(cls, input_usd: float, output_usd: float) -> ModelPrice:
        """OpenAI bills cached input at half, and does not bill a write at all.

        The zero write price is a fact about the provider rather than a
        placeholder: caching there is automatic and there is no write to pay
        for, which is also why the client cannot ask for it.
        """
        return cls(
            input_usd_per_mtok=input_usd,
            output_usd_per_mtok=output_usd,
            cache_read_usd_per_mtok=input_usd * 0.50,
            cache_write_usd_per_mtok=0.0,
        )

    def cost(self, usage: TokenUsage) -> float:
        """Price one usage record. Every field is charged at its own rate."""
        return (
            usage.input_tokens * self.input_usd_per_mtok
            + usage.output_tokens * self.output_usd_per_mtok
            + usage.cache_read_tokens * self.cache_read_usd_per_mtok
            + usage.cache_write_tokens * self.cache_write_usd_per_mtok
        ) / _PER_MILLION


# Anthropic list prices, USD per million tokens, as published on 2026-06-24.
# Sonnet 5 carries an introductory rate of $2.00 / $10.00 through 2026-08-31.
# The sticker price is used here on purpose: overstating a discount that has an
# expiry produces a cost report that silently becomes wrong on a known date,
# whereas overstating cost during the promotion is visible and conservative.
# Callers running inside the promotional window can override the entry.
ANTHROPIC_PRICES: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice.anthropic(10.00, 50.00),
    "claude-opus-5": ModelPrice.anthropic(5.00, 25.00),
    "claude-opus-4-8": ModelPrice.anthropic(5.00, 25.00),
    "claude-opus-4-7": ModelPrice.anthropic(5.00, 25.00),
    "claude-opus-4-6": ModelPrice.anthropic(5.00, 25.00),
    "claude-sonnet-5": ModelPrice.anthropic(3.00, 15.00),
    "claude-sonnet-4-6": ModelPrice.anthropic(3.00, 15.00),
    "claude-haiku-4-5": ModelPrice.anthropic(1.00, 5.00),
}

# OpenAI list prices, USD per million tokens, as published on 2026-06-24.
#
# This table is deliberately short. Only the two entries below are ones this
# author can state without hedging, and the instruction that governs this file
# is that an unpriced model must cost None rather than a plausible number. The
# reasoning models are the conspicuous omission: their prices have moved more
# than once and a stale entry for an expensive model is the worst possible
# error here, because it is wrong in the direction of looking cheap and it
# never announces itself. Calls to them are reported as unpriced, which shows
# up in `PriceTable.unpriced_models` and in a warning, and a caller who knows
# the current rate can supply it through `with_overrides`.
OPENAI_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice.openai(2.50, 10.00),
    "gpt-4o-mini": ModelPrice.openai(0.15, 0.60),
}

DEFAULT_PRICES: dict[str, ModelPrice] = {**ANTHROPIC_PRICES, **OPENAI_PRICES}


class PriceTable:
    """A lookup from model name to price, plus a record of what it could not price.

    The record is the point. An unpriced model is not an error and must not
    stop a run, but it does change what every total downstream means, so it has
    to be discoverable somewhere other than a diff between a report and an
    invoice. Every miss is logged once and accumulated in `unpriced_models`, so
    a harness can print the set at the end of a run and a reader can tell the
    difference between "this run cost nothing" and "this run was not priced".

    Lookup is by exact model name. Prefix matching would let a dated snapshot
    inherit the price of the alias it looks like, which is right until a
    provider reprices one and not the other, and wrong silently when they do.
    """

    __slots__ = ("_prices", "_unpriced")

    def __init__(self, prices: Mapping[str, ModelPrice] | None = None) -> None:
        self._prices: dict[str, ModelPrice] = (
            dict(DEFAULT_PRICES) if prices is None else dict(prices)
        )
        self._unpriced: set[str] = set()

    def __contains__(self, model: str) -> bool:
        return model in self._prices

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._prices))

    def __len__(self) -> int:
        return len(self._prices)

    def get(self, model: str) -> ModelPrice | None:
        """The price for a model, or None. Does not record a miss."""
        return self._prices.get(model)

    def cost_usd(self, model: str, usage: TokenUsage) -> float | None:
        """Price one call, or return None and remember that we could not.

        None rather than zero, always. See the module docstring for why that
        distinction is the whole reason this file exists.
        """
        price = self._prices.get(model)
        if price is None:
            if model not in self._unpriced:
                self._unpriced.add(model)
                _log.warning(
                    "no price for model %r; its calls will report cost_usd=None "
                    "rather than 0.0. Supply one with PriceTable.with_overrides "
                    "if you need cost totals for this model.",
                    model,
                )
            return None
        return price.cost(usage)

    def with_overrides(self, overrides: Mapping[str, ModelPrice]) -> PriceTable:
        """A new table with entries replaced or added. The original is untouched.

        Returning a new table rather than mutating in place keeps a client's
        pricing fixed for its lifetime, which matters because a cost total
        assembled from two different tables is not a number anybody can check.
        """
        merged = dict(self._prices)
        merged.update(overrides)
        return PriceTable(merged)

    @property
    def unpriced_models(self) -> frozenset[str]:
        """Every model this table was asked about and could not price."""
        return frozenset(self._unpriced)
