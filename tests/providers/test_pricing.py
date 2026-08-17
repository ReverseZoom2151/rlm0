"""The price table, and specifically its refusal to price what it does not know."""

from __future__ import annotations

import logging

import pytest

from rlm0.providers import ModelPrice, PriceTable
from rlm0.run import TokenUsage


def test_unknown_model_costs_none_and_not_zero() -> None:
    """The whole point of the module: absence of a price is not a price of zero.

    A cost ceiling checked against a total that silently absorbs unpriced calls
    as 0.0 can never fire, which is how several surveyed implementations ended
    up with budgets that were believed and inert.
    """
    table = PriceTable()
    cost = table.cost_usd("some-model-nobody-priced", TokenUsage(1_000_000, 1_000_000))
    assert cost is None
    assert cost != 0.0


def test_unpriced_models_are_exposed_not_silent() -> None:
    table = PriceTable()
    table.cost_usd("mystery-1", TokenUsage(10, 10))
    table.cost_usd("mystery-2", TokenUsage(10, 10))
    table.cost_usd("mystery-1", TokenUsage(10, 10))
    assert table.unpriced_models == frozenset({"mystery-1", "mystery-2"})


def test_unpriced_model_logs_once(caplog: pytest.LogCaptureFixture) -> None:
    table = PriceTable()
    with caplog.at_level(logging.WARNING, logger="rlm0.providers.pricing"):
        table.cost_usd("mystery", TokenUsage(10, 10))
        table.cost_usd("mystery", TokenUsage(10, 10))
    warnings = [r for r in caplog.records if "mystery" in r.getMessage()]
    assert len(warnings) == 1


def test_known_model_prices_each_token_category_separately() -> None:
    table = PriceTable()
    usage = TokenUsage(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        cache_write_tokens=1_000_000,
    )
    cost = table.cost_usd("claude-sonnet-4-6", usage)
    assert cost is not None
    # 3.00 input + 15.00 output + 0.30 cache read + 3.75 cache write.
    assert cost == pytest.approx(22.05)


def test_cache_reads_are_cheaper_than_the_same_tokens_uncached() -> None:
    """The saving prefix caching is supposed to produce must show up in the price."""
    table = PriceTable()
    tokens = 500_000
    uncached = table.cost_usd("claude-opus-5", TokenUsage(input_tokens=tokens))
    cached = table.cost_usd("claude-opus-5", TokenUsage(cache_read_tokens=tokens))
    assert uncached is not None
    assert cached is not None
    assert cached < uncached


def test_overrides_add_a_model_without_mutating_the_original() -> None:
    base = PriceTable()
    extended = base.with_overrides(
        {"house-model": ModelPrice(input_usd_per_mtok=2.0, output_usd_per_mtok=4.0)}
    )
    assert base.cost_usd("house-model", TokenUsage(10, 10)) is None
    assert extended.cost_usd("house-model", TokenUsage(1_000_000, 0)) == 2.0


def test_overrides_replace_a_stale_published_price() -> None:
    base = PriceTable()
    cheaper = base.with_overrides(
        {"claude-opus-5": ModelPrice(input_usd_per_mtok=1.0, output_usd_per_mtok=1.0)}
    )
    assert cheaper.cost_usd("claude-opus-5", TokenUsage(1_000_000, 0)) == 1.0
    assert base.cost_usd("claude-opus-5", TokenUsage(1_000_000, 0)) == 5.0


def test_lookup_is_exact_so_a_snapshot_does_not_inherit_an_alias_price() -> None:
    table = PriceTable()
    assert table.get("claude-opus-5") is not None
    assert table.get("claude-opus-5-20260101") is None


def test_negative_prices_are_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ModelPrice(input_usd_per_mtok=-1.0, output_usd_per_mtok=1.0)
