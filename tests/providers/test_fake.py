"""The fake, which the rest of the project's tests will lean on entirely."""

from __future__ import annotations

import pytest

from rlm0.ports import LMClient
from rlm0.providers import FakeClient, FakeReply, PriceTable, ScriptExhaustedError


def test_satisfies_the_port() -> None:
    assert isinstance(FakeClient(), LMClient)


def test_scripted_replies_are_served_in_order() -> None:
    client = FakeClient(replies=[FakeReply("one"), FakeReply("two")])
    first = client.complete(
        system="s", messages=[{"role": "user", "content": "a"}],
        model="fake-model", max_tokens=64,
    )
    second = client.complete(
        system="s", messages=[{"role": "user", "content": "b"}],
        model="fake-model", max_tokens=64,
    )
    assert (first.text, second.text) == ("one", "two")


def test_running_out_of_script_is_an_error_not_a_repeat() -> None:
    client = FakeClient(replies=[FakeReply("only")])
    client.complete(
        system="", messages=[{"role": "user", "content": "a"}],
        model="fake-model", max_tokens=64,
    )
    with pytest.raises(ScriptExhaustedError):
        client.complete(
            system="", messages=[{"role": "user", "content": "b"}],
            model="fake-model", max_tokens=64,
        )


def test_a_default_reply_covers_an_unbounded_number_of_calls() -> None:
    client = FakeClient(default_reply=FakeReply("always"))
    for _ in range(5):
        client.complete(
            system="", messages=[{"role": "user", "content": "a"}],
            model="fake-model", max_tokens=64,
        )
    assert client.call_count == 5


def test_a_scripted_exception_is_raised() -> None:
    """So the layers above can be tested against provider failure without one."""
    client = FakeClient(replies=[RuntimeError("provider is down")])
    with pytest.raises(RuntimeError, match="provider is down"):
        client.complete(
            system="", messages=[{"role": "user", "content": "a"}],
            model="fake-model", max_tokens=64,
        )


def test_requests_are_recorded_for_assertion() -> None:
    client = FakeClient(default_reply=FakeReply("x"))
    client.complete(
        system="sys",
        messages=[{"role": "user", "content": "hello"}],
        model="fake-model",
        max_tokens=99,
        cache_prefix=True,
    )
    call = client.calls[0]
    assert call.system == "sys"
    assert call.messages == (("user", "hello"),)
    assert call.max_tokens == 99
    assert call.cache_prefix is True


def test_usage_is_whatever_the_script_said() -> None:
    client = FakeClient(
        replies=[FakeReply("x", input_tokens=1234, output_tokens=56)],
    )
    response = client.complete(
        system="", messages=[{"role": "user", "content": "a"}],
        model="fake-model", max_tokens=64,
    )
    assert response.usage.input_tokens == 1234
    assert response.usage.output_tokens == 56


def test_a_stop_reason_can_be_scripted_to_exercise_truncation() -> None:
    client = FakeClient(replies=[FakeReply("cut off", stop_reason="max_tokens")])
    response = client.complete(
        system="", messages=[{"role": "user", "content": "a"}],
        model="fake-model", max_tokens=4,
    )
    assert response.truncated is True


def test_the_fictional_fake_model_is_priced_so_budgets_have_numbers() -> None:
    client = FakeClient(
        replies=[FakeReply("x", input_tokens=1_000_000, output_tokens=0)],
    )
    response = client.complete(
        system="", messages=[{"role": "user", "content": "a"}],
        model="fake-model", max_tokens=64,
    )
    assert response.cost_usd == pytest.approx(1.0)


def test_an_unpriced_model_name_still_yields_none() -> None:
    """So the unpriced path itself is testable from the fake."""
    client = FakeClient(default_reply=FakeReply("x"), model="not-in-any-table")
    response = client.complete(
        system="", messages=[{"role": "user", "content": "a"}],
        model="not-in-any-table", max_tokens=64,
    )
    assert response.cost_usd is None


def test_a_repeated_prefix_reports_a_cache_read_on_the_second_call() -> None:
    """The fan-out shape: same system and head, different tail per child.

    A fixture that always reported zero cache reads could never exercise the
    code that notices a fan-out whose prefix is being invalidated, which is the
    failure this project most wants to be able to see.
    """
    client = FakeClient(default_reply=FakeReply("x", input_tokens=1000))
    system = "shared instructions, quite long so the prefix dominates" * 20
    head = {"role": "user", "content": "shared context " * 100}

    first = client.complete(
        system=system,
        messages=[head, {"role": "user", "content": "child one"}],
        model="fake-model",
        max_tokens=64,
        cache_prefix=True,
    )
    second = client.complete(
        system=system,
        messages=[head, {"role": "user", "content": "child two"}],
        model="fake-model",
        max_tokens=64,
        cache_prefix=True,
    )
    assert first.usage.cache_write_tokens > 0
    assert first.usage.cache_read_tokens == 0
    assert first.cached_prefix is False
    assert second.usage.cache_read_tokens > 0
    assert second.cached_prefix is True
    assert second.usage.billed_input == 1000


def test_a_changed_prefix_reports_no_cache_read() -> None:
    """Editing history invalidates, which is what callers are warned about."""
    client = FakeClient(default_reply=FakeReply("x", input_tokens=1000))
    for system in ("version one of the instructions", "version two, edited"):
        response = client.complete(
            system=system,
            messages=[
                {"role": "user", "content": "context"},
                {"role": "user", "content": "tail"},
            ],
            model="fake-model",
            max_tokens=64,
            cache_prefix=True,
        )
        assert response.usage.cache_read_tokens == 0


def test_simulation_is_off_when_the_flag_is_off() -> None:
    client = FakeClient(
        default_reply=FakeReply("x", input_tokens=1000),
        simulate_prefix_cache=False,
    )
    response = client.complete(
        system="s",
        messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
        model="fake-model",
        max_tokens=64,
        cache_prefix=True,
    )
    assert response.usage.input_tokens == 1000
    assert response.usage.cache_read_tokens == 0
    assert response.usage.cache_write_tokens == 0


def test_a_caller_supplied_price_table_wins() -> None:
    client = FakeClient(default_reply=FakeReply("x"), prices=PriceTable())
    response = client.complete(
        system="", messages=[{"role": "user", "content": "a"}],
        model="fake-model", max_tokens=64,
    )
    assert response.cost_usd is None
