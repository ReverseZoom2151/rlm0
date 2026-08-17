"""Tests for the escalation rule, which is mostly a set of refusals to spend.

The assertions that matter here are the negative ones. A policy that escalates
when the budget cannot fund the deeper attempt is not a slightly worse policy,
it is a policy that converts a clean shallow refusal into an expensive
truncated one, and that trade is the shape of the cost tail in every published
result on this idea.
"""

from __future__ import annotations

import pytest

from rlm0.policy import Escalating, Fixed, Never, can_fund_another_attempt
from rlm0.ports import CallReservation, DepthPolicy, EscalationContext
from rlm0.run import Outcome

RICH = CallReservation(
    granted=True, calls_remaining=50, usd_remaining=1.0, seconds_remaining=60.0
)
BROKE = CallReservation(granted=False, reason="call ceiling reached")
NEARLY_BROKE = CallReservation(granted=True, calls_remaining=1, usd_remaining=0.01)
UNKNOWN = CallReservation(granted=True)


def ctx(
    *,
    outcome: Outcome = Outcome.ITERATIONS_EXHAUSTED,
    attempts: int = 1,
    reservation: CallReservation = RICH,
    signals: dict[str, float] | None = None,
) -> EscalationContext:
    return EscalationContext(
        task="which supplier appears in both logs",
        context_chars=4_000_000,
        attempts_so_far=attempts,
        last_outcome=outcome.value,
        last_answer=None,
        spent_usd=0.01,
        elapsed_s=2.0,
        budget_reservation=reservation,
        signals=signals or {},
    )


def test_all_policies_satisfy_the_seam() -> None:
    for policy in (Never(), Fixed(2, experimental=True), Escalating()):
        assert isinstance(policy, DepthPolicy)
        assert policy.describe().strip()


class TestNever:
    def test_never_escalates_even_after_failure(self) -> None:
        assert Never().next_depth(ctx()) is None

    def test_never_escalates_with_money_to_burn(self) -> None:
        assert Never().next_depth(ctx(reservation=RICH, attempts=3)) is None


class TestFixed:
    def test_escalates_once_to_its_bound(self) -> None:
        assert Fixed(2, experimental=True).next_depth(ctx()) == 2

    def test_escalates_even_when_depth_zero_answered(self) -> None:
        # The ablation case: the deeper row is wanted on every task, including
        # the ones the control already handled, or the table has no control.
        assert Fixed(1).next_depth(ctx(outcome=Outcome.ANSWERED)) == 1

    def test_stops_after_one_deeper_attempt(self) -> None:
        assert Fixed(2, experimental=True).next_depth(ctx(attempts=2)) is None

    def test_refuses_when_the_budget_cannot_fund_the_attempt(self) -> None:
        assert Fixed(2, experimental=True).next_depth(ctx(reservation=BROKE)) is None
    assert Fixed(2, experimental=True).next_depth(ctx(reservation=NEARLY_BROKE)) is None

    def test_depth_zero_is_not_a_fixed_depth(self) -> None:
        with pytest.raises(ValueError, match="use Never"):
            Fixed(0)


class TestEscalating:
    def test_stops_when_the_previous_attempt_answered(self) -> None:
        assert Escalating().next_depth(ctx(outcome=Outcome.ANSWERED)) is None

    def test_steps_to_one_after_a_failed_control(self) -> None:
        assert Escalating().next_depth(ctx()) == 1

    def test_climbs_one_rung_per_failed_attempt(self) -> None:
        policy = Escalating(max_depth=3, experimental=True)
        assert policy.next_depth(ctx(attempts=2)) == 2
        assert policy.next_depth(ctx(attempts=3)) == 3
        assert policy.next_depth(ctx(attempts=4)) is None

    def test_respects_a_larger_step(self) -> None:
        assert Escalating(max_depth=4, step=2, experimental=True).next_depth(ctx()) == 2

    def test_stops_at_its_ceiling(self) -> None:
        assert Escalating(max_depth=0).next_depth(ctx()) is None

    def test_refuses_a_deep_attempt_the_budget_cannot_fund(self) -> None:
        assert Escalating().next_depth(ctx(reservation=BROKE)) is None
        assert Escalating().next_depth(ctx(reservation=NEARLY_BROKE)) is None

    def test_escalates_after_a_timeout(self) -> None:
        assert Escalating().next_depth(ctx(outcome=Outcome.TIMED_OUT)) == 1

    def test_does_not_escalate_after_an_environment_error(self) -> None:
        assert Escalating().next_depth(ctx(outcome=Outcome.ERRORED)) is None

    def test_can_be_told_to_escalate_after_an_error(self) -> None:
        policy = Escalating(stop_on_error=False)
        assert policy.next_depth(ctx(outcome=Outcome.ERRORED)) == 1

    def test_ignores_the_task_text_entirely(self) -> None:
        # Two contexts differing only in what the task says must decide the
        # same way, which is the property a task classifier would break.
        one = ctx()
        two = EscalationContext(
            task="summarise every incident in the archive by quarter",
            context_chars=one.context_chars,
            attempts_so_far=one.attempts_so_far,
            last_outcome=one.last_outcome,
            last_answer=one.last_answer,
            spent_usd=one.spent_usd,
            elapsed_s=one.elapsed_s,
            budget_reservation=one.budget_reservation,
        )
        assert Escalating().next_depth(one) == Escalating().next_depth(two)

    def test_rejects_a_ladder_that_never_climbs(self) -> None:
        with pytest.raises(ValueError, match="step"):
            Escalating(step=0)


class TestCanFund:
    def test_an_ungranted_reservation_funds_nothing(self) -> None:
        assert not can_fund_another_attempt(BROKE)

    def test_unknown_remainders_are_not_treated_as_zero(self) -> None:
        assert can_fund_another_attempt(UNKNOWN)

    def test_one_call_left_does_not_fund_a_recursive_attempt(self) -> None:
        assert not can_fund_another_attempt(
            CallReservation(granted=True, calls_remaining=1)
        )
        assert can_fund_another_attempt(
            CallReservation(granted=True, calls_remaining=1), min_calls=1
        )

    def test_spent_money_or_time_blocks_escalation(self) -> None:
        assert not can_fund_another_attempt(
            CallReservation(granted=True, calls_remaining=9, usd_remaining=0.0)
        )
        assert not can_fund_another_attempt(
            CallReservation(granted=True, calls_remaining=9, seconds_remaining=0.0)
        )
