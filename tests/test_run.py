"""Tests for the contract, which is mostly a set of refusals.

The interesting assertions here are the ones about what cannot be built. A
type whose invariants are only documented is a type whose invariants are
optional, and the specific invariant this project rests on, that a run carries
the control it is measured against, is exactly the kind that gets dropped
under deadline pressure unless the constructor refuses.
"""

from __future__ import annotations

import pytest

from rlm0.run import (
    Attempt,
    BaselineWaiver,
    CallRecord,
    Outcome,
    Role,
    Run,
    TokenUsage,
    Verdict,
)

GOOD_REASON = "the context far exceeds every model window available today"


def root_call(cost: float | None = 0.01, **kw: object) -> CallRecord:
    params: dict[str, object] = {
        "role": Role.ROOT,
        "depth": 0,
        "model": "test-model",
        "usage": TokenUsage(input_tokens=1000, output_tokens=50),
        "wall_clock_s": 1.0,
        "cost_usd": cost,
    }
    params.update(kw)
    return CallRecord(**params)  # type: ignore[arg-type]


def sub_call(cost: float | None = 0.005, depth: int = 1) -> CallRecord:
    return CallRecord(
        role=Role.SUB,
        depth=depth,
        model="test-model",
        usage=TokenUsage(input_tokens=500, output_tokens=20, cache_read_tokens=400),
        wall_clock_s=0.5,
        cost_usd=cost,
    )


def baseline(outcome: Outcome = Outcome.ITERATIONS_EXHAUSTED) -> Attempt:
    return Attempt(
        max_depth=0,
        outcome=outcome,
        calls=(root_call(),),
        wall_clock_s=1.0,
        answer="control answer" if outcome.produced_answer else None,
    )


def escalated(outcome: Outcome = Outcome.ANSWERED) -> Attempt:
    return Attempt(
        max_depth=1,
        outcome=outcome,
        calls=(root_call(), sub_call()),
        wall_clock_s=1.5,
        answer="deep answer" if outcome.produced_answer else None,
    )


class TestTheControlCannotBeSkipped:
    """The invariant the project is built on."""

    def test_a_run_without_depth_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no depth-zero control"):
            Run(task="t", attempts=(escalated(),), budget_summary="b")

    def test_the_control_must_come_first(self) -> None:
        # Present but second is not the same thing. If the deep attempt ran
        # first, the control was a second experiment rather than the cheap
        # path that happened to be tried first.
        with pytest.raises(ValueError, match="no depth-zero control"):
            Run(task="t", attempts=(escalated(), baseline()), budget_summary="b")

    def test_attempts_must_escalate(self) -> None:
        deeper = Attempt(
            max_depth=2, outcome=Outcome.TIMED_OUT, calls=(), wall_clock_s=0.1
        )
        with pytest.raises(ValueError, match="must escalate"):
            Run(
                task="t",
                attempts=(baseline(), deeper, escalated()),
                budget_summary="b",
            )

    def test_a_waiver_permits_the_run_but_poisons_the_verdict(self) -> None:
        run = Run(
            task="t",
            attempts=(escalated(),),
            budget_summary="b",
            waiver=BaselineWaiver(reason=GOOD_REASON, approved_by="adrian"),
        )
        assert run.baseline is None
        assert run.recursion_verdict().verdict is Verdict.UNTESTED

    def test_a_waiver_on_a_run_that_has_its_control_is_refused(self) -> None:
        # Otherwise a stale waiver sits on file and covers the next run that
        # quietly drops the comparison.
        with pytest.raises(ValueError, match="comparison that was made"):
            Run(
                task="t",
                attempts=(baseline(), escalated()),
                budget_summary="b",
                waiver=BaselineWaiver(reason=GOOD_REASON, approved_by="adrian"),
            )

    def test_a_run_must_record_its_budget(self) -> None:
        with pytest.raises(ValueError, match="budget"):
            Run(task="t", attempts=(baseline(),), budget_summary="   ")


class TestWaiverFriction:
    def test_a_short_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="actually be a reason"):
            BaselineWaiver(reason="too slow", approved_by="adrian")

    def test_padding_with_short_words_does_not_help(self) -> None:
        with pytest.raises(ValueError, match="actually be a reason"):
            BaselineWaiver(reason="it is a bit of an ok no", approved_by="adrian")

    def test_a_waiver_must_name_an_approver(self) -> None:
        with pytest.raises(ValueError, match="who approved it"):
            BaselineWaiver(reason=GOOD_REASON, approved_by="  ")


class TestAttemptInvariants:
    def test_a_depth_zero_attempt_cannot_hold_sub_calls(self) -> None:
        with pytest.raises(ValueError, match="cannot contain sub-calls"):
            Attempt(
                max_depth=0,
                outcome=Outcome.ANSWERED,
                calls=(root_call(), sub_call(depth=0)),
                wall_clock_s=1.0,
                answer="x",
            )

    def test_a_call_deeper_than_the_bound_is_refused(self) -> None:
        with pytest.raises(ValueError, match="bound was not enforced"):
            Attempt(
                max_depth=1,
                outcome=Outcome.ANSWERED,
                calls=(root_call(), sub_call(depth=2)),
                wall_clock_s=1.0,
                answer="x",
            )

    def test_an_answer_that_did_not_survive_its_stop_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not an answer"):
            Attempt(
                max_depth=1,
                outcome=Outcome.TIMED_OUT,
                calls=(),
                wall_clock_s=1.0,
                answer="half a thought",
            )

    def test_answering_without_an_answer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must carry its answer"):
            Attempt(
                max_depth=1, outcome=Outcome.ANSWERED, calls=(), wall_clock_s=1.0
            )

    def test_a_root_call_below_depth_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="root call sits at depth 0"):
            CallRecord(
                role=Role.ROOT,
                depth=1,
                model="m",
                usage=TokenUsage(),
                wall_clock_s=0.0,
            )


class TestCostHonesty:
    def test_an_unpriced_call_makes_the_attempt_unpriced(self) -> None:
        # Not a partial sum. A total that silently omits what it could not
        # price is the number a budget ceiling gets checked against, which is
        # how ceilings end up never firing.
        attempt = Attempt(
            max_depth=1,
            outcome=Outcome.ANSWERED,
            calls=(root_call(), sub_call(cost=None)),
            wall_clock_s=1.0,
            answer="x",
        )
        assert attempt.cost_usd is None

    def test_an_unpriced_attempt_makes_the_run_unpriced(self) -> None:
        attempt = Attempt(
            max_depth=1,
            outcome=Outcome.ANSWERED,
            calls=(sub_call(cost=None),),
            wall_clock_s=1.0,
            answer="x",
        )
        run = Run(task="t", attempts=(baseline(), attempt), budget_summary="b")
        assert run.cost_usd is None
        assert run.recursion_verdict().extra_cost_usd is None

    def test_priced_costs_sum_across_attempts(self) -> None:
        run = Run(task="t", attempts=(baseline(), escalated()), budget_summary="b")
        assert run.cost_usd == pytest.approx(0.01 + 0.01 + 0.005)


class TestUsage:
    def test_cache_tokens_are_billed_but_tracked_apart(self) -> None:
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=10,
            cache_read_tokens=900,
            cache_write_tokens=50,
        )
        assert usage.billed_input == 1050
        assert usage.total == 1060

    def test_a_zero_cache_ratio_is_visible_rather_than_absent(self) -> None:
        # The diagnostic only works if a fan-out that reads nothing from cache
        # reports 0.0 rather than None.
        attempt = Attempt(
            max_depth=1,
            outcome=Outcome.ANSWERED,
            calls=(
                root_call(),
                CallRecord(
                    role=Role.SUB,
                    depth=1,
                    model="m",
                    usage=TokenUsage(input_tokens=500, output_tokens=5),
                    wall_clock_s=0.1,
                    cost_usd=0.001,
                ),
            ),
            wall_clock_s=1.0,
            answer="x",
        )
        assert attempt.cache_read_ratio == 0.0

    def test_cache_ratio_is_none_when_nothing_was_billed(self) -> None:
        attempt = Attempt(
            max_depth=0, outcome=Outcome.TIMED_OUT, calls=(), wall_clock_s=0.1
        )
        assert attempt.cache_read_ratio is None

    def test_usage_separates_root_from_sub(self) -> None:
        run = Run(task="t", attempts=(baseline(), escalated()), budget_summary="b")
        by_role = run.usage_by_role()
        assert by_role[Role.ROOT].total == 2 * 1050
        assert by_role[Role.SUB].total == 920

    def test_negative_token_counts_are_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            TokenUsage(input_tokens=-1)


class TestVerdict:
    def test_recursion_helped_when_only_the_deeper_attempt_answered(self) -> None:
        run = Run(task="t", attempts=(baseline(), escalated()), budget_summary="b")
        verdict = run.recursion_verdict()
        assert verdict.verdict is Verdict.HELPED
        assert verdict.extra_sub_calls == 1
        assert verdict.extra_cost_usd == pytest.approx(0.015)

    def test_recursion_did_not_help_when_the_control_also_answered(self) -> None:
        run = Run(
            task="t",
            attempts=(baseline(Outcome.ANSWERED), escalated()),
            budget_summary="b",
        )
        assert run.recursion_verdict().verdict is Verdict.DID_NOT_HELP

    def test_not_attempted_when_depth_zero_sufficed(self) -> None:
        run = Run(
            task="t", attempts=(baseline(Outcome.ANSWERED),), budget_summary="b"
        )
        verdict = run.recursion_verdict()
        assert verdict.verdict is Verdict.NOT_ATTEMPTED
        assert verdict.extra_cost_usd == 0.0

    def test_the_reported_answer_is_the_last_one_that_survived(self) -> None:
        run = Run(
            task="t",
            attempts=(
                baseline(),
                escalated(),
                Attempt(
                    max_depth=2,
                    outcome=Outcome.BUDGET_EXHAUSTED,
                    calls=(),
                    wall_clock_s=0.1,
                ),
            ),
            budget_summary="b",
        )
        assert run.answer == "deep answer"


def test_describe_names_every_attempt_and_the_verdict() -> None:
    run = Run(
        task="who signed it",
        attempts=(baseline(), escalated()),
        budget_summary="max $0.50, 60s, 20 calls",
    )
    text = run.describe()
    assert "depth<=0" in text
    assert "depth<=1" in text
    assert "max $0.50" in text
    assert "recursion helped" in text
