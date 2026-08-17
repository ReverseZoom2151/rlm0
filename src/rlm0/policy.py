"""Whether to try again deeper, decided from outcome and budget alone.

Every policy here is deliberately blind to what the task is about. The
temptation in this position is to classify the query and pick a depth from the
class, and the surveyed implementations that gave in to it are the ones whose
routing is inseparable from the benchmark it was tuned on: one spends an LLM
call labelling the task and then runs a fixed pipeline per label, another
regex-matches the model's generated code against patterns that only exist in
one dataset. Both report a win that no third party can reproduce off that
dataset.

So the inputs used here are the ones `EscalationContext` was kept small to
provide: did the previous attempt answer, and can the budget fund another go.
Anything richer belongs in `signals`, computed by whoever can afford it, with
the thing that computes it named.

The budget check is not a nicety. Escalating into a ceiling that cannot fund
the deeper attempt buys a truncated deep trajectory, which costs more than the
shallow attempt and answers less than it, and the published cost tail is made
almost entirely of trajectories like that.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlm0.ports import CallReservation, EscalationContext
from rlm0.run import Outcome

__all__ = [
    "Escalating",
    "Fixed",
    "Never",
    "can_fund_another_attempt",
]


def can_fund_another_attempt(
    reservation: CallReservation, *, min_calls: int = 2
) -> bool:
    """Whether the remaining budget can pay for a whole deeper attempt.

    A deeper attempt needs at least a root turn and the sub-call it exists to
    make, so a ceiling with one call left funds a deep attempt that stops
    before it can differ from the shallow one. Unknown remainders (None) are
    treated as no objection rather than as zero, because a budget that does not
    track a dimension is not the same as a budget with none of it left.

    `Budget.remaining()` returns `granted=False` because it takes nothing. A
    refused reservation also returns `granted=False`, so the two cases need a
    named distinction: a read carries the exact "query only" reason, while a
    refusal carries the reason a real reservation was denied. The headroom
    fields then decide whether the read can fund another attempt.
    """
    is_read = reservation.reason == "query only, nothing reserved"
    is_unbounded_read = reservation.reason.startswith("unbounded:")
    if not reservation.granted and not (is_read or is_unbounded_read):
        return False
    calls = reservation.calls_remaining
    if calls is not None and calls < min_calls:
        return False
    usd = reservation.usd_remaining
    if usd is not None and usd <= 0.0:
        return False
    seconds = reservation.seconds_remaining
    return seconds is None or seconds > 0.0


@dataclass(frozen=True, slots=True)
class Never:
    """Depth zero and nothing else.

    The pure control, and the one policy whose behaviour is already known to be
    competitive: the best-engineered open implementation of this idea reports
    that across every measured run its model never once chose to recurse, so a
    system that only ever runs depth zero reproduces those results exactly and
    for less money. Keeping it as a first-class policy means the comparison is
    a configuration rather than a code change.
    """

    def next_depth(self, context: EscalationContext) -> int | None:
        return None

    def describe(self) -> str:
        return "never (depth zero only)"


@dataclass(frozen=True, slots=True)
class Fixed:
    """One deeper attempt at a fixed bound, whatever depth zero did.

    Unconditional on outcome, and that is the point: this is the policy an
    ablation needs. Running depth zero and depth n on every task, including the
    tasks depth zero already answered, is what produces a table with a control
    row, and it is the only way to catch the case where the deeper attempt
    answers differently rather than merely later.

    It is not the default, because paying for a deep attempt after a successful
    shallow one is a measurement expense and not a way to serve a query.
    """

    depth: int
    min_calls_for_deeper: int = 2

    def __post_init__(self) -> None:
        if self.depth < 1:
            # Fixed(0) would be Never wearing a name that implies recursion,
            # which is exactly how a config file ends up claiming a depth the
            # run never had.
            raise ValueError("Fixed depth must be at least 1; use Never for depth zero")
        if self.min_calls_for_deeper < 1:
            raise ValueError("min_calls_for_deeper must be at least 1")

    def next_depth(self, context: EscalationContext) -> int | None:
        if context.attempts_so_far != 1:
            return None
        if not can_fund_another_attempt(
            context.budget_reservation, min_calls=self.min_calls_for_deeper
        ):
            return None
        return self.depth

    def describe(self) -> str:
        return f"fixed (depth 0, then depth {self.depth} unconditionally)"


@dataclass(frozen=True, slots=True)
class Escalating:
    """Start at zero, step deeper only after a failure the budget can fund.

    The default, and the reason the project is named after depth zero. Two
    conditions gate every step and both are conditions the run itself produced:
    the previous attempt did not answer, and the budget can still pay for a
    whole attempt rather than the front half of one.

    `stop_on_error` treats ERRORED as terminal. An error is the environment
    failing rather than the task being hard, and the deeper attempt would run
    in the same environment with more calls in it, so escalating on an error
    reliably converts one failure into a more expensive one.
    """

    max_depth: int = 2
    step: int = 1
    min_calls_for_deeper: int = 2
    stop_on_error: bool = True

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if self.step < 1:
            raise ValueError("step must be at least 1, or the ladder never climbs")
        if self.min_calls_for_deeper < 1:
            raise ValueError("min_calls_for_deeper must be at least 1")

    def next_depth(self, context: EscalationContext) -> int | None:
        if context.last_outcome == Outcome.ANSWERED:
            return None
        if self.stop_on_error and context.last_outcome == Outcome.ERRORED:
            return None
        # The runtime always opens at depth zero, so the number of attempts
        # closed so far is also the rung of the ladder to climb to next. The
        # policy owns the ladder rather than reading the last bound back out
        # of the context, which keeps the two from disagreeing.
        candidate = context.attempts_so_far * self.step
        if candidate > self.max_depth:
            return None
        if not can_fund_another_attempt(
            context.budget_reservation, min_calls=self.min_calls_for_deeper
        ):
            return None
        return candidate

    def describe(self) -> str:
        return (
            f"escalating (depth 0 first, +{self.step} per failed attempt, "
            f"max depth {self.max_depth})"
        )
