"""The ceiling, and the reservation that makes it real.

The cost problem in recursive language models is not that calls are expensive.
It is that the number of them is decided by a model, at runtime, inside a tree
whose width nobody declared. The published cost distributions have standard
deviations at roughly double their means, and the tail is not random: cost
explodes exactly on the trajectories where the model cannot find the answer and
keeps spawning children to look for it. Cost and quality therefore fail
together, which removes the usual comfort that an expensive run at least bought
something.

Two design choices follow, and both are load bearing.

The budget is shared across the whole run rather than granted per level. A
per-level budget multiplies with fan-out: give each of eight children the
parent's allowance and the tree is authorised for eight times the parent's
spend without anybody having written that number down. That multiplication is
the mechanism behind the published tail.

Permission is taken before dispatch, not checked after. A counter inspected
after a call bounds nothing, it only reports. A fan-out that checks between
dispatches has already landed half its calls by the time it notices. So
`reserve` takes the lock, tests every ceiling against what is already spent
plus what is already in flight, and either provisionally debits the whole batch
or refuses it whole. `settle` then reconciles the provisional debit against
what actually happened.

Refusal is a signal rather than an exception. The runtime that gets refused
tells the model what is left and winds down to a final answer, which is a
strictly better outcome than an exception unwinding a tree that was halfway to
an answer. That is why `CallReservation` carries the remaining headroom, and
why a granted reservation can also carry an advisory: a model told it has two
calls left behaves differently from a model that is simply refused at the wall.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from rlm0.ports import CallReservation
from rlm0.run import TokenUsage

__all__ = [
    "BudgetSnapshot",
    "RunBudget",
    "Unbounded",
]

_log = logging.getLogger("rlm0.budget")


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """A consistent read of the ledger, for tests and for the run record.

    Taken under the lock and returned as a value, because handing out live
    counters would let a caller read a half-applied settlement and conclude the
    ledger had drifted when it had not.
    """

    max_usd: float | None
    max_seconds: float | None
    max_calls: int | None
    max_tokens: int | None
    calls_settled: int
    calls_in_flight: int
    tokens_settled: int
    tokens_pending: float
    usd_settled: float
    usd_pending: float
    unpriced_calls: int
    unmatched_settlements: int
    elapsed_s: float

    @property
    def calls_used(self) -> int:
        """Everything a further reservation has to make room after."""
        return self.calls_settled + self.calls_in_flight

    @property
    def tokens_used(self) -> float:
        return self.tokens_settled + self.tokens_pending

    @property
    def usd_used(self) -> float:
        return self.usd_settled + self.usd_pending


class RunBudget:
    """A ceiling shared by every call in one run, at every depth.

    Any subset of the four ceilings may be left unset, but at least one must be
    set. A `RunBudget` with nothing set is an unbounded run wearing the name of
    a bounded one, and the run record would then read as bounded to anybody who
    saw only `summary()`. Use `Unbounded` when that is genuinely what is
    wanted, because its summary says so.

    Unpriced cost, and why this class refuses rather than shrugs
    -----------------------------------------------------------
    `settle` accepts `cost_usd=None`, which means the provider returned no
    price and no price table covered the model. That is not zero. Treating it
    as zero is the specific defect that makes a USD ceiling unfireable: the
    ledger stays at $0.00 while real money is spent, so the ceiling is believed
    and never fires, which is worse than having no ceiling at all.

    The policy here is fail closed. When a USD ceiling is set and an unpriced
    settlement arrives, the budget records it, logs it at warning level once,
    and after `max_unpriced_calls` such settlements every subsequent
    reservation is refused with a reason naming the problem. The run then winds
    down through the ordinary exhaustion path and its answer survives.

    The alternatives were considered and rejected. Counting unpriced calls at
    zero is the defect being fixed. Raising from `settle` destroys the result
    of a call already paid for and turns a controlled wind-down into an
    exception in the middle of a fan-out. Substituting a guessed price makes
    the ledger a fiction that reads as a measurement, which is the same failure
    one layer further along. Refusing is the only option that keeps the ceiling
    enforceable, keeps the spend visible, and still lets the run finish.

    `max_unpriced_calls` defaults to 0, so the first unpriced settlement trips
    it. Raising it is a deliberate, recorded decision to spend an unknown
    amount, and `summary()` names the count either way. With no USD ceiling
    set there is nothing to enforce, so unpriced settlements are merely counted.

    What a ceiling can and cannot promise
    -------------------------------------
    The call ceiling is exact, because the number of calls in a batch is known
    before dispatch. Tokens and USD are exact only to the extent the caller's
    estimates are: actual usage is known after the call, so the ceiling can be
    overshot by the error on the calls that were already in flight. It cannot
    be overshot repeatedly, because once the settled ledger crosses the line
    every further reservation is refused. The wall clock ceiling stops new
    dispatch; it does not abort a call already running, which is the sandbox
    layer's job.
    """

    def __init__(
        self,
        *,
        max_usd: float | None = None,
        max_seconds: float | None = None,
        max_calls: int | None = None,
        max_tokens: int | None = None,
        max_unpriced_calls: int = 0,
        soft_fraction: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        for name, value in (
            ("max_usd", max_usd),
            ("max_seconds", max_seconds),
            ("max_calls", max_calls),
            ("max_tokens", max_tokens),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set, got {value}")
        if max_unpriced_calls < 0:
            raise ValueError("max_unpriced_calls cannot be negative")
        if not 0.0 <= soft_fraction < 1.0:
            raise ValueError("soft_fraction must be in [0.0, 1.0)")
        if (max_usd, max_seconds, max_calls, max_tokens) == (None, None, None, None):
            raise ValueError(
                "a RunBudget with no ceiling set is an unbounded run under a "
                "bounded name, and its summary would misreport the run. Set at "
                "least one ceiling, or use Unbounded, whose summary says plainly "
                "that nothing was bounded."
            )

        self.max_usd = max_usd
        self.max_seconds = max_seconds
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_unpriced_calls = max_unpriced_calls
        self.soft_fraction = soft_fraction

        self._clock = clock
        self._started_at = clock()
        self._lock = threading.Lock()

        # Settled is what happened. Pending is what has been granted and not
        # yet reported back. Every ceiling is tested against the sum, because
        # a fan-out of twenty in flight has already committed its cost.
        self._calls_settled = 0
        self._calls_in_flight = 0
        self._tokens_settled = 0
        self._tokens_pending = 0.0
        self._usd_settled = 0.0
        self._usd_pending = 0.0
        self._unpriced_calls = 0
        self._unmatched_settlements = 0
        self._warned_unpriced = False

    # -- the reservation -------------------------------------------------

    def reserve(
        self,
        *,
        n_calls: int,
        estimated_tokens: int,
        estimated_usd: float | None = None,
    ) -> CallReservation:
        """Take permission for a whole batch, before any of it is dispatched.

        All or nothing. A batch of twenty either lands twenty or lands zero.
        Granting twelve and refusing eight would be arithmetically tidier and
        operationally useless: the caller wrote code that fans out over twenty
        slices and a partial grant leaves it to discover mid-loop that eight of
        its slices have no answer, which is the between-dispatch check this
        module exists to remove. `CallReservation` also has no field for "how
        many of the twenty", so a partial grant would be indistinguishable on
        the wire from a full one. Callers who want fewer calls should reserve
        fewer calls; the loop that walks a batch size down is a caller-side
        policy and does not belong behind a lock.

        `estimated_usd` is optional and is the only way the USD ceiling can be
        enforced before dispatch rather than after. Callers with a price table
        should pass it. Callers without one get a ceiling that is checked
        against settlements, which is the best that can be done when the price
        is not knowable until the provider says so.

        A granted reservation may carry a non-empty `reason`: the soft-threshold
        advisory, meant to be shown to the model so it can start winding down
        before it hits the wall. Callers must branch on `granted`, never on
        whether `reason` is empty.
        """
        if n_calls < 1:
            raise ValueError(
                f"a reservation must cover at least one call, got {n_calls}"
            )
        if estimated_tokens < 0:
            raise ValueError("estimated_tokens cannot be negative")
        if estimated_usd is not None and estimated_usd < 0:
            raise ValueError("estimated_usd cannot be negative")

        with self._lock:
            refusal = self._refusal_locked(
                n_calls=n_calls,
                estimated_tokens=estimated_tokens,
                estimated_usd=estimated_usd or 0.0,
            )
            if refusal:
                return self._reservation_locked(granted=False, reason=refusal)
            self._calls_in_flight += n_calls
            self._tokens_pending += float(estimated_tokens)
            self._usd_pending += estimated_usd or 0.0
            return self._reservation_locked(
                granted=True, reason=self._advisory_locked()
            )

    def settle(self, usage: TokenUsage, cost_usd: float | None) -> None:
        """Reconcile one granted call against what it actually consumed.

        Exactly one call's worth of provisional debit is released per call, and
        the actual usage is added in its place. The provisional pool is shared
        rather than tracked per reservation because `settle` carries no handle
        back to the reservation that authorised it, so each settlement releases
        an equal share of what is still in flight. When the last in-flight call
        settles the pool is zeroed outright, which keeps the ledger free of
        floating point residue no matter how the batches interleaved.

        A settlement with nothing in flight is a bug in the caller, not in the
        budget. It is still charged, because the money was still spent, and it
        is counted separately so the summary can say the accounting was not
        clean rather than quietly absorbing it.
        """
        if cost_usd is not None and cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")

        with self._lock:
            if self._calls_in_flight > 0:
                share_tokens = self._tokens_pending / self._calls_in_flight
                share_usd = self._usd_pending / self._calls_in_flight
                self._calls_in_flight -= 1
                self._tokens_pending -= share_tokens
                self._usd_pending -= share_usd
                if self._calls_in_flight == 0:
                    self._tokens_pending = 0.0
                    self._usd_pending = 0.0
            else:
                self._unmatched_settlements += 1
            self._calls_settled += 1
            self._tokens_settled += usage.total
            if cost_usd is None:
                self._unpriced_calls += 1
                self._warn_unpriced_locked()
            else:
                self._usd_settled += cost_usd

    # -- what the runtime asks ------------------------------------------

    @property
    def exhausted(self) -> bool:
        """True when not even one more minimal call could be reserved."""
        with self._lock:
            return bool(
                self._refusal_locked(n_calls=1, estimated_tokens=0, estimated_usd=0.0)
            )

    @property
    def near_limit(self) -> bool:
        """True once any ceiling is inside its soft threshold.

        Separate from `exhausted` so the runtime can warn the model while it
        still has calls to spend. A model told two calls remain writes a
        different next block than one that is simply refused.
        """
        with self._lock:
            return bool(self._advisory_locked())

    def advisory(self) -> str:
        """The warning line for the model, or empty when there is headroom."""
        with self._lock:
            return self._advisory_locked()

    def remaining(self) -> CallReservation:
        """Headroom without taking any, for reporting and for wind-down text.

        Not granted, and deliberately shaped like a reservation so the caller
        that already knows how to render a refusal can render this too.
        """
        with self._lock:
            return self._reservation_locked(
                granted=False, reason="query only, nothing reserved"
            )

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return BudgetSnapshot(
                max_usd=self.max_usd,
                max_seconds=self.max_seconds,
                max_calls=self.max_calls,
                max_tokens=self.max_tokens,
                calls_settled=self._calls_settled,
                calls_in_flight=self._calls_in_flight,
                tokens_settled=self._tokens_settled,
                tokens_pending=self._tokens_pending,
                usd_settled=self._usd_settled,
                usd_pending=self._usd_pending,
                unpriced_calls=self._unpriced_calls,
                unmatched_settlements=self._unmatched_settlements,
                elapsed_s=self._elapsed_locked(),
            )

    def summary(self) -> str:
        """One line naming every ceiling and what has been spent against it.

        This string goes into the `Run` record and is what a reader sees when
        asking whether a run was bounded, so it names the unset ceilings too.
        A summary that mentioned only the ceilings that happened to be set
        would let a run with one loose bound read as fully bounded.
        """
        with self._lock:
            usd = "unset" if self.max_usd is None else f"${self.max_usd:.4f}"
            seconds = "unset" if self.max_seconds is None else f"{self.max_seconds:.0f}"
            calls = "unset" if self.max_calls is None else str(self.max_calls)
            tokens = "unset" if self.max_tokens is None else str(self.max_tokens)
            ceilings = f"usd={usd} seconds={seconds} calls={calls} tokens={tokens}"
            spent_usd = (
                f"${self._usd_settled:.4f}"
                if self._unpriced_calls == 0
                else f"${self._usd_settled:.4f}+{self._unpriced_calls} unpriced"
            )
            spent = (
                f"spent {spent_usd}, {self._calls_settled} calls, "
                f"{self._tokens_settled} tokens, {self._elapsed_locked():.1f}s"
            )
            flags = ""
            if self._unmatched_settlements:
                flags = f", {self._unmatched_settlements} unreserved settlements"
            return f"RunBudget[shared] {ceilings} ({spent}{flags})"

    # -- internals, all called with the lock held ------------------------

    def _elapsed_locked(self) -> float:
        return self._clock() - self._started_at

    def _seconds_remaining_locked(self) -> float | None:
        if self.max_seconds is None:
            return None
        return max(0.0, self.max_seconds - self._elapsed_locked())

    def _calls_remaining_locked(self) -> int | None:
        if self.max_calls is None:
            return None
        used = self._calls_settled + self._calls_in_flight
        return max(0, self.max_calls - used)

    def _usd_remaining_locked(self) -> float | None:
        if self.max_usd is None:
            return None
        used = self._usd_settled + self._usd_pending
        return max(0.0, self.max_usd - used)

    def _tokens_remaining_locked(self) -> float | None:
        if self.max_tokens is None:
            return None
        used = self._tokens_settled + self._tokens_pending
        return max(0.0, self.max_tokens - used)

    def _unpriced_blocks_locked(self) -> bool:
        return (
            self.max_usd is not None and self._unpriced_calls > self.max_unpriced_calls
        )

    def _warn_unpriced_locked(self) -> None:
        if self._warned_unpriced or self.max_usd is None:
            return
        self._warned_unpriced = True
        _log.warning(
            "unpriced call settled against a $%.4f ceiling: the provider "
            "returned no price and no table covered the model, so this run's "
            "cost ceiling can no longer be enforced from its own ledger. "
            "Unpriced calls are not free and are not counted as zero; "
            "reservations will be refused past max_unpriced_calls=%d.",
            self.max_usd,
            self.max_unpriced_calls,
        )

    def _refusal_locked(
        self, *, n_calls: int, estimated_tokens: int, estimated_usd: float
    ) -> str:
        """The reason this batch cannot be granted, or empty if it can."""
        if self._unpriced_blocks_locked():
            return (
                f"{self._unpriced_calls} call(s) settled with no price against a "
                f"${self.max_usd:.4f} ceiling, exceeding max_unpriced_calls="
                f"{self.max_unpriced_calls}. Unpriced is not free, so the ceiling "
                "can no longer be enforced and this budget refuses rather than "
                "run on an unenforceable bound. Wind down and report the "
                "unpriced spend."
            )
        seconds_left = self._seconds_remaining_locked()
        if seconds_left is not None and seconds_left <= 0.0:
            return (
                f"wall clock ceiling of {self.max_seconds:.0f}s reached after "
                f"{self._elapsed_locked():.1f}s"
            )
        calls_left = self._calls_remaining_locked()
        if calls_left is not None and n_calls > calls_left:
            return (
                f"batch of {n_calls} call(s) does not fit in {calls_left} "
                f"remaining of {self.max_calls}; a partial grant would land "
                "half a fan-out, so the whole batch is refused"
            )
        tokens_left = self._tokens_remaining_locked()
        if tokens_left is not None and (
            tokens_left <= 0.0 or estimated_tokens > tokens_left
        ):
            # The <= 0 arm matters because a call whose actual usage blew past
            # its estimate leaves no room, and a zero-token estimate would
            # otherwise slip through a ceiling that is already spent.
            return (
                f"estimated {estimated_tokens} token(s) does not fit in "
                f"{tokens_left:.0f} remaining of {self.max_tokens}"
            )
        usd_left = self._usd_remaining_locked()
        if usd_left is not None and (usd_left <= 0.0 or estimated_usd > usd_left):
            return (
                f"USD ceiling of ${self.max_usd:.4f} reached; ${usd_left:.4f} remains"
            )
        return ""

    def _advisory_locked(self) -> str:
        """The soft-threshold warning, phrased for the model that reads it."""
        if self.soft_fraction <= 0.0:
            return ""
        notes: list[str] = []
        calls_left = self._calls_remaining_locked()
        if calls_left is not None and self.max_calls is not None:
            threshold = math.ceil(self.max_calls * self.soft_fraction)
            if calls_left <= threshold:
                notes.append(f"{calls_left} of {self.max_calls} calls left")
        tokens_left = self._tokens_remaining_locked()
        if (
            tokens_left is not None
            and self.max_tokens is not None
            and tokens_left <= self.max_tokens * self.soft_fraction
        ):
            notes.append(f"{tokens_left:.0f} of {self.max_tokens} tokens left")
        usd_left = self._usd_remaining_locked()
        if (
            usd_left is not None
            and self.max_usd is not None
            and usd_left <= self.max_usd * self.soft_fraction
        ):
            notes.append(f"${usd_left:.4f} of ${self.max_usd:.4f} left")
        seconds_left = self._seconds_remaining_locked()
        if (
            seconds_left is not None
            and self.max_seconds is not None
            and seconds_left <= self.max_seconds * self.soft_fraction
        ):
            notes.append(f"{seconds_left:.1f}s of {self.max_seconds:.0f}s left")
        if not notes:
            return ""
        return "budget nearly spent: " + ", ".join(notes) + "; start winding down"

    def _reservation_locked(self, *, granted: bool, reason: str) -> CallReservation:
        return CallReservation(
            granted=granted,
            reason=reason,
            calls_remaining=self._calls_remaining_locked(),
            usd_remaining=self._usd_remaining_locked(),
            seconds_remaining=self._seconds_remaining_locked(),
        )


class Unbounded:
    """A budget that bounds nothing, and says so where it matters.

    For tests and local experimentation. It exists so that no run record can
    imply a ceiling that was never set: `summary()` leads with the word
    unbounded, and `Run` stores that string verbatim. The alternative, a
    `RunBudget` constructed with every ceiling None, would print a line full of
    the word unset that a reader could skim as bounded, which is why that
    construction is refused outright.

    The ledger is still kept. Knowing what an unbounded run cost is the whole
    reason anybody would want a ceiling on the next one.
    """

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._lock = threading.Lock()
        self._calls = 0
        self._tokens = 0
        self._usd = 0.0
        self._unpriced_calls = 0

    def reserve(
        self,
        *,
        n_calls: int,
        estimated_tokens: int,
        estimated_usd: float | None = None,
    ) -> CallReservation:
        """Always granted, with every remaining field None.

        None here means no ceiling exists, which is exactly what the runtime
        should tell the model. Reporting a large number instead would invent a
        bound that nobody set.
        """
        if n_calls < 1:
            raise ValueError(
                f"a reservation must cover at least one call, got {n_calls}"
            )
        return CallReservation(granted=True, reason="unbounded")

    def settle(self, usage: TokenUsage, cost_usd: float | None) -> None:
        with self._lock:
            self._calls += 1
            self._tokens += usage.total
            if cost_usd is None:
                self._unpriced_calls += 1
            else:
                self._usd += cost_usd

    @property
    def exhausted(self) -> bool:
        return False

    def summary(self) -> str:
        with self._lock:
            cost = (
                f"${self._usd:.4f}"
                if self._unpriced_calls == 0
                else f"${self._usd:.4f}+{self._unpriced_calls} unpriced"
            )
            elapsed = self._clock() - self._started_at
            return (
                "UNBOUNDED: no ceiling of any kind was set on this run "
                f"(spent {cost}, {self._calls} calls, {self._tokens} tokens, "
                f"{elapsed:.1f}s)"
            )
