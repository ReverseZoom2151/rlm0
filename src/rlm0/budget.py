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

A reservation is only as good as the estimate behind it, which is why
`FanOutEstimator` lives here rather than in the runtime. The published budget
work concedes four to six times static over-reservation and a little over two
times for its adaptive variant, and it concedes that because it is estimating
from outside: it does not know how many calls are about to be issued or how
much text each of them carries. A runtime that has just decided to fan out over
eleven slices knows both numbers exactly, and knows the tokens-per-character
that this run's own settled calls have been showing. Estimating from those
three facts plus the model's output ceiling is not a cleverer heuristic, it is
a better-informed one, and the estimator records the ratio it actually achieved
so the improvement is a measurement rather than another claim.

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
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rlm0.ports import CallReservation
from rlm0.run import TokenUsage

__all__ = [
    "BudgetSnapshot",
    "FanOutEstimate",
    "FanOutEstimator",
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
    calls_released: int = 0
    unmatched_releases: int = 0

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
        self._calls_released = 0
        self._unmatched_releases = 0
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

    def release(self, *, n_calls: int) -> None:
        """Give back calls that were granted and will never be made.

        The missing third verb. Reserve and settle alone cannot describe a
        caller that reserves two calls for one turn, holding the second so a
        wind-down is always fundable, and then makes only one of them: the
        unused half stays in flight forever, every ceiling is tested against
        it, and the run ends up bounded by something lower than the number
        `summary()` prints into the run record. The published budget lifecycle
        this project cites is reserve, reconcile and refund for exactly this
        reason.

        An explicit release rather than a reservation handle threaded through
        `settle`. Handles were the other candidate and they are strictly more
        precise, since each settlement would then release its own reservation's
        share rather than an equal one. They were rejected because `settle` is
        on the `Budget` port and is called from the provider-facing edge of the
        runtime, so adding a handle changes every implementation and every call
        site to fix a case that only the reserving caller can even detect. The
        caller that took the hold is the one that knows it went unused, so the
        refund belongs where that knowledge is.

        No token argument, for the reason `settle` gives: the provisional pool
        is shared and released in equal per-call shares, because nothing here
        can tell which reservation a given call belonged to. Releasing a call
        returns its share of what is still pending, which is the same
        arithmetic a settlement does, minus the spend.
        """
        if n_calls < 1:
            raise ValueError(f"a release must cover at least one call, got {n_calls}")
        with self._lock:
            if self._calls_in_flight <= 0:
                # Releasing what was never held would credit the run with
                # headroom nobody reserved, which is a ceiling that reads as
                # enforced and is not. Counted, not applied.
                self._unmatched_releases += n_calls
                return
            returned = min(n_calls, self._calls_in_flight)
            share_tokens = self._tokens_pending / self._calls_in_flight * returned
            share_usd = self._usd_pending / self._calls_in_flight * returned
            self._calls_in_flight -= returned
            self._tokens_pending -= share_tokens
            self._usd_pending -= share_usd
            self._calls_released += returned
            if self._calls_in_flight == 0:
                self._tokens_pending = 0.0
                self._usd_pending = 0.0
            if returned < n_calls:
                self._unmatched_releases += n_calls - returned

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
                calls_released=self._calls_released,
                unmatched_releases=self._unmatched_releases,
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
                flags += f", {self._unmatched_settlements} unreserved settlements"
            if self._unmatched_releases:
                flags += f", {self._unmatched_releases} unheld releases"
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


@dataclass(frozen=True, slots=True)
class FanOutEstimate:
    """How many tokens a batch is expected to need, and what said so.

    `basis` names where the tokens-per-character came from, because an estimate
    derived from this run's own settled calls and one derived from a shipped
    prior are the same number with very different standing, and a ratio
    reported without saying which one produced it cannot be read.
    """

    n_calls: int
    tokens: int
    input_tokens: int
    output_tokens: int
    tokens_per_char: float
    output_tokens_per_call: float
    basis: str

    def describe(self) -> str:
        return (
            f"{self.n_calls} call(s), ~{self.tokens} tokens "
            f"({self.input_tokens} in + {self.output_tokens} out) at "
            f"{self.tokens_per_char:.3f} tokens/char [{self.basis}]"
        )


class FanOutEstimator:
    """Sizes a batch reservation from what the dispatcher already knows.

    Four inputs, all available at the moment of dispatch and none of them a
    guess about the future:

    - the batch size, because the runtime has already decided how many children
      it is about to issue;
    - the characters going into each child, which is the shared prefix plus the
      slice that child was handed, and the slice is the thing the runtime just
      cut;
    - the tokens-per-character this run has actually been billed at, measured
      over calls that have already settled with provider-reported counts;
    - the model's output ceiling, which is a hard cap that no call can exceed.

    The output term is where a static estimator loses most of its accuracy. It
    has to assume every call runs to the ceiling, because it has no way to know
    better, and a 4096-token ceiling against a 300-token answer is a twelvefold
    error on that half of the estimate on its own. Here the mean observed output
    is used once there is anything to observe, and the ceiling is kept only as
    the cap it really is.

    `headroom` is applied last and is deliberately small. It exists because an
    estimate that lands under the truth turns a shared ceiling into an overshoot
    on the calls already in flight, which is the one error this module is not
    allowed to make. It is not a substitute for estimating well.
    """

    def __init__(
        self,
        *,
        prior_tokens_per_char: float = 0.28,
        prior_output_fraction: float = 0.35,
        headroom: float = 1.2,
        min_samples: int = 2,
    ) -> None:
        if prior_tokens_per_char <= 0.0:
            raise ValueError("prior_tokens_per_char must be positive")
        if not 0.0 < prior_output_fraction <= 1.0:
            raise ValueError("prior_output_fraction must be in (0.0, 1.0]")
        if headroom < 1.0:
            raise ValueError(
                "headroom below 1.0 estimates deliberately low, which spends a "
                "shared ceiling it did not reserve"
            )
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        self.prior_tokens_per_char = prior_tokens_per_char
        self.prior_output_fraction = prior_output_fraction
        self.headroom = headroom
        self.min_samples = min_samples

        self._lock = threading.Lock()
        self._samples = 0
        self._observed_chars = 0
        self._observed_input = 0
        self._observed_output = 0
        self._reserved_tokens = 0
        self._actual_tokens = 0
        self._batches = 0

    # -- learning from this run ------------------------------------------

    def observe(self, *, prompt_chars: int, usage: TokenUsage) -> None:
        """Fold one settled call into the run's own rate.

        Billed input over prompt characters, rather than uncached input over
        them. A cached call is billed for the same prefix at a different price,
        not for fewer tokens, so measuring against the uncached field alone
        would make the rate fall as the cache warmed and would then
        under-reserve exactly when a fan-out is widest.
        """
        if prompt_chars < 0:
            raise ValueError("prompt_chars cannot be negative")
        with self._lock:
            self._samples += 1
            self._observed_chars += prompt_chars
            self._observed_input += usage.billed_input
            self._observed_output += usage.output_tokens

    def estimate(
        self,
        *,
        batch_size: int,
        shared_prefix_chars: int,
        slice_chars: Sequence[int],
        max_output_tokens: int,
    ) -> FanOutEstimate:
        """Size one batch, before any of it is dispatched."""
        if batch_size < 1:
            raise ValueError("a batch covers at least one call")
        if max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if len(slice_chars) not in (0, batch_size):
            raise ValueError(
                f"got {len(slice_chars)} slice sizes for a batch of "
                f"{batch_size}; a per-child estimate needs one size per child "
                "or none at all"
            )
        if any(n < 0 for n in slice_chars):
            raise ValueError("slice sizes cannot be negative")

        with self._lock:
            enough = self._samples >= self.min_samples and self._observed_chars > 0
            if enough:
                rate = self._observed_input / self._observed_chars
                out_each = self._observed_output / self._samples
                basis = f"observed over {self._samples} settled call(s)"
            else:
                rate = self.prior_tokens_per_char
                out_each = self.prior_output_fraction * max_output_tokens
                basis = "prior, nothing settled yet"
        out_each = min(out_each, float(max_output_tokens))
        sizes = list(slice_chars) if slice_chars else [0] * batch_size
        in_chars = sum(shared_prefix_chars + size for size in sizes)
        input_tokens = math.ceil(in_chars * rate * self.headroom)
        output_tokens = math.ceil(out_each * batch_size * self.headroom)
        return FanOutEstimate(
            n_calls=batch_size,
            tokens=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tokens_per_char=rate,
            output_tokens_per_call=out_each,
            basis=basis,
        )

    # -- checking the claim ----------------------------------------------

    def record_batch(self, *, reserved_tokens: int, actual_tokens: int) -> None:
        """Log what a finished batch reserved against what it really used.

        Both totals are kept rather than a running mean of per-batch ratios,
        because a mean of ratios lets a tiny batch that over-reserved by a
        factor of ten outweigh a large one that was accurate, and it is the
        tokens that get billed.
        """
        if reserved_tokens < 0 or actual_tokens < 0:
            raise ValueError("token counts cannot be negative")
        with self._lock:
            self._batches += 1
            self._reserved_tokens += reserved_tokens
            self._actual_tokens += actual_tokens

    @property
    def over_reservation_ratio(self) -> float | None:
        """Tokens reserved over tokens actually spent, across every batch.

        None until a batch has both reserved and settled something. The number
        to beat is the 4x to 6x that the published static estimators concede,
        and the 2.11x their adaptive variant concedes.
        """
        with self._lock:
            if self._batches == 0 or self._actual_tokens == 0:
                return None
            return self._reserved_tokens / self._actual_tokens

    def describe(self) -> str:
        """One line for the run record, naming the ratio actually achieved."""
        ratio = self.over_reservation_ratio
        with self._lock:
            batches = self._batches
            reserved = self._reserved_tokens
            actual = self._actual_tokens
            samples = self._samples
        measured = "no batch measured yet" if ratio is None else f"{ratio:.2f}x"
        return (
            f"FanOutEstimator[{samples} sample(s)] over-reservation {measured} "
            f"({reserved} reserved / {actual} spent across {batches} batch(es))"
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

    def release(self, *, n_calls: int) -> None:
        """Nothing to give back, because nothing was held.

        Present so that a caller written against a budget that refunds does not
        have to ask whether this one does.
        """
        if n_calls < 1:
            raise ValueError(f"a release must cover at least one call, got {n_calls}")

    def remaining(self) -> CallReservation:
        """Every dimension None, because none of them is bounded.

        `granted` is False for the reason the port gives: nothing was taken.
        Callers deciding whether a further attempt fits must read the headroom
        fields, and here they say, correctly, that no ceiling exists to fit
        inside.
        """
        return CallReservation(
            granted=False, reason="unbounded: nothing was reserved, nothing bounds it"
        )

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
