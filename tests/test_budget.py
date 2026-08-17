"""Tests for the shared run budget.

The concurrency test is the one that matters. Everything else here checks
arithmetic that would be obvious from reading the module, whereas a ceiling
that holds in a single thread and leaks under a fan-out is exactly the bug this
module exists to prevent, and it is invisible without threads. Sub-calls are
dispatched concurrently by construction, so a budget tested only sequentially
has not been tested in the conditions it will actually run in.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from rlm0.budget import RunBudget, Unbounded
from rlm0.ports import Budget, CallReservation
from rlm0.run import TokenUsage


class FakeClock:
    """A clock the test moves by hand.

    Wall clock ceilings tested against real time are either slow or flaky, and
    a budget whose deadline can only be exercised by sleeping will end up
    untested.
    """

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def usage(total: int) -> TokenUsage:
    return TokenUsage(input_tokens=total)


# -- the contract --------------------------------------------------------


def test_run_budget_satisfies_the_budget_protocol() -> None:
    budget = RunBudget(max_calls=4)
    assert isinstance(budget, Budget)
    # Static compatibility too, which is what the runtime actually relies on.
    as_port: Budget = budget
    assert as_port.reserve(n_calls=1, estimated_tokens=0).granted


def test_unbounded_satisfies_the_budget_protocol() -> None:
    budget = Unbounded()
    assert isinstance(budget, Budget)
    as_port: Budget = budget
    assert as_port.reserve(n_calls=99, estimated_tokens=10**9).granted


def test_a_budget_with_no_ceilings_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="no ceiling set"):
        RunBudget()


@pytest.mark.parametrize(
    "build",
    [
        lambda: RunBudget(max_calls=0),
        lambda: RunBudget(max_usd=-1.0),
        lambda: RunBudget(max_seconds=0.0),
        lambda: RunBudget(max_tokens=-5),
    ],
)
def test_non_positive_ceilings_are_refused(build: Callable[[], RunBudget]) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build()


# -- reservation before dispatch ----------------------------------------


def test_reservation_debits_before_the_call_happens() -> None:
    budget = RunBudget(max_calls=3)
    assert budget.reserve(n_calls=2, estimated_tokens=0).granted
    # Nothing has settled, but the ceiling already accounts for the two in
    # flight. This is the whole mechanic: a second fan-out cannot be authorised
    # against headroom the first one has already committed.
    assert budget.snapshot().calls_in_flight == 2
    assert budget.reserve(n_calls=2, estimated_tokens=0).granted is False


def test_a_refused_batch_lands_nothing() -> None:
    budget = RunBudget(max_calls=20)
    assert budget.reserve(n_calls=12, estimated_tokens=0).granted
    refused = budget.reserve(n_calls=20, estimated_tokens=0)
    assert refused.granted is False
    assert refused.calls_remaining == 8
    # Not 12 granted and 8 refused: the ledger is untouched by the refusal.
    assert budget.snapshot().calls_used == 12
    assert budget.reserve(n_calls=8, estimated_tokens=0).granted


def test_zero_call_reservations_are_a_caller_bug() -> None:
    budget = RunBudget(max_calls=3)
    with pytest.raises(ValueError, match="at least one call"):
        budget.reserve(n_calls=0, estimated_tokens=0)


def test_token_ceiling_is_checked_against_the_estimate() -> None:
    budget = RunBudget(max_tokens=1000)
    assert budget.reserve(n_calls=1, estimated_tokens=900).granted
    assert budget.reserve(n_calls=1, estimated_tokens=200).granted is False
    assert budget.reserve(n_calls=1, estimated_tokens=100).granted


def test_usd_ceiling_can_be_enforced_before_dispatch_when_priced() -> None:
    budget = RunBudget(max_usd=1.0)
    assert budget.reserve(n_calls=1, estimated_tokens=0, estimated_usd=0.9).granted
    refused = budget.reserve(n_calls=1, estimated_tokens=0, estimated_usd=0.5)
    assert refused.granted is False
    assert refused.usd_remaining == pytest.approx(0.1)


def test_wall_clock_ceiling_refuses_new_dispatch() -> None:
    clock = FakeClock()
    budget = RunBudget(max_seconds=60.0, clock=clock)
    granted = budget.reserve(n_calls=1, estimated_tokens=0)
    assert granted.granted
    assert granted.seconds_remaining == pytest.approx(60.0)
    clock.advance(61.0)
    refused = budget.reserve(n_calls=1, estimated_tokens=0)
    assert refused.granted is False
    assert "wall clock" in refused.reason
    assert refused.seconds_remaining == 0.0
    assert budget.exhausted


def test_unset_ceilings_report_none_rather_than_a_number() -> None:
    budget = RunBudget(max_calls=5)
    reservation = budget.reserve(n_calls=1, estimated_tokens=0)
    assert reservation.calls_remaining == 4
    assert reservation.usd_remaining is None
    assert reservation.seconds_remaining is None


# -- settlement ----------------------------------------------------------


def test_settle_releases_the_provisional_debit() -> None:
    budget = RunBudget(max_tokens=1000)
    assert budget.reserve(n_calls=2, estimated_tokens=800).granted
    budget.settle(usage(100), 0.01)
    budget.settle(usage(100), 0.01)
    snap = budget.snapshot()
    assert snap.calls_in_flight == 0
    assert snap.tokens_pending == 0.0
    assert snap.tokens_settled == 200
    # The 800 estimate is gone, so the room the calls did not use is available.
    assert budget.reserve(n_calls=1, estimated_tokens=700).granted


def test_an_overspending_call_closes_the_budget_for_the_next_one() -> None:
    budget = RunBudget(max_tokens=1000)
    assert budget.reserve(n_calls=1, estimated_tokens=100).granted
    budget.settle(usage(5000), 0.02)  # the estimate was badly wrong
    assert budget.exhausted
    assert budget.reserve(n_calls=1, estimated_tokens=1).granted is False


def test_settling_without_a_reservation_is_still_charged_and_flagged() -> None:
    budget = RunBudget(max_calls=5)
    budget.settle(usage(10), 0.5)
    snap = budget.snapshot()
    assert snap.calls_settled == 1
    assert snap.unmatched_settlements == 1
    assert "unreserved settlements" in budget.summary()


def test_negative_cost_is_refused() -> None:
    budget = RunBudget(max_usd=1.0)
    with pytest.raises(ValueError, match="cost_usd cannot be negative"):
        budget.settle(usage(1), -0.01)


# -- the unpriced cost policy -------------------------------------------


def test_unpriced_cost_is_not_treated_as_zero() -> None:
    """The defect being fixed: a USD ceiling that can never fire.

    An unpriced settlement leaves the ledger unable to say what was spent. A
    budget that carries on regardless has a ceiling in name only, so this one
    refuses further reservations and the run winds down.
    """
    budget = RunBudget(max_usd=10.0)
    assert budget.reserve(n_calls=1, estimated_tokens=0).granted
    budget.settle(usage(1000), None)
    assert budget.exhausted
    refused = budget.reserve(n_calls=1, estimated_tokens=0)
    assert refused.granted is False
    assert "no price" in refused.reason
    # And the ledger says so out loud rather than reporting a tidy $0.0000.
    assert "unpriced" in budget.summary()


def test_unpriced_cost_is_logged_once_and_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    budget = RunBudget(max_usd=10.0)
    with caplog.at_level("WARNING", logger="rlm0.budget"):
        budget.settle(usage(1), None)
        budget.settle(usage(1), None)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "no price" in warnings[0].getMessage()


def test_unpriced_calls_are_tolerated_only_up_to_the_declared_allowance() -> None:
    budget = RunBudget(max_usd=10.0, max_unpriced_calls=2)
    for _ in range(2):
        budget.settle(usage(1), None)
    assert budget.exhausted is False
    budget.settle(usage(1), None)
    assert budget.exhausted


def test_unpriced_cost_without_a_usd_ceiling_is_merely_counted() -> None:
    """Nothing to enforce, so nothing to refuse.

    The policy is about an unenforceable ceiling, not about unpriced calls
    being forbidden in themselves.
    """
    budget = RunBudget(max_calls=100)
    for _ in range(5):
        budget.settle(usage(1), None)
    assert budget.exhausted is False
    assert "5 unpriced" in budget.summary()


def test_priced_settlements_accumulate_against_the_usd_ceiling() -> None:
    budget = RunBudget(max_usd=1.0)
    for _ in range(4):
        assert budget.reserve(n_calls=1, estimated_tokens=0).granted
        budget.settle(usage(10), 0.3)
    assert budget.exhausted
    assert budget.snapshot().usd_settled == pytest.approx(1.2)


# -- the soft threshold --------------------------------------------------


def test_the_soft_threshold_warns_before_the_wall() -> None:
    budget = RunBudget(max_calls=10, soft_fraction=0.2)
    for _ in range(7):
        assert budget.reserve(n_calls=1, estimated_tokens=0).reason == ""
        budget.settle(usage(1), 0.0)
    warned = budget.reserve(n_calls=1, estimated_tokens=0)
    assert warned.granted  # still granted, and still says so
    assert "2 of 10 calls left" in warned.reason
    assert budget.near_limit
    assert budget.exhausted is False


def test_the_advisory_is_available_without_reserving() -> None:
    clock = FakeClock()
    budget = RunBudget(max_seconds=100.0, soft_fraction=0.5, clock=clock)
    assert budget.advisory() == ""
    clock.advance(60.0)
    assert "40.0s of 100s left" in budget.advisory()
    assert budget.remaining().granted is False


def test_soft_fraction_zero_disables_the_advisory() -> None:
    budget = RunBudget(max_calls=2, soft_fraction=0.0)
    assert budget.reserve(n_calls=1, estimated_tokens=0).reason == ""
    assert budget.advisory() == ""


# -- summary and Unbounded ----------------------------------------------


def test_summary_names_every_ceiling_including_the_unset_ones() -> None:
    line = RunBudget(max_usd=5.0, max_calls=100).summary()
    assert "\n" not in line
    assert "usd=$5.0000" in line
    assert "calls=100" in line
    # A summary listing only the ceilings that exist would let a run with one
    # loose bound read as fully bounded.
    assert "seconds=unset" in line
    assert "tokens=unset" in line


def test_unbounded_says_so_plainly() -> None:
    budget = Unbounded()
    budget.settle(usage(100), 0.25)
    line = budget.summary()
    assert line.startswith("UNBOUNDED")
    assert "no ceiling" in line
    assert budget.exhausted is False
    reservation = budget.reserve(n_calls=10_000, estimated_tokens=10**9)
    assert reservation.granted
    # None, not a large number: inventing a bound nobody set is the failure.
    assert reservation.calls_remaining is None
    assert reservation.usd_remaining is None


def test_unbounded_still_keeps_the_ledger() -> None:
    budget = Unbounded()
    budget.settle(usage(10), None)
    budget.settle(usage(10), 1.5)
    line = budget.summary()
    assert "$1.5000+1 unpriced" in line
    assert "2 calls" in line
    assert "20 tokens" in line


# -- concurrency: the point of the module -------------------------------


@pytest.mark.parametrize(("ceiling", "threads"), [(5, 64), (1, 32), (17, 100)])
def test_concurrent_reservations_never_overshoot_the_call_ceiling(
    ceiling: int, threads: int
) -> None:
    """Many threads, one small ceiling, exact count granted.

    A budget that checks and then increments without holding a lock passes
    every sequential test in this file and grants more than its ceiling here.
    That is the failure this module exists to prevent, so it is tested rather
    than asserted in a docstring.
    """
    budget = RunBudget(max_calls=ceiling)
    start = threading.Barrier(threads)
    results: list[CallReservation] = []
    results_lock = threading.Lock()

    def worker() -> None:
        start.wait()
        reservation = budget.reserve(n_calls=1, estimated_tokens=0)
        with results_lock:
            results.append(reservation)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    granted = [r for r in results if r.granted]
    assert len(results) == threads
    assert len(granted) == ceiling
    assert budget.snapshot().calls_used == ceiling
    assert budget.exhausted


def test_concurrent_batches_are_all_or_nothing() -> None:
    """Batches of four against a ceiling of ten: two land, the rest land zero.

    The ceiling is deliberately not a multiple of the batch size, so a budget
    that grants partial batches would leave two calls used out of the third
    batch and the assertion on calls_used would catch it.
    """
    ceiling = 10
    batch = 4
    budget = RunBudget(max_calls=ceiling)
    start = threading.Barrier(16)
    granted_batches = 0
    counter_lock = threading.Lock()

    def worker() -> None:
        nonlocal granted_batches
        start.wait()
        if budget.reserve(n_calls=batch, estimated_tokens=0).granted:
            with counter_lock:
                granted_batches += 1

    workers = [threading.Thread(target=worker) for _ in range(16)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    assert granted_batches == ceiling // batch
    assert budget.snapshot().calls_used == granted_batches * batch


def test_concurrent_reserve_and_settle_leave_the_ledger_exact() -> None:
    """Interleaved dispatch and reconciliation must not drift.

    Settlement releases a share of a pool that other threads are adding to at
    the same time, which is precisely where an unlocked implementation loses
    or invents tokens.
    """
    budget = RunBudget(max_calls=400, max_tokens=10**9)
    threads = 40
    per_thread = 10
    start = threading.Barrier(threads)

    def worker() -> None:
        start.wait()
        for _ in range(per_thread):
            reservation = budget.reserve(n_calls=1, estimated_tokens=100)
            assert reservation.granted
            budget.settle(usage(7), 0.001)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    snap = budget.snapshot()
    total = threads * per_thread
    assert snap.calls_settled == total
    assert snap.calls_in_flight == 0
    assert snap.tokens_pending == 0.0
    assert snap.usd_pending == 0.0
    assert snap.tokens_settled == total * 7
    assert snap.usd_settled == pytest.approx(total * 0.001)
    assert snap.unmatched_settlements == 0


def test_concurrent_token_ceiling_is_not_overshot_by_estimates() -> None:
    budget = RunBudget(max_tokens=1000)
    threads = 50
    start = threading.Barrier(threads)
    granted = 0
    lock = threading.Lock()

    def worker() -> None:
        nonlocal granted
        start.wait()
        if budget.reserve(n_calls=1, estimated_tokens=100).granted:
            with lock:
                granted += 1

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()

    assert granted == 10
    assert budget.snapshot().tokens_used == pytest.approx(1000.0)


# -- arithmetic invariants ----------------------------------------------

_ops = st.lists(
    st.tuples(
        st.integers(min_value=1, max_value=6),  # calls in the batch
        st.integers(min_value=0, max_value=500),  # estimated tokens
        st.integers(min_value=0, max_value=800),  # actual tokens per call
        st.floats(min_value=0.0, max_value=0.05, allow_nan=False),  # actual usd
    ),
    min_size=0,
    max_size=40,
)


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    ceiling=st.integers(min_value=1, max_value=50),
    ops=_ops,
)
def test_call_ceiling_is_never_overshot(
    ceiling: int, ops: list[tuple[int, int, int, float]]
) -> None:
    """The call ceiling is exact, because batch size is known before dispatch.

    Unlike tokens and cost, there is no estimate involved, so there is no
    excuse for an overshoot of even one call.
    """
    budget = RunBudget(max_calls=ceiling)
    landed = 0
    for n_calls, est, actual, cost in ops:
        if budget.reserve(n_calls=n_calls, estimated_tokens=est).granted:
            landed += n_calls
            for _ in range(n_calls):
                budget.settle(usage(actual), cost)
        assert budget.snapshot().calls_used <= ceiling
    assert landed <= ceiling
    assert budget.snapshot().calls_settled == landed


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(ops=_ops)
def test_reserve_then_settle_never_drifts(
    ops: list[tuple[int, int, int, float]],
) -> None:
    """Every granted call settles exactly once, and the pools return to zero.

    Drift here is the quiet failure: a pool that does not empty makes the
    budget look more spent than it is and strangles a run that had headroom,
    while one that empties early makes the ceiling porous.
    """
    budget = RunBudget(max_calls=10_000, max_tokens=10**9, max_usd=10**6)
    settled_tokens = 0
    settled_usd = 0.0
    settled_calls = 0
    for n_calls, est, actual, cost in ops:
        if not budget.reserve(n_calls=n_calls, estimated_tokens=est).granted:
            continue
        assert budget.snapshot().calls_in_flight == n_calls
        for _ in range(n_calls):
            budget.settle(usage(actual), cost)
            settled_calls += 1
            settled_tokens += actual
            settled_usd += cost
        snap = budget.snapshot()
        assert snap.calls_in_flight == 0
        assert snap.tokens_pending == 0.0
        assert snap.usd_pending == 0.0
        assert snap.calls_settled == settled_calls
        assert snap.tokens_settled == settled_tokens
        assert snap.usd_settled == pytest.approx(settled_usd)
        assert snap.unmatched_settlements == 0


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(
    ceiling=st.integers(min_value=1, max_value=5000),
    ops=_ops,
)
def test_settlements_never_exceed_a_granted_reservations_headroom(
    ceiling: int, ops: list[tuple[int, int, int, float]]
) -> None:
    """Settled usage stays inside the ceiling while estimates hold.

    Stated with the estimate as an upper bound on the actual, because that is
    the only condition under which a token ceiling can be a guarantee: the
    price of a call is known after it is made, so a caller who under-estimates
    by a factor of ten is buying an overshoot that no locking can prevent.
    What the budget does guarantee is that it refuses everything afterwards,
    which the overshoot test above covers.
    """
    budget = RunBudget(max_tokens=ceiling)
    for n_calls, est, actual, _cost in ops:
        assume(actual * n_calls <= est)
        if budget.reserve(n_calls=n_calls, estimated_tokens=est).granted:
            for _ in range(n_calls):
                budget.settle(usage(actual), None)
        snap = budget.snapshot()
        assert snap.tokens_used <= ceiling + 1e-9
        assert snap.tokens_settled <= ceiling


@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    ceiling=st.integers(min_value=1, max_value=20),
    batch=st.integers(min_value=1, max_value=25),
)
def test_a_refused_batch_never_moves_the_ledger(ceiling: int, batch: int) -> None:
    budget = RunBudget(max_calls=ceiling)
    before = budget.snapshot()
    reservation = budget.reserve(n_calls=batch, estimated_tokens=1)
    after = budget.snapshot()
    if reservation.granted:
        assert after.calls_used == batch
    else:
        assert after.calls_used == before.calls_used == 0
