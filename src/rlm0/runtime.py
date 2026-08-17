"""The loop that runs the control first and only then goes deeper.

One function drives every REPL in this file. The depth-zero attempt, the
depth-two attempt and the grandchild three levels down are all the same call to
`_drive`, differing in two arguments and nothing else. That is not tidiness. A
`Run` claims that its first attempt is the counterfactual for the ones after
it, and that claim is only true if the two went through the same prompt, the
same parser, the same environment and the same observation formatting. A
depth-zero path written separately, as a single-shot completion or a trimmed
loop, silently converts the run's headline measurement into a measurement of
the scaffolding. Several surveyed implementations have exactly that shape, with
a `use_repl` flag choosing between two unrelated code paths, and their reported
gains cannot be attributed.

The second structural commitment is that sub-call wiring is not conditional on
which sandbox was chosen. In one surveyed implementation recursion was
unreachable for every remote environment because the host-call bridge sat
inside a branch that only the local one entered, and nothing said so; runs
completed, numbers were published, and the recursion never happened. Here the
wiring is attempted for every attempt bounded above zero, and a sandbox that
cannot service host calls raises `RecursionUnavailableError` rather than
quietly producing a flat run wearing a depth label.

Prompting, parsing and observation formatting are taken as constructor
arguments typed by the Protocols in `ports.py`, which now names those seams.
The names are re-exported from here because they were declared in this module
first and callers import them from it.

The third commitment is that a fan-out is warm before it is wide. Anthropic's
documentation states that a cache entry becomes available only once the first
response has begun, and that parallel requests sharing a prefix do not read
each other's cache, so firing N children at a cold prefix writes that prefix N
times at 1.25x base input and costs more than never asking for caching at all.
The barrier that prevents it is therefore not a tuning step, it is part of what
makes the fan-out correct, and it is placed inside `_call_model` where every
call in the file already goes rather than offered as something a dispatcher can
remember to do.

The fourth is that every stop condition winds down. Budget exhaustion,
iteration exhaustion and the attempt deadline all end in the same salvage call,
because a run that has already spent real money should surrender the best
answer it can rather than throwing away what it bought. Truncating at the wall
is free to implement and is the reason a failed trajectory in the published
cost distributions costs the most and returns the least.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable

from rlm0.budget import FanOutEstimator
from rlm0.policy import Escalating
from rlm0.ports import (
    Budget,
    CallReservation,
    DepthPolicy,
    EscalationContext,
    LMClient,
    ObservationFormatter,
    ParsedTurn,
    Prompter,
    Sandbox,
    TurnParser,
)
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run

__all__ = [
    "RLM",
    "HostCallHandler",
    "ObservationFormatter",
    "ParsedTurn",
    "Prompter",
    "RecursionUnavailableError",
    "RefundableBudget",
    "SubCallSandbox",
    "TurnParser",
]


class RecursionUnavailableError(RuntimeError):
    """A deeper attempt was asked for, and the sandbox cannot service sub-calls.

    Loud on purpose. The alternative, running the attempt anyway with the
    sub-call name unbound, produces a run whose attempts are labelled depth one
    and depth two and whose behaviour is depth zero three times over. That
    failure has already been shipped by at least one open implementation and is
    invisible in its results.
    """


HostCallHandler = Callable[..., object]
"""Services one sub-call from inside the sandbox, out here where the key lives.

Returns `object` rather than `str` because a fan-out returns a list. The value
crosses the boundary as JSON, so a list of answers is as carryable as one
answer, and narrowing this to `str` would force the batch shape through a
string encoding for no reason other than an alias.
"""


@runtime_checkable
class SubCallSandbox(Protocol):
    """The older spelling of the half of the host-call seam that binds.

    `ports.Sandbox` now names `bind_host_call`, which is the seam this Protocol
    was standing in for, and the runtime prefers it. This one is kept and still
    accepted because a sandbox written against the older shape is otherwise
    silently unable to service a sub-call, and the failure it produces is
    exactly the one this project refuses to reproduce: a run labelled depth two
    that never recursed. Either hook satisfies the runtime; neither does not.
    """

    def set_host_call_handler(self, name: str, handler: HostCallHandler) -> None:
        """Bind the host-side implementation of a registered name."""
        ...


@runtime_checkable
class RefundableBudget(Protocol):
    """A budget that will take back a reservation nobody used.

    `ports.Budget` names `reserve` and `settle` and nothing between them, so a
    granted call that never happens has no way home and the in-flight pool only
    ever grows. This runtime reserves a wind-down call before every turn that
    might need one and reserves a whole batch before a fan-out, so it creates
    exactly that situation on purpose and several times per attempt.

    Detected structurally, in the same way and for the same reason as
    `SubCallSandbox`: the seam is real, the port does not name it yet, and a
    budget that cannot refund still works here, it is merely tighter than it
    says it is.
    """

    def release(self, *, n_calls: int) -> None:
        """Return calls that were granted and will not be made."""
        ...


class _WarmingBarrier:
    """One call goes first and the rest wait for it to land.

    The whole cost argument for a fan-out rests on the children sharing a
    cached prefix, and a cache entry does not exist until the response that
    writes it has begun. Releasing N children at once against a cold prefix
    therefore does not produce one write and N reads, it produces N writes at
    1.25x base input, which is more expensive than not asking for caching at
    all and shows up afterwards as a cache hit rate of zero that reads like a
    provider bug.

    Leadership is taken rather than assigned: whichever worker reaches its
    first model call first becomes the one that warms the prefix. Assigning it
    to index zero would look tidier and would idle every other worker behind a
    child that might be slow to start for reasons of its own.

    `open` is idempotent and is also called from the dispatcher's `finally`, so
    a leader that dies before its call returns releases its siblings instead of
    parking them until the timeout. The timeout is the second guard and exists
    because a barrier that can hang a run is worse than a barrier that
    occasionally fails to save money.
    """

    def __init__(self, *, timeout_s: float) -> None:
        self._timeout_s = max(timeout_s, 0.0)
        self._opened = threading.Event()
        self._lock = threading.Lock()
        self._leader_taken = False
        self.waits = 0
        """How many calls were actually held back. Read by tests, and by nobody
        else, because a barrier that silently stopped working would otherwise
        be invisible."""

    def take_lead(self) -> bool:
        """True for exactly one caller, and only while the prefix is cold."""
        with self._lock:
            if self._leader_taken or self._opened.is_set():
                return False
            self._leader_taken = True
            return True

    def wait(self) -> None:
        if self._opened.is_set():
            return
        with self._lock:
            self.waits += 1
        self._opened.wait(self._timeout_s)

    def open(self) -> None:
        self._opened.set()

    @property
    def is_open(self) -> bool:
        return self._opened.is_set()


@dataclass(frozen=True, slots=True)
class _Closed:
    """How one REPL, at one depth, stopped."""

    outcome: Outcome
    answer: str | None
    detail: str = ""


@dataclass(slots=True)
class _AttemptState:
    """Everything one attempt accumulates, across every depth inside it.

    Shared by the root REPL and every descendant, because an attempt is the
    unit the run compares and a sub-call that billed against this attempt
    belongs in this attempt's call list.

    Every mutation goes through a method holding the lock, because a fan-out
    writes here from several threads at once. `list.append` would survive that
    on this interpreter and `n += 1` would not, and a cost table that loses a
    call under concurrency is the accounting failure this project is about.
    """

    calls: list[CallRecord] = field(default_factory=list)
    iterations: int = 0
    exec_failures: int = 0
    sub_calls_refused: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record_call(self, call: CallRecord) -> None:
        with self.lock:
            self.calls.append(call)

    def note_iteration(self) -> None:
        with self.lock:
            self.iterations += 1

    def note_exec_failure(self) -> None:
        with self.lock:
            self.exec_failures += 1

    def note_sub_calls_refused(self, n: int = 1) -> None:
        with self.lock:
            self.sub_calls_refused += n

    def signals(self) -> dict[str, float]:
        """Cheap, content-free counters, computed here by the runtime.

        Every one of these is a count of something the attempt did. None of
        them look at the task text, the context text or the model's output, so
        a policy built on them cannot become a task classifier by accident.
        """
        with self.lock:
            root = sum(1 for call in self.calls if call.role is Role.ROOT)
            return {
                "root_calls": float(root),
                "sub_calls": float(len(self.calls) - root),
                "iterations": float(self.iterations),
                "exec_failures": float(self.exec_failures),
                "sub_calls_refused": float(self.sub_calls_refused),
            }


class _BatchMeter:
    """What one batch's dispatched calls actually cost, for the ratio.

    Only the first call of each child is counted, because that is the set the
    reservation was an estimate of. A child that goes on to take four more
    turns reserves those turns itself through the ordinary per-turn path, and
    folding them in here would make the over-reservation ratio a measurement of
    how talkative the children were rather than of how well the batch was
    sized.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tokens = 0
        self.calls = 0

    def record(self, tokens: int) -> None:
        with self._lock:
            self.tokens += tokens
            self.calls += 1


class _Dispatch:
    """One child's place in a warmed batch: the barrier, and the meter.

    Held per child rather than per batch so that the barrier applies to a
    child's first call and to nothing after it. A child's second turn is not
    part of the fan-out and has no reason to wait for anybody.
    """

    def __init__(self, barrier: _WarmingBarrier, meter: _BatchMeter) -> None:
        self._barrier = barrier
        self._meter = meter
        self._first = True

    def before_call(self) -> bool:
        """Hold this call behind the barrier. True if it is the one warming it."""
        if not self._first:
            return False
        if self._barrier.take_lead():
            return True
        self._barrier.wait()
        return False

    def after_call(self, *, leader: bool, tokens: int) -> None:
        if not self._first:
            return
        self._first = False
        self._meter.record(tokens)
        if leader:
            self._barrier.open()


class _HeldCall:
    """A reservation taken for a call that has not happened yet.

    The runtime reserves the wind-down call before the turn that might need it,
    which means that on every path where the wind-down does not happen there is
    a granted call nobody will ever make. Handing it back is not tidiness: the
    provisional pool only shrinks on `settle`, so an unreturned hold is
    permanent, and after enough turns the run is bounded by a ceiling lower
    than the one its own `summary()` prints into the run record.

    Both methods are idempotent so this can sit in a `finally` and still be
    correct on the paths that consumed the hold deliberately.
    """

    def __init__(self, release: Callable[[int], None]) -> None:
        self._release = release
        self._held = 0

    @property
    def held(self) -> int:
        return self._held

    def add(self, n: int) -> None:
        self._held += n

    def consume(self, n: int = 1) -> bool:
        """Spend a held call. False when there was nothing held to spend."""
        if self._held < n:
            return False
        self._held -= n
        return True

    def give_back(self) -> None:
        if self._held:
            n, self._held = self._held, 0
            self._release(n)


class RLM:
    """A recursive language model that attempts depth zero first, always.

    The escalation ladder is the policy's, the ceiling is the budget's, and
    this object owns only the loop and the accounting. It never inspects the
    task to decide anything.
    """

    def __init__(
        self,
        *,
        lm: LMClient,
        sandbox_factory: Callable[[], Sandbox],
        budget: Budget,
        prompter: Prompter,
        parser: TurnParser,
        observer: ObservationFormatter,
        model: str,
        policy: DepthPolicy | None = None,
        sub_model: str | None = None,
        max_iterations: int = 8,
        max_tokens: int = 4096,
        max_attempts: int = 4,
        exec_timeout_s: float = 30.0,
        attempt_timeout_s: float | None = None,
        context_variable: str = "CONTEXT",
        sub_call_name: str = "rlm_call",
        max_parallel_sub_calls: int = 4,
        warm_timeout_s: float = 120.0,
        estimator: FanOutEstimator | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if max_parallel_sub_calls < 1:
            raise ValueError("max_parallel_sub_calls must be at least 1")
        if warm_timeout_s <= 0.0:
            raise ValueError("warm_timeout_s must be positive")
        self._lm = lm
        self._sandbox_factory = sandbox_factory
        self._budget = budget
        self._prompter = prompter
        self._parser = parser
        self._observer = observer
        self._model = model
        self._policy: DepthPolicy = policy if policy is not None else Escalating()
        self._sub_model = sub_model or model
        self._max_iterations = max_iterations
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._exec_timeout_s = exec_timeout_s
        self._attempt_timeout_s = attempt_timeout_s
        self._context_variable = context_variable
        self._sub_call_name = sub_call_name
        self._max_parallel_sub_calls = max_parallel_sub_calls
        self._warm_timeout_s = warm_timeout_s
        self._estimator = estimator if estimator is not None else FanOutEstimator()
        self._clock = clock

    @property
    def estimator(self) -> FanOutEstimator:
        """The fan-out estimator, so the ratio it achieved can be read out.

        Exposed rather than kept private because the claim this runtime makes
        about over-reservation is only worth making if somebody can check it
        after a real run.
        """
        return self._estimator

    def complete(self, task: str, context: str = "") -> Run:
        """Answer one task, cheapest first, and return the whole trajectory.

        The depth-zero attempt is not optional here and there is no argument
        that skips it. A caller who genuinely cannot run the control builds the
        `Run` with a `BaselineWaiver` itself; this entry point has no way to
        produce one, so the expensive, attributed refusal stays the only way to
        lose the comparison.
        """
        attempts: list[Attempt] = []
        depth: int | None = 0
        while depth is not None and len(attempts) < self._max_attempts:
            attempt, signals = self._run_attempt(
                task=task, context=context, max_depth=depth
            )
            attempts.append(attempt)
            depth = self._next_depth(
                task=task,
                context=context,
                attempts=attempts,
                last=attempt,
                signals=signals,
            )
        return Run(
            task=task,
            attempts=tuple(attempts),
            budget_summary=self._budget.summary(),
            labels={
                "policy": self._policy.describe(),
                "fan_out": self._estimator.describe(),
            },
        )

    def _next_depth(
        self,
        *,
        task: str,
        context: str,
        attempts: Sequence[Attempt],
        last: Attempt,
        signals: dict[str, float],
    ) -> int | None:
        """Ask the policy, then refuse anything that would break the ordering.

        A policy that returns a bound at or below the one just run would make
        the run's attempts non-monotonic, which `Run` refuses at construction
        and which would mean the first attempt was no longer the cheapest. It
        is treated as "stop" rather than as an error, because the policy seam
        exists to be replaced by strangers.
        """
        spent = _total_cost(attempts)
        elapsed = sum(a.wall_clock_s for a in attempts)
        proposal = self._policy.next_depth(
            EscalationContext(
                task=task,
                context_chars=len(context),
                attempts_so_far=len(attempts),
                last_outcome=last.outcome.value,
                last_answer=last.answer,
                spent_usd=spent,
                elapsed_s=elapsed,
                budget_reservation=self._probe_budget(),
                signals=dict(signals),
                last_max_depth=last.max_depth,
            )
        )
        if proposal is None or proposal <= last.max_depth:
            return None
        return proposal

    def _probe_budget(self) -> CallReservation:
        """Ask the budget what is left without spending any of it.

        `Budget.remaining` names this read. It used to be a reservation for
        zero calls, which worked only for as long as every implementation
        agreed that zero was free, and both implementations in this package
        refuse a reservation below one call outright.
        """
        remaining = getattr(self._budget, "remaining", None)
        if callable(remaining):
            return cast(CallReservation, remaining())
        # Compatibility with pre-existing Budget implementations. The port
        # now requires remaining(), but a zero-call probe was the only prior
        # convention and keeping it here lets older adapters fail closed on
        # their own terms rather than crashing after a completed control run.
        return self._budget.reserve(n_calls=0, estimated_tokens=0)

    def _release(self, n_calls: int) -> None:
        """Hand back calls that were granted and will not be made.

        Silently a no-op for a budget that names no refund. That is the right
        failure: such a budget is tighter than it advertises, which is the safe
        direction, and refusing to run against it would exclude every third
        party implementation of a port that does not require this yet.
        """
        release = getattr(self._budget, "release", None)
        if n_calls > 0 and callable(release):
            release(n_calls=n_calls)

    # -- one attempt ----------------------------------------------------

    def _run_attempt(
        self, *, task: str, context: str, max_depth: int
    ) -> tuple[Attempt, dict[str, float]]:
        """One bounded try, from a fresh environment to a closed `Outcome`."""
        started = self._clock()
        state = _AttemptState()
        deadline = (
            None
            if self._attempt_timeout_s is None
            else started + self._attempt_timeout_s
        )
        sandbox = self._sandbox_factory()
        try:
            closed = self._drive(
                sandbox=sandbox,
                task=task,
                payload=context,
                depth=0,
                max_depth=max_depth,
                deadline=deadline,
                state=state,
            )
        finally:
            sandbox.close()
        attempt = Attempt(
            max_depth=max_depth,
            outcome=closed.outcome,
            calls=tuple(state.calls),
            wall_clock_s=self._clock() - started,
            answer=closed.answer,
            detail=closed.detail,
        )
        return attempt, state.signals()

    # -- one REPL, at one depth -----------------------------------------

    def _drive(
        self,
        *,
        sandbox: Sandbox,
        task: str,
        payload: str,
        depth: int,
        max_depth: int,
        deadline: float | None,
        state: _AttemptState,
        dispatch: _Dispatch | None = None,
    ) -> _Closed:
        """Run one REPL to a stop. The only function in this file that does.

        Root and child, depth zero and depth two, all arrive here. `depth` and
        `max_depth` change what is bound inside the sandbox and what the system
        prompt is told exists; they change nothing else, which is what makes
        the depth-zero attempt a control rather than a different experiment.
        """
        sub_calls_available = depth < max_depth
        if sub_calls_available:
            self._wire_sub_calls(
                sandbox=sandbox,
                depth=depth,
                max_depth=max_depth,
                deadline=deadline,
                state=state,
            )
        # The payload goes into the environment, never into the window. That is
        # the whole mechanism at depth zero and the whole point of a handle at
        # depth one.
        sandbox.set_variable(self._context_variable, payload)
        system = self._prompter.system(
            max_depth=max_depth,
            sub_call_name=self._sub_call_name if sub_calls_available else None,
        )
        messages: list[dict[str, str]] = [
            {
                "role": "user",
                "content": self._prompter.opening(
                    task=task,
                    context_variable=self._context_variable,
                    context_chars=len(payload),
                ),
            }
        ]
        try:
            return self._turns(
                sandbox=sandbox,
                system=system,
                messages=messages,
                depth=depth,
                deadline=deadline,
                state=state,
            )
        except RecursionUnavailableError:
            raise
        except Exception as exc:
            # An error closes this attempt rather than the run. The policy is
            # then told the environment failed, which is a different fact from
            # the task being hard and is why `Outcome` keeps them apart.
            return _Closed(Outcome.ERRORED, None, f"{type(exc).__name__}: {exc}")

    def _turns(
        self,
        *,
        sandbox: Sandbox,
        system: str,
        messages: list[dict[str, str]],
        depth: int,
        deadline: float | None,
        state: _AttemptState,
    ) -> _Closed:
        for _ in range(self._max_iterations):
            if deadline is not None and self._clock() >= deadline:
                return _Closed(
                    Outcome.TIMED_OUT, None, "deadline passed before the next call"
                )
            # Two calls are reserved for one turn: this one, and the wind-down
            # this turn may turn out to need. A ceiling that grants the last
            # call to a working turn leaves nothing to ask for a final answer
            # with, which is how budget exhaustion becomes an empty result
            # instead of a shorter one.
            reservation = self._budget.reserve(
                n_calls=2, estimated_tokens=_estimate_tokens(system, messages)
            )
            if not reservation.granted:
                return self._wind_down(
                    system=system,
                    messages=messages,
                    depth=depth,
                    refusal=reservation,
                    state=state,
                )
            state.iterations += 1
            text = self._call_model(
                system=system,
                messages=messages,
                depth=depth,
                state=state,
                release_after=1,
            )
            parsed = self._parser.parse(text)
            answer = parsed.final_answer
            if answer is not None:
                return _Closed(Outcome.ANSWERED, answer)
            messages.append({"role": "assistant", "content": text})
            code = parsed.code
            if code is None:
                messages.append(
                    {"role": "user", "content": self._observer.format_no_action(text)}
                )
                continue
            result = sandbox.execute(code, timeout_s=self._exec_timeout_s)
            if not result.ok:
                state.exec_failures += 1
            messages.append({"role": "user", "content": self._observer.format(result)})
        return _Closed(
            Outcome.ITERATIONS_EXHAUSTED,
            None,
            f"stopped after {self._max_iterations} iterations",
        )

    def _wind_down(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        depth: int,
        refusal: CallReservation,
        state: _AttemptState,
    ) -> _Closed:
        """Spend the held-back call telling the model to stop and summarise.

        Running out of budget is a decision the runtime made and not an
        exception the model caused, so it gets told what happened and asked for
        whatever it has. What comes back is recorded in `detail` and never as
        the attempt's answer: `Attempt` refuses an answer on an outcome that is
        not ANSWERED, and it is right to, because a conclusion reached under a
        truncated budget did not survive its own stop condition. Keeping it as
        detail means a human can still read it without a scorer counting it.
        """
        final = self._budget.reserve(n_calls=1, estimated_tokens=256)
        if not final.granted:
            return _Closed(
                Outcome.BUDGET_EXHAUSTED,
                None,
                f"budget refused even the wind-down call: {final.reason}",
            )
        messages.append(
            {
                "role": "user",
                "content": self._prompter.wind_down(
                    reason=refusal.reason or "the call budget for this run is spent"
                ),
            }
        )
        text = self._call_model(
            system=system, messages=messages, depth=depth, state=state
        )
        parsed = self._parser.parse(text)
        partial = parsed.final_answer if parsed.final_answer is not None else text
        return _Closed(
            Outcome.BUDGET_EXHAUSTED,
            None,
            f"budget exhausted; partial reply not counted as an answer: {partial}",
        )

    def _call_model(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        depth: int,
        state: _AttemptState,
        release_after: int = 0,
    ) -> str:
        """One completion, settled against the budget and attributed to a depth.

        Every call in the file goes through here, so there is no path that can
        bill a run without appearing in its cost table.
        """
        role = Role.ROOT if depth == 0 else Role.SUB
        response = self._lm.complete(
            system=system,
            messages=messages,
            model=self._model if depth == 0 else self._sub_model,
            max_tokens=self._max_tokens,
            cache_prefix=True,
        )
        self._budget.settle(response.usage, response.cost_usd)
        self._release(release_after)
        state.calls.append(
            CallRecord(
                role=role,
                depth=depth,
                model=response.model,
                usage=response.usage,
                wall_clock_s=response.wall_clock_s,
                cost_usd=response.cost_usd,
                cached_prefix=response.cached_prefix,
            )
        )
        return response.text

    # -- sub-calls ------------------------------------------------------

    def _wire_sub_calls(
        self,
        *,
        sandbox: Sandbox,
        depth: int,
        max_depth: int,
        deadline: float | None,
        state: _AttemptState,
    ) -> None:
        """Bridge the sub-call out to the host, or refuse to run at all.

        Unconditional on which sandbox this is. The runtime does not know and
        must not learn: the moment recursion depends on the environment choice,
        somebody adds an environment and recursion stops happening without a
        single test failing.
        """
        sandbox.register_host_call(self._sub_call_name, 2)
        if not isinstance(sandbox, SubCallSandbox):
            raise RecursionUnavailableError(
                f"attempt bounded at depth {max_depth} needs sub-calls, but "
                f"{type(sandbox).__name__} exposes no way to service "
                f"{self._sub_call_name!r} on the host. Refusing to run a flat "
                "attempt under a depth label."
            )

        def handler(*args: str) -> str:
            return self._service_sub_call(
                parent=sandbox,
                args=args,
                depth=depth,
                max_depth=max_depth,
                deadline=deadline,
                state=state,
            )

        sandbox.set_host_call_handler(self._sub_call_name, handler)

    def _service_sub_call(
        self,
        *,
        parent: Sandbox,
        args: Sequence[str],
        depth: int,
        max_depth: int,
        deadline: float | None,
        state: _AttemptState,
    ) -> str:
        """Give the child its own REPL over a slice it was handed by name.

        The second argument is the name of a variable in the parent's
        environment, not its contents. A surveyed implementation passes the
        child a single string that is both its query and its entire context,
        which reproduces at depth one the exact problem the REPL exists to
        solve and makes depth two pointless. Here the slice is read out of the
        parent environment and bound into the child's, so it never crosses a
        prompt.
        """
        child_depth = depth + 1
        if child_depth > max_depth:
            state.sub_calls_refused += 1
            return f"<sub-call refused: depth bound {max_depth} reached>"
        query = args[0] if args else ""
        handle = args[1] if len(args) > 1 else ""
        payload = "" if not handle else (parent.get_variable(handle) or "")
        child = self._sandbox_factory()
        try:
            closed = self._drive(
                sandbox=child,
                task=query,
                payload=payload,
                depth=child_depth,
                max_depth=max_depth,
                deadline=deadline,
                state=state,
            )
        finally:
            child.close()
        if closed.answer is None:
            return f"<sub-call did not answer: {closed.outcome.value}>"
        return closed.answer


def _estimate_tokens(system: str, messages: Sequence[dict[str, str]]) -> int:
    """A rough size for the reservation, and only for the reservation.

    Characters over four is a bad token count and a fine reservation hint. It
    is never recorded as usage: `TokenUsage` on a `CallRecord` always comes
    from what the provider reported, because a cost table built on estimates is
    wrong in a way nobody notices until the invoice.
    """
    chars = len(system) + sum(len(m.get("content", "")) for m in messages)
    return chars // 4


def _total_cost(attempts: Sequence[Attempt]) -> float | None:
    """None when anything was unpriced, never a partial sum."""
    if any(attempt.cost_usd is None for attempt in attempts):
        return None
    return sum(attempt.cost_usd or 0.0 for attempt in attempts)
