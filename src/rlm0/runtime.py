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
arguments typed by the Protocols below rather than imported. `ports.py` does
not name these seams yet, and inventing an import to a concrete module would
put the orchestrator back in the business of knowing who renders its prompts.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rlm0.policy import Escalating
from rlm0.ports import (
    Budget,
    CallReservation,
    DepthPolicy,
    EscalationContext,
    ExecResult,
    LMClient,
    Sandbox,
)
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run

__all__ = [
    "RLM",
    "HostCallHandler",
    "ObservationFormatter",
    "ParsedTurn",
    "Prompter",
    "RecursionUnavailableError",
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


HostCallHandler = Callable[..., str]
"""Services one sub-call from inside the sandbox, out here where the key lives."""


@runtime_checkable
class SubCallSandbox(Protocol):
    """The half of the host-call seam that `ports.Sandbox` does not yet name.

    `Sandbox.register_host_call` declares that a name exists inside and says
    nothing about who answers when it is called. Until that seam is stated in
    `ports.py`, the runtime detects the handler hook structurally and refuses
    to run a deep attempt without it, which is the behaviour that keeps a
    missing bridge from passing as a completed recursive run.
    """

    def set_host_call_handler(self, name: str, handler: HostCallHandler) -> None:
        """Bind the host-side implementation of a registered name."""
        ...


class ParsedTurn(Protocol):
    """What one model turn asked for: code to run, an answer, or neither."""

    @property
    def code(self) -> str | None: ...

    @property
    def final_answer(self) -> str | None: ...


class TurnParser(Protocol):
    """Turns raw completion text into an action, identically at every depth."""

    def parse(self, text: str) -> ParsedTurn: ...


class Prompter(Protocol):
    """Renders the system and opening messages.

    `sub_call_name` is None exactly when sub-calls are unavailable, and it is
    the only depth-dependent input the prompter gets. Handing it the depth as
    well would let a prompt fork on it, which reintroduces the two-code-paths
    problem one layer down.
    """

    def system(self, *, max_depth: int, sub_call_name: str | None) -> str: ...

    def opening(
        self, *, task: str, context_variable: str, context_chars: int
    ) -> str: ...

    def wind_down(self, *, reason: str) -> str: ...


class ObservationFormatter(Protocol):
    """Renders execution results back into the conversation."""

    def format(self, result: ExecResult) -> str: ...

    def format_no_action(self, text: str) -> str: ...


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
    """

    calls: list[CallRecord] = field(default_factory=list)
    iterations: int = 0
    exec_failures: int = 0
    sub_calls_refused: int = 0

    def signals(self) -> dict[str, float]:
        """Cheap, content-free counters, computed here by the runtime.

        Every one of these is a count of something the attempt did. None of
        them look at the task text, the context text or the model's output, so
        a policy built on them cannot become a task classifier by accident.
        """
        root = sum(1 for call in self.calls if call.role is Role.ROOT)
        return {
            "root_calls": float(root),
            "sub_calls": float(len(self.calls) - root),
            "iterations": float(self.iterations),
            "exec_failures": float(self.exec_failures),
            "sub_calls_refused": float(self.sub_calls_refused),
        }


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least 1")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
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
        self._clock = clock

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
            labels={"policy": self._policy.describe()},
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
            )
        )
        if proposal is None or proposal <= last.max_depth:
            return None
        return proposal

    def _probe_budget(self) -> CallReservation:
        """Ask the budget what is left without spending any of it.

        `Budget` has no read-only accessor, so a reservation for zero calls is
        used as the probe. Reserving one call instead would either leak an
        allowance when the policy declines to escalate, or bound the policy's
        view to a single call when what it needs to know is whether a whole
        attempt fits.
        """
        return self._budget.reserve(n_calls=0, estimated_tokens=0)

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
                system=system, messages=messages, depth=depth, state=state
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
