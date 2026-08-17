"""Tests for the loop, run entirely against fakes.

The load-bearing tests here are the ones about sameness. It is easy to write a
runtime that produces a depth-zero attempt and a depth-one attempt, and hard to
show that the two differ only in whether sub-calls existed. It is exactly that
difference the `Run` is later read as having measured, so several of these
tests assert on what the two attempts had in common rather than on what either
of them returned.

The fake model is keyed on the task it was handed and on whether its system
prompt told it a sub-call exists, which is the only information a real model
has about its own depth.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from rlm0.policy import Escalating, Fixed, Never
from rlm0.ports import (
    CallReservation,
    DepthPolicy,
    EscalationContext,
    ExecResult,
    LMResponse,
)
from rlm0.run import Outcome, Role, TokenUsage, Verdict
from rlm0.runtime import RLM, HostCallHandler, RecursionUnavailableError

Responder = Callable[[str, list[dict[str, str]]], str]

# -- fakes ---------------------------------------------------------------


@dataclass
class FakeLM:
    """Scripted completions carrying provider-reported usage, as the port asks."""

    responder: Responder
    calls: list[tuple[str, list[dict[str, str]], str]] = field(default_factory=list)

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        cache_prefix: bool = False,
    ) -> LMResponse:
        self.calls.append((system, [dict(m) for m in messages], model))
        return LMResponse(
            text=self.responder(system, messages),
            usage=TokenUsage(input_tokens=100, output_tokens=20),
            model=model,
            wall_clock_s=0.1,
            cost_usd=0.001,
            cached_prefix=cache_prefix,
        )

    @property
    def prompts(self) -> list[str]:
        """Every string that ever crossed into a model window."""
        out: list[str] = []
        for system, messages, _ in self.calls:
            out.append(system)
            out.extend(m["content"] for m in messages)
        return out


@dataclass
class FakeSandbox:
    """A REPL that understands one instruction and one host call.

    `SUB:<query>|<handle>` is how the fake model asks for a sub-call. When
    nothing bound the name it fails with a name error, which is what a real
    interpreter does inside a depth-zero attempt.
    """

    variables: dict[str, str] = field(default_factory=dict)
    executed: list[str] = field(default_factory=list)
    registered: list[tuple[str, int]] = field(default_factory=list)
    handlers: dict[str, HostCallHandler] = field(default_factory=dict)
    closed: int = 0

    def execute(self, code: str, *, timeout_s: float) -> ExecResult:
        self.executed.append(code)
        if code.startswith("SUB:"):
            query, _, handle = code[4:].partition("|")
            handler = self.handlers.get("rlm_call")
            if handler is None:
                return ExecResult(
                    stdout="",
                    stderr="NameError: name 'rlm_call' is not defined",
                    wall_clock_s=0.01,
                    ok=False,
                    variables=tuple(self.variables),
                )
            return ExecResult(
                stdout=handler(query, handle),
                stderr="",
                wall_clock_s=0.01,
                ok=True,
                variables=tuple(self.variables),
            )
        return ExecResult(
            stdout=f"ran {code}",
            stderr="",
            wall_clock_s=0.01,
            ok=True,
            variables=tuple(self.variables),
        )

    def set_variable(self, name: str, value: str) -> None:
        self.variables[name] = value

    def get_variable(self, name: str) -> str | None:
        return self.variables.get(name)

    def register_host_call(self, name: str, arity: int) -> None:
        self.registered.append((name, arity))

    def set_host_call_handler(self, name: str, handler: HostCallHandler) -> None:
        self.handlers[name] = handler

    def close(self) -> None:
        self.closed += 1


class ExplodingSandbox(FakeSandbox):
    """An environment that dies mid-attempt."""

    def execute(self, code: str, *, timeout_s: float) -> ExecResult:
        raise OSError("the container went away")


@dataclass
class DeafSandbox:
    """Registers a host call and has no way to service one.

    The exact shape of the bug this project refuses to reproduce: everything
    looks wired, and the sub-call never reaches a host.
    """

    variables: dict[str, str] = field(default_factory=dict)
    registered: list[tuple[str, int]] = field(default_factory=list)

    def execute(self, code: str, *, timeout_s: float) -> ExecResult:
        return ExecResult(stdout="", stderr="", wall_clock_s=0.0, ok=True)

    def set_variable(self, name: str, value: str) -> None:
        self.variables[name] = value

    def get_variable(self, name: str) -> str | None:
        return self.variables.get(name)

    def register_host_call(self, name: str, arity: int) -> None:
        self.registered.append((name, arity))

    def close(self) -> None:
        return None


@dataclass
class FakeBudget:
    """A call ceiling that reserves before dispatch and settles after."""

    calls_allowed: int = 20
    used: int = 0
    settled: int = 0
    reservations: list[int] = field(default_factory=list)

    def reserve(self, *, n_calls: int, estimated_tokens: int) -> CallReservation:
        self.reservations.append(n_calls)
        remaining = self.calls_allowed - self.used
        if n_calls > remaining:
            return CallReservation(
                granted=False,
                reason=f"{remaining} calls left, {n_calls} asked for",
                calls_remaining=remaining,
            )
        return CallReservation(granted=True, calls_remaining=remaining)

    def settle(self, usage: TokenUsage, cost_usd: float | None) -> None:
        self.used += 1
        self.settled += 1

    def summary(self) -> str:
        return f"max {self.calls_allowed} calls"

    @property
    def exhausted(self) -> bool:
        return self.used >= self.calls_allowed


@dataclass
class FakePrompter:
    """Records exactly what it was asked to render, and under which bound."""

    systems: list[tuple[int, str | None]] = field(default_factory=list)
    openings: list[tuple[str, str, int]] = field(default_factory=list)
    wind_downs: list[str] = field(default_factory=list)

    def system(self, *, max_depth: int, sub_call_name: str | None) -> str:
        self.systems.append((max_depth, sub_call_name))
        return f"SYSTEM sub={sub_call_name}"

    def opening(self, *, task: str, context_variable: str, context_chars: int) -> str:
        self.openings.append((task, context_variable, context_chars))
        return f"TASK: {task} | VAR: {context_variable} | CHARS: {context_chars}"

    def wind_down(self, *, reason: str) -> str:
        self.wind_downs.append(reason)
        return f"WIND DOWN: {reason}"


@dataclass(frozen=True)
class FakeParsed:
    code: str | None = None
    final_answer: str | None = None


@dataclass
class FakeParser:
    seen: list[str] = field(default_factory=list)

    def parse(self, text: str) -> FakeParsed:
        self.seen.append(text)
        if text.startswith("FINAL:"):
            return FakeParsed(final_answer=text[len("FINAL:") :].strip())
        if text.startswith("CODE:"):
            return FakeParsed(code=text[len("CODE:") :].strip())
        return FakeParsed()


@dataclass
class FakeObserver:
    formatted: list[ExecResult] = field(default_factory=list)
    nudged: list[str] = field(default_factory=list)

    def format(self, result: ExecResult) -> str:
        self.formatted.append(result)
        return f"OBSERVATION ok={result.ok} out={result.stdout} err={result.stderr}"

    def format_no_action(self, text: str) -> str:
        self.nudged.append(text)
        return "OBSERVATION: no code and no answer"


Script = dict[str, list[str]]


def scripted(flat: Script, deep: Script | None = None) -> Responder:
    """Reply from a table keyed by the task, the turn, and sub-call availability.

    Two tables rather than one because a model that has been told a tool exists
    genuinely behaves differently, and because the depth-zero and depth-one
    root conversations are otherwise indistinguishable, which is itself the
    property under test.
    """
    deep_table = deep if deep is not None else {}

    def pick(table: Script, opening: str, turn: int) -> str | None:
        for key, replies in table.items():
            if key in opening and replies:
                return replies[min(turn, len(replies) - 1)]
        return None

    def responder(system: str, messages: list[dict[str, str]]) -> str:
        opening = messages[0]["content"]
        turn = sum(1 for m in messages if m["role"] == "assistant")
        first, second = (
            (deep_table, flat) if "sub=rlm_call" in system else (flat, deep_table)
        )
        reply = pick(first, opening, turn)
        if reply is None:
            reply = pick(second, opening, turn)
        return reply if reply is not None else "CODE: NOOP"

    return responder


@dataclass
class Harness:
    """One assembled runtime plus every fake it was built from."""

    rlm: RLM
    lm: FakeLM
    budget: FakeBudget
    prompter: FakePrompter
    parser: FakeParser
    observer: FakeObserver
    sandboxes: list[FakeSandbox]


def build(
    responder: Responder,
    *,
    policy: DepthPolicy | None = None,
    calls_allowed: int = 20,
    max_iterations: int = 2,
    make_box: Callable[[], FakeSandbox] = FakeSandbox,
) -> Harness:
    lm = FakeLM(responder=responder)
    budget = FakeBudget(calls_allowed=calls_allowed)
    prompter = FakePrompter()
    parser = FakeParser()
    observer = FakeObserver()
    sandboxes: list[FakeSandbox] = []

    def factory() -> FakeSandbox:
        box = make_box()
        sandboxes.append(box)
        return box

    rlm = RLM(
        lm=lm,
        sandbox_factory=factory,
        budget=budget,
        prompter=prompter,
        parser=parser,
        observer=observer,
        model="fake-model",
        policy=policy if policy is not None else Escalating(),
        max_iterations=max_iterations,
    )
    return Harness(rlm, lm, budget, prompter, parser, observer, sandboxes)


# -- the control path ----------------------------------------------------


def test_depth_zero_answer_is_a_single_attempt_that_never_recurses() -> None:
    harness = build(scripted({"ROOT": ["CODE: NOOP", "FINAL: 1994"]}))
    run = harness.rlm.complete("ROOT question", context="a" * 100)

    assert len(run.attempts) == 1
    assert run.attempts[0].max_depth == 0
    assert run.attempts[0].outcome is Outcome.ANSWERED
    assert run.answer == "1994"
    assert run.recursion_verdict().verdict is Verdict.NOT_ATTEMPTED
    assert run.attempts[0].n_sub_calls == 0
    assert all(call.role is Role.ROOT for call in run.calls)
    # The sub-call machinery was never even offered to the environment.
    assert harness.sandboxes[0].registered == []
    assert harness.sandboxes[0].handlers == {}


def test_the_context_never_enters_the_window() -> None:
    secret = "PAYLOAD-" + "x" * 5000
    harness = build(scripted({"ROOT": ["FINAL: done"]}))
    harness.rlm.complete("ROOT question", context=secret)

    assert harness.sandboxes[0].variables["CONTEXT"] == secret
    assert not any(secret in prompt for prompt in harness.lm.prompts)


def test_a_run_records_the_budget_and_the_policy_it_ran_under() -> None:
    harness = build(scripted({"ROOT": ["FINAL: done"]}), policy=Never())
    run = harness.rlm.complete("ROOT question")

    assert run.budget_summary == "max 20 calls"
    assert run.labels["policy"] == Never().describe()


def test_never_stops_after_the_control_even_when_it_failed() -> None:
    harness = build(scripted({"ROOT": ["CODE: NOOP"]}), policy=Never())
    run = harness.rlm.complete("ROOT question")

    assert len(run.attempts) == 1
    assert run.attempts[0].outcome is Outcome.ITERATIONS_EXHAUSTED
    assert run.answer is None


# -- escalation ----------------------------------------------------------


def escalating_harness(calls_allowed: int = 20) -> Harness:
    """Depth zero stalls forever; depth one delegates and answers."""
    return build(
        scripted(
            {"ROOT": ["CODE: NOOP"]},
            {
                "ROOT": ["CODE: SUB:count the mentions|CONTEXT", "FINAL: seventeen"],
                "count the mentions": ["FINAL: 17"],
            },
        ),
        calls_allowed=calls_allowed,
    )


def test_failure_at_depth_zero_escalates_and_recursion_is_credited() -> None:
    harness = escalating_harness()
    run = harness.rlm.complete("ROOT question", context="corpus")

    assert [a.max_depth for a in run.attempts] == [0, 1]
    assert run.attempts[0].outcome is Outcome.ITERATIONS_EXHAUSTED
    assert run.attempts[0].answer is None
    assert run.attempts[1].outcome is Outcome.ANSWERED
    assert run.answer == "seventeen"

    verdict = run.recursion_verdict()
    assert verdict.verdict is Verdict.HELPED
    assert verdict.extra_sub_calls == 1
    assert verdict.extra_cost_usd is not None

    sub_calls = [c for c in run.calls if c.role is Role.SUB]
    assert [c.depth for c in sub_calls] == [1]


def test_both_attempts_went_through_the_same_prompt_and_environment() -> None:
    harness = escalating_harness()
    run = harness.rlm.complete("ROOT question", context="corpus")

    # Same opening, from the same prompter, for both bounds.
    root_openings = [o for o in harness.prompter.openings if o[0] == "ROOT question"]
    assert len(root_openings) == 2
    assert root_openings[0] == root_openings[1]

    # The system prompts differ in the one permitted way and in nothing else.
    assert harness.prompter.systems[0] == (0, None)
    assert harness.prompter.systems[1] == (1, "rlm_call")

    # Both attempts really drove a REPL, through the same execute path, the
    # same parser and the same observation formatting.
    for box in (harness.sandboxes[0], harness.sandboxes[1]):
        assert box.variables["CONTEXT"] == "corpus"
        assert box.executed
        assert box.closed == 1
    assert harness.observer.formatted
    assert len(harness.parser.seen) == len(harness.lm.calls)
    assert all(attempt.calls for attempt in run.attempts)


def test_a_depth_zero_attempt_cannot_make_a_sub_call() -> None:
    # The model tries to recurse at depth zero. The name is unbound, so the
    # environment answers with a name error and the control stays a control.
    harness = build(
        scripted(
            {
                "ROOT": ["CODE: SUB:go deeper|CONTEXT", "CODE: NOOP"],
                "go deeper": ["FINAL: should never happen"],
            }
        ),
        policy=Never(),
    )
    run = harness.rlm.complete("ROOT question", context="corpus")

    assert run.attempts[0].n_sub_calls == 0
    assert len(harness.sandboxes) == 1
    assert harness.sandboxes[0].registered == []
    assert "NameError" in harness.observer.formatted[0].stderr
    assert all(c.role is Role.ROOT and c.depth == 0 for c in run.calls)


# -- recursion, bounded --------------------------------------------------


def test_a_child_gets_its_own_repl_over_a_handle_not_a_prompt() -> None:
    shard = "SLICE-" + "y" * 4000

    def make_box() -> FakeSandbox:
        return FakeSandbox(variables={"SHARD": shard})

    harness = build(
        scripted(
            {"ROOT": ["FINAL: shallow"], "count them": ["FINAL: 12"]},
            {"ROOT": ["CODE: SUB:count them|SHARD", "FINAL: counted"]},
        ),
        policy=Fixed(1),
        make_box=make_box,
    )
    run = harness.rlm.complete("ROOT question", context="corpus")

    child = harness.sandboxes[-1]
    assert child.variables["CONTEXT"] == shard
    assert not any(shard in prompt for prompt in harness.lm.prompts)
    assert run.attempts[1].n_sub_calls == 1
    assert run.attempts[1].answer == "counted"


def test_recursion_is_bounded_by_the_attempts_own_depth_limit() -> None:
    harness = build(
        scripted(
            {"ROOT": ["FINAL: shallow"]},
            {
                "ROOT": ["CODE: SUB:child work|CONTEXT", "FINAL: done"],
                "child work": ["CODE: SUB:grandchild work|CONTEXT", "FINAL: child"],
                "grandchild work": ["FINAL: grandchild"],
            },
        ),
        policy=Fixed(1),
    )
    run = harness.rlm.complete("ROOT question", context="corpus")

    assert sorted({c.depth for c in run.attempts[1].calls}) == [0, 1]
    # The child asked to go deeper and found nothing bound, because the bound
    # is what decides the wiring.
    assert any("NameError" in obs.stderr for obs in harness.observer.formatted)


def test_recursion_reaches_depth_two_when_the_bound_allows_it() -> None:
    harness = build(
        scripted(
            {"ROOT": ["FINAL: shallow"], "grandchild work": ["FINAL: grandchild"]},
            {
                "ROOT": ["CODE: SUB:child work|CONTEXT", "FINAL: done"],
                "child work": ["CODE: SUB:grandchild work|CONTEXT", "FINAL: child"],
            },
        ),
        policy=Fixed(2),
    )
    run = harness.rlm.complete("ROOT question", context="corpus")

    deep = run.attempts[1]
    assert deep.max_depth == 2
    assert sorted({c.depth for c in deep.calls}) == [0, 1, 2]
    # Two turns from the child plus the one turn the grandchild needed.
    assert deep.n_sub_calls == 3
    assert all(c.role is Role.SUB for c in deep.calls if c.depth > 0)
    assert deep.answer == "done"


def test_a_sandbox_that_cannot_service_sub_calls_refuses_loudly() -> None:
    rlm = RLM(
        lm=FakeLM(responder=scripted({"ROOT": ["CODE: NOOP"]})),
        sandbox_factory=DeafSandbox,
        budget=FakeBudget(),
        prompter=FakePrompter(),
        parser=FakeParser(),
        observer=FakeObserver(),
        model="fake-model",
        policy=Fixed(1),
        max_iterations=1,
    )
    with pytest.raises(RecursionUnavailableError, match="DeafSandbox"):
        rlm.complete("ROOT question", context="corpus")


# -- winding down --------------------------------------------------------


def test_budget_refusal_winds_down_instead_of_raising() -> None:
    harness = build(
        scripted({"ROOT": ["CODE: NOOP", "FINAL: too late"]}),
        policy=Never(),
        calls_allowed=2,
        max_iterations=5,
    )
    run = harness.rlm.complete("ROOT question", context="corpus")

    attempt = run.attempts[0]
    assert attempt.outcome is Outcome.BUDGET_EXHAUSTED
    assert attempt.answer is None
    assert "too late" in attempt.detail
    assert harness.prompter.wind_downs
    # The wind-down call is billed and attributed like every other call.
    assert len(attempt.calls) == 2
    assert harness.budget.settled == 2


def test_a_budget_with_nothing_left_skips_the_wind_down_call() -> None:
    harness = build(
        scripted({"ROOT": ["FINAL: unreachable"]}),
        policy=Never(),
        calls_allowed=0,
    )
    run = harness.rlm.complete("ROOT question")

    attempt = run.attempts[0]
    assert attempt.outcome is Outcome.BUDGET_EXHAUSTED
    assert attempt.answer is None
    assert attempt.calls == ()
    assert harness.lm.calls == []


def test_the_policy_sees_a_reservation_that_cost_nothing_to_take() -> None:
    seen: list[CallReservation] = []

    class Recording:
        def next_depth(self, context: EscalationContext) -> int | None:
            seen.append(context.budget_reservation)
            return None

        def describe(self) -> str:
            return "recording"

    harness = build(scripted({"ROOT": ["FINAL: done"]}), policy=Recording())
    harness.rlm.complete("ROOT question")

    assert len(seen) == 1
    assert seen[0].granted
    assert 0 in harness.budget.reservations
    assert harness.budget.settled == 1


def test_the_policy_is_shown_only_cheap_content_free_signals() -> None:
    captured: list[dict[str, float]] = []

    class Recording:
        def next_depth(self, context: EscalationContext) -> int | None:
            captured.append(dict(context.signals))
            return None

        def describe(self) -> str:
            return "recording"

    harness = build(scripted({"ROOT": ["CODE: NOOP"]}), policy=Recording())
    harness.rlm.complete("ROOT question", context="corpus")

    assert captured[0]["root_calls"] == 2.0
    assert captured[0]["sub_calls"] == 0.0
    assert captured[0]["iterations"] == 2.0
    assert all(isinstance(v, float) for v in captured[0].values())


# -- failure modes -------------------------------------------------------


def test_a_broken_environment_closes_the_attempt_not_the_process() -> None:
    harness = build(
        scripted({"ROOT": ["CODE: NOOP"]}),
        policy=Escalating(),
        make_box=ExplodingSandbox,
    )
    run = harness.rlm.complete("ROOT question")

    assert run.attempts[0].outcome is Outcome.ERRORED
    assert "OSError" in run.attempts[0].detail
    # Escalating does not answer an environment failure with a costlier one.
    assert len(run.attempts) == 1
    assert harness.sandboxes[0].closed == 1


def test_a_model_that_says_nothing_useful_is_nudged_rather_than_crashed() -> None:
    harness = build(scripted({"ROOT": ["chatter", "FINAL: eventually"]}))
    run = harness.rlm.complete("ROOT question")

    assert harness.observer.nudged == ["chatter"]
    assert run.answer == "eventually"


def test_attempts_are_appended_in_ascending_depth_order() -> None:
    harness = build(
        scripted({"ROOT": ["CODE: NOOP"]}, {"ROOT": ["CODE: NOOP"]}),
        policy=Escalating(max_depth=3),
        calls_allowed=100,
    )
    run = harness.rlm.complete("ROOT question")

    depths = [a.max_depth for a in run.attempts]
    assert depths == [0, 1, 2, 3]


def test_a_policy_that_will_not_escalate_still_produces_a_run() -> None:
    class Backwards:
        def next_depth(self, context: EscalationContext) -> int | None:
            return 0

        def describe(self) -> str:
            return "backwards"

    harness = build(scripted({"ROOT": ["CODE: NOOP"]}), policy=Backwards())
    run = harness.rlm.complete("ROOT question")

    assert [a.max_depth for a in run.attempts] == [0]


def test_every_model_call_is_attributed_to_a_role_and_a_depth() -> None:
    harness = escalating_harness()
    run = harness.rlm.complete("ROOT question", context="corpus")

    assert len(run.calls) == len(harness.lm.calls)
    by_role = run.usage_by_role()
    assert by_role[Role.ROOT].total > 0
    assert by_role[Role.SUB].total > 0
    assert run.cost_usd == pytest.approx(0.001 * len(run.calls))
