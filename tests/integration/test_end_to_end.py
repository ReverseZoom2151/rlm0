"""The tests this project did not have: the layers, running against each other.

Every other test directory here proves that one layer behaves against fakes of
its neighbours. That is worth having and it is not this. Six layers each
passing against its own doubles is exactly the situation in which two of them
can disagree about the number of arguments a sub-call takes, about whether a
name may be registered before it is bound, and about whether reserving zero
calls is a question or an error, and no test anywhere fails.

So the sandbox in these tests is a real `SubprocessSandbox` or a real
`DockerSandbox`, running real model-written Python in a real child process, and
sub-calls really are marshalled out over the pipe and serviced by a second REPL
on the host. The budget is a real `RunBudget` and the policy is the real
`Escalating`. The model is scripted, because the alternative is a test that
needs a key and therefore never runs, and every reply below is text a model
could plausibly have emitted against the prompt it was given.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rlm0.assembly import SandboxChoice, build_rlm, default_budget, sandbox_factory
from rlm0.budget import RunBudget
from rlm0.policy import Escalating
from rlm0.ports import Budget, SandboxUnavailableError
from rlm0.providers.fake import FakeClient, FakeReply
from rlm0.run import Outcome, Role, Run, Verdict
from rlm0.sandbox import docker_available

CONTEXT = "alpha bravo charlie delta 41 echo foxtrot 42 golf\n" * 40
SECRET = "sk-rlm0-integration-fixture-0000000000000000"


def repl(code: str) -> str:
    """One assistant turn that runs a block, as the prompt asks for it."""
    return f"Here is the next step.\n\n```repl\n{code}\n```"


def drive(
    replies: Sequence[str | FakeReply],
    *,
    sandbox: SandboxChoice,
    budget: Budget | None = None,
    task: str = "what number appears most often",
    context: str = CONTEXT,
    max_iterations: int = 2,
    policy: object | None = None,
) -> tuple[Run, FakeClient]:
    """Run one scripted trajectory end to end and hand back both halves.

    The client is returned as well as the run because half of what these tests
    assert is about what did and did not cross into a model window, and that
    question is only answerable from the recorded calls.
    """
    scripted = [r if isinstance(r, FakeReply) else FakeReply(text=r) for r in replies]
    lm = FakeClient(replies=scripted)
    rlm = build_rlm(
        model="fake-model",
        lm=lm,
        sandbox=sandbox,
        budget=RunBudget(max_calls=30) if budget is None else budget,
        max_iterations=max_iterations,
        policy=Escalating() if policy is None else policy,  # type: ignore[arg-type]
        exec_timeout_s=60.0,
    )
    return rlm.complete(task, context), lm


def windows(lm: FakeClient) -> str:
    """Every character that ever entered a model window, concatenated."""
    parts: list[str] = []
    for call in lm.calls:
        parts.append(call.system)
        parts.extend(content for _, content in call.messages)
    return "\n".join(parts)


# -- the whole loop, escalating ------------------------------------------


def test_a_two_attempt_escalation_over_a_real_sandbox(
    sandbox_choice: SandboxChoice,
) -> None:
    """The test that proves the halves fit.

    Depth zero runs out of iterations, the policy steps to depth one, the
    deeper attempt makes a genuine sub-call through the sandbox boundary and
    answers with what came back. Everything the `Run` then claims about the
    comparison is a by-product of that trajectory rather than a second
    experiment.
    """
    run, lm = drive(
        [
            repl("print('chars', len(context))"),
            repl("print('still looking')"),
            # depth one: the sub-call, by name, so the slice never enters a
            # prompt written by the root model.
            repl(
                "out = llm_query('which number wins', 'context')\n"
                "print('GOT:' + out)"
            ),
            repl("print('child sees', len(context))"),
            "FINAL(42)",
            "FINAL(42)",
        ],
        sandbox=sandbox_choice,
    )

    assert [a.max_depth for a in run.attempts] == [0, 1]
    baseline = run.baseline
    assert baseline is not None
    assert baseline.outcome is Outcome.ITERATIONS_EXHAUSTED
    assert baseline.n_sub_calls == 0

    deeper = run.attempts[1]
    assert deeper.outcome is Outcome.ANSWERED
    assert run.answer == "42"
    assert deeper.n_sub_calls == 2, "the child REPL's calls belong to this attempt"

    verdict = run.recursion_verdict()
    assert verdict.verdict is Verdict.HELPED
    assert verdict.extra_sub_calls == 2
    assert verdict.extra_cost_usd is not None and verdict.extra_cost_usd > 0

    # The accounting the run record exists for, filled in by real calls.
    assert run.usage_by_role()[Role.SUB].total > 0
    assert run.budget_summary.startswith("RunBudget[shared]")
    assert "recursion helped" in run.describe()

    # The sub-call really was made: the child printed the size of the slice it
    # was handed, and that slice is the parent's context.
    assert f"child sees {len(CONTEXT)}" in windows(lm)
    assert "GOT:42" in windows(lm)


def test_depth_zero_answers_and_the_sub_call_name_is_unbound_there(
    sandbox_choice: SandboxChoice,
) -> None:
    """One attempt, and the control is a control inside the interpreter too.

    A depth-zero attempt that merely omits the sub-call from its prompt is not
    a control, because the model can still call the name. This asserts on what
    the running interpreter says, not on what the prompt claims.
    """
    run, lm = drive(
        [
            repl(
                "try:\n"
                "    llm_query\n"
                "    print('SUBCALL BOUND')\n"
                "except NameError:\n"
                "    print('SUBCALL UNBOUND')"
            ),
            "FINAL(42, and depth zero was enough)",
        ],
        sandbox=sandbox_choice,
    )

    assert len(run.attempts) == 1
    assert run.attempts[0].max_depth == 0
    assert run.attempts[0].outcome is Outcome.ANSWERED
    assert run.answer == "42, and depth zero was enough"
    assert run.recursion_verdict().verdict is Verdict.NOT_ATTEMPTED
    assert all(call.role is Role.ROOT for call in run.calls)

    seen = windows(lm)
    # `format_observation` deliberately echoes the code that ran, so the
    # source text contains both markers. Assert on the observed stdout, not on
    # the complete transcript, or this test would reject the code it supplied
    # as its own fixture.
    assert "REPL output:\nSUBCALL UNBOUND" in seen
    assert "REPL output:\nSUBCALL BOUND" not in seen


def test_a_sub_call_crosses_the_boundary_and_no_credential_goes_with_it(
    sandbox_choice: SandboxChoice,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The value comes back; the key never leaves the host.

    This is the property the whole host-call design exists for: the sandbox can
    reach a model without holding the means to reach one. The host process is
    given a credential-shaped environment variable for the duration, and the
    test asserts it is invisible from inside, absent from every model window,
    and absent from everything the sandbox printed.
    """
    monkeypatch.setenv("RLM0_INTEGRATION_API_KEY", SECRET)

    run, lm = drive(
        [
            repl(
                "import os\n"
                "print('KEY:', os.environ.get('RLM0_INTEGRATION_API_KEY'))\n"
                "print('names', len([k for k in os.environ if 'KEY' in k]))"
            ),
            repl("print('no answer yet')"),
            repl(
                "reply = llm_query('summarise the context', 'context')\n"
                "print('CHILD SAID:' + reply)"
            ),
            repl("print('child chars', len(context))"),
            "FINAL(the child read it)",
            "FINAL(42)",
        ],
        sandbox=sandbox_choice,
    )

    assert run.answer == "42"
    seen = windows(lm)
    assert "CHILD SAID:the child read it" in seen, "the value reached the caller's code"
    assert f"child chars {len(CONTEXT)}" in seen, "the payload reached the child's REPL"

    assert "KEY: None" in seen, "the credential was not inside the sandbox"
    assert SECRET not in seen, "no credential entered a model window"
    for attempt in run.attempts:
        assert attempt.calls, "every attempt made real calls"


# -- the ceiling ----------------------------------------------------------


def test_a_budget_that_binds_mid_flight_winds_down_instead_of_raising() -> None:
    """Refusal is a signal, and the run survives it as a shorter run.

    Three calls is enough for one working turn and the wind-down that follows
    it, and not enough for a second turn. The attempt must close as
    BUDGET_EXHAUSTED carrying its partial reply as detail rather than as an
    answer, and the policy must then decline to escalate into a spent ceiling
    rather than buying the front half of a deeper trajectory.
    """
    budget = RunBudget(max_calls=3)
    run, lm = drive(
        [repl("print('looking')")] * 4,
        sandbox="subprocess",
        budget=budget,
        max_iterations=8,
    )

    assert len(run.attempts) == 1, "a spent budget must not fund a deeper attempt"
    attempt = run.attempts[0]
    assert attempt.outcome is Outcome.BUDGET_EXHAUSTED
    assert attempt.answer is None
    assert "partial reply not counted as an answer" in attempt.detail
    assert run.answer is None

    # The wind-down really was sent, and it named the ceiling to the model.
    assert "The run is stopping now" in windows(lm)
    assert lm.call_count == 3
    assert budget.snapshot().calls_settled == 3
    assert "calls=3" in run.budget_summary


def test_the_zero_call_probe_does_not_consume_the_budget() -> None:
    """The runtime asks what is left between attempts; asking must be free.

    `RunBudget` refuses a reservation below one call outright, so the probe is
    answered from `remaining()`. If that bridge ever regresses the symptom is
    not a crash but a run that is quietly one call poorer per escalation
    decision, so it is asserted on the ledger.
    """
    budget = RunBudget(max_calls=30)
    run, _ = drive(
        [repl("print('one')"), "FINAL(42)"],
        sandbox="subprocess",
        budget=budget,
    )
    assert run.answer == "42"
    assert budget.snapshot().calls_settled == 2
    assert budget.snapshot().calls_in_flight == 0


# -- answers larger than a window ----------------------------------------


def test_final_var_returns_a_value_that_never_entered_a_window() -> None:
    """The mechanism for an answer too large to generate into the transcript.

    The model binds the answer in the REPL and names the variable. The value
    has to be read out of the sandbox by the runtime, and the assertion that
    matters is the second one: the answer is in the `Run` and nowhere in the
    conversation.
    """
    body = "long-answer-fragment " * 500
    run, lm = drive(
        [
            repl("answer = 'long-answer-fragment ' * 500\nprint(len(answer))"),
            "The answer is in the variable.\n\nFINAL_VAR(answer)",
        ],
        sandbox="subprocess",
    )

    assert run.attempts[0].outcome is Outcome.ANSWERED
    assert run.answer == body
    assert body not in windows(lm), "the answer must not have crossed a window"


def test_a_variable_bound_inside_the_repl_cannot_be_passed_by_name() -> None:
    """A seam that does not fit, asserted rather than papered over.

    `runtime.py` services a sub-call by reading the named variable out of the
    parent's environment while the parent is mid-execution. The channel cannot
    carry that: the guest is blocked awaiting the reply to its own host call
    and rejects anything else, so a control round trip there desynchronizes the
    stream and destroys the REPL. `_SandboxPort` serves the names the host
    itself bound, which covers `context`, and refuses the rest loudly.

    Loudly is the whole point. Returning None would hand the child an empty
    context and produce a sub-call that reports success and read nothing.
    """
    run, lm = drive(
        [
            repl("print('first')"),
            repl("print('second')"),
            repl(
                "slice_ = context[:100]\n"
                "print(llm_query('what is here', 'slice_'))"
            ),
            repl("print('unreachable')"),
        ],
        sandbox="subprocess",
    )

    seen = windows(lm)
    assert "cannot be read back out while your code is still running" in seen
    # The run survives the refusal: the attempt closes on its own terms.
    assert run.attempts[1].outcome is not Outcome.ERRORED
    assert run.baseline is not None


# -- the assembly's own defaults -----------------------------------------


def test_the_default_budget_is_bounded_on_every_axis() -> None:
    summary = default_budget().summary()
    assert "usd=unset" not in summary
    assert "calls=unset" not in summary
    assert "seconds=unset" not in summary


def test_the_default_policy_is_escalating_and_is_recorded_on_the_run() -> None:
    run, _ = drive(
        [repl("print('one')"), "FINAL(42)"],
        sandbox="subprocess",
    )
    assert run.labels["policy"].startswith("escalating")


def test_no_provider_is_chosen_for_you() -> None:
    with pytest.raises(ValueError, match="LMClient"):
        build_rlm(model="fake-model", lm=None, sandbox="subprocess")


@pytest.mark.skipif(docker_available(), reason="Docker is present here")
def test_the_default_sandbox_refuses_at_build_time_when_docker_is_absent() -> None:
    """Fail at configuration, not at the first block the model writes.

    Tested only where there is no daemon, and it is the case that matters: a
    long run set up against a sandbox that was never going to exist.
    """
    with pytest.raises(SandboxUnavailableError, match="Docker"):
        sandbox_factory("docker")
