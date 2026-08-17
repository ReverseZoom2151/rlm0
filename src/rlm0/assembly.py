"""The one place that wires the six layers into something a person can run.

Every layer in this project was built against its own fakes and every layer
passed. Nothing constructed an `RLM` with a real sandbox, a real provider and a
real budget until this module existed, and the moment something did, four
seams turned out not to meet. That is the ordinary outcome of building to a
port rather than to a caller, and it is why the wiring gets a module and a test
directory of its own instead of living in a README snippet.

The mismatches, and where each is bridged
-----------------------------------------
1. The sub-call contract has two names and two shapes. `prompt.py` tells the
   model about `llm_query(prompt)`, one argument, and `runtime.py` registers
   `rlm_call` with arity two, a query and the name of a variable in the
   parent's environment. The guest enforces arity exactly, so a model following
   its own instructions would have been handed a TypeError. Bridged by naming
   the call `llm_query` everywhere, registering it variadic so both shapes
   work, and adding one sentence to the sub-call half of the system prompt that
   documents the second argument. Without that sentence the by-name slice, the
   mechanism that keeps a sub-call's context out of the prompt, is a feature no
   model is ever told about.

2. Registration order. `ChannelSandbox.register_host_call` refuses a name with
   nothing bound behind it, and `runtime.py` registers before it binds. The
   sandbox port here holds the registration until a handler arrives, so either
   order works.

3. The budget probe. `runtime.py` reads remaining headroom with a zero-call
   reservation; `RunBudget` rejects a reservation below one call. The budget
   port here answers a zero-call reservation from `remaining()` without
   consuming anything. It also has to set `granted` itself: `remaining()`
   returns an ungranted reservation by design, and `policy.can_fund_another_
   attempt` reads `granted` first and would refuse every escalation. Granted
   here means "not exhausted", which is the question the policy is asking.

4. Prompting, parsing and observation are modules of functions, and the
   runtime wants objects. `_Bridge` is that adapter, and it carries the small
   amount of state the function signatures need and the port signatures do not
   pass: which code is being observed, whether anything has run yet this
   attempt, how large the context in front of this REPL is, and which sandbox
   to resolve a FINAL_VAR answer against.

Nothing here changes another module. If a bridge below looks like it is
compensating for a signature, it is, and the comment says which one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from rlm0.budget import RunBudget
from rlm0.observation import (
    DEFAULT_STDERR_CAP,
    DEFAULT_STDOUT_CAP,
    format_observation,
)
from rlm0.parse import CompletionSource, FinalKind, Rejection, parse_turn
from rlm0.policy import Escalating
from rlm0.ports import (
    Budget,
    CallReservation,
    DepthPolicy,
    ExecResult,
    HostCallable,
    LMClient,
    Sandbox,
    SandboxUnavailableError,
)
from rlm0.prompt import (
    SUB_CALL_CHAR_BUDGET,
    ContextShape,
    build_system_prompt,
    build_turn_prompt,
)
from rlm0.run import TokenUsage
from rlm0.runtime import RLM
from rlm0.sandbox import (
    DockerSandbox,
    MicroVMSandbox,
    SubprocessSandbox,
    docker_available,
    microvm_available,
)

__all__ = [
    "CONTEXT_VARIABLE",
    "DEFAULT_MAX_CALLS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MAX_USD",
    "SUB_CALL_NAME",
    "SandboxChoice",
    "build_rlm",
    "default_budget",
    "sandbox_factory",
]

SandboxChoice = Literal["docker", "microvm", "subprocess"]

CONTEXT_VARIABLE = "context"
"""The name the context is bound to inside the REPL.

Fixed rather than configurable because `prompt.py` writes the name into the
model's instructions as a literal. A configurable name would have to be
threaded into the prompt text, and a prompt that talks about `context` while
the runtime binds `CONTEXT` is a run where every block the model writes fails
with a NameError it cannot diagnose.
"""

SUB_CALL_NAME = "llm_query"
"""The name of the host call, for the same reason. See the module docstring."""

DEFAULT_MAX_USD = 2.0
DEFAULT_MAX_CALLS = 40
DEFAULT_MAX_SECONDS = 900.0
"""A ceiling on all three axes, because a default with none is the failure mode.

The numbers are meant to be small enough that a misconfigured run stops before
it is interesting on a bill, and large enough that a genuine depth-two
trajectory over a few million characters finishes. They are a default and not a
recommendation: a caller who knows the workload should pass a `RunBudget` that
says so.
"""

_SUB_CALL_HANDLE_NOTE = """\
One addition to the description above: `{name}` takes an optional second
argument, the *name* of a variable that was seeded into this REPL from outside,
as in `{name}("your question", "{context}")`. That form hands the sub-model the
whole of `{context}` without the text being copied into the call, which is
worth doing when you want a sub-model to read all of it.

The name form works only for `{context}` itself. For a slice you built here,
put the text in the prompt string, as the strategies above do; it goes straight
to the sub-model and never enters your own window either way."""


def default_budget() -> RunBudget:
    """A bounded run on every axis this project can bound.

    Exists as a function rather than a module constant so that the wall clock
    ceiling starts when the run does. A shared instance would carry one
    `_started_at` across every run in a process and would refuse the second
    sweep of a benchmark for reasons belonging to the first.
    """
    return RunBudget(
        max_usd=DEFAULT_MAX_USD,
        max_calls=DEFAULT_MAX_CALLS,
        max_seconds=DEFAULT_MAX_SECONDS,
    )


# -- the sub-call sandbox port -------------------------------------------


class _SandboxPort:
    """One real sandbox, wearing the shape `runtime.py` actually calls.

    Three jobs, all of them compensating for a seam rather than adding
    behaviour of its own.

    It holds a registration until something binds a handler for the name, so
    that the runtime's register-then-bind order works against a sandbox that
    refuses to expose a name it cannot service. Both orders work; whichever
    call completes the pair performs the registration.

    It registers host calls variadic. The guest checks arity exactly, and the
    two halves of this project disagree about how many arguments a sub-call
    takes, so an exact arity would break one of them. See the module docstring.

    It exposes `set_host_call_handler` as well as `bind_host_call`. The first
    is the structural hook the current `runtime.py` looks for and refuses to
    run a deep attempt without; the second is the seam now named in `ports.py`.
    Carrying both means this wiring does not break when the runtime migrates
    from one to the other, and the migration is the point of naming the seam.

    And it answers `get_variable` from a host-side copy while an execution is
    in flight, which is the one place where two layers of this project do not
    actually fit. See `_reentrant_read`.
    """

    _VARIADIC = -1

    def __init__(self, inner: Sandbox, bridge: _Bridge) -> None:
        self._inner = inner
        self._bridge = bridge
        self._pending: dict[str, int] = {}
        self._bound: set[str] = set()
        self._registered: set[str] = set()
        self._host_written: dict[str, str] = {}
        self._executing = False
        self._open = True
        bridge.push(self)

    # The Sandbox port

    def execute(self, code: str, *, timeout_s: float) -> ExecResult:
        self._executing = True
        try:
            return self._inner.execute(code, timeout_s=timeout_s)
        finally:
            self._executing = False

    def set_variable(self, name: str, value: str) -> None:
        # The one hook that tells the prompter how large the context in front
        # of this REPL is. `_drive` binds the context before it renders the
        # system prompt, and `Prompter.system` is not passed a size, so this is
        # where the shape for the next prompt comes from. A child REPL gets its
        # own sandbox and its own slice, so this is per-REPL and correct at
        # every depth.
        if name == self._bridge.context_variable:
            self._bridge.note_context(len(value))
        self._host_written[name] = value
        self._inner.set_variable(name, value)

    def get_variable(self, name: str) -> str | None:
        if self._executing:
            return self._reentrant_read(name)
        return self._inner.get_variable(name)

    def _reentrant_read(self, name: str) -> str | None:
        """A read taken while this sandbox is in the middle of an execution.

        This happens on every sub-call. The runtime hands a child the *name* of
        a variable in the parent's environment and reads the value out of the
        parent while servicing the call, which is what keeps a large slice from
        travelling through a prompt. But the call is being serviced from inside
        `execute`, and the sandbox channel cannot carry a control round trip
        then: the guest is blocked awaiting the reply to its host call and
        rejects anything that is not one, so a `get_var` at that moment
        desynchronizes the stream, kills the sandbox, and loses the whole REPL
        state. That is not a bug in either layer on its own. It is the two
        designs meeting: the runtime assumes the sandbox is randomly
        addressable, and the channel is a strictly request-response pipe.

        The half that can be served without the channel is served: anything the
        host bound with `set_variable` is still held here, and `context`, the
        variable this mechanism was built for, is always one of those.

        The half that cannot is refused loudly. A name the model bound inside
        the REPL cannot be read out mid-execution by anybody, so returning None
        would hand the child an empty context and produce a sub-call that looks
        serviced and read nothing, which is the exact failure this project
        exists to make visible. Raising instead surfaces on the model's side as
        a failed `llm_query` with this text in it, and the model can pass the
        value inline instead.
        """
        if name in self._host_written:
            return self._host_written[name]
        raise RuntimeError(
            f"{name!r} was bound inside the REPL, and a variable bound inside "
            "cannot be read back out while your code is still running: the "
            "sandbox channel is busy carrying this very call. Only variables "
            "the runtime seeded from outside, such as "
            f"{self._bridge.context_variable!r}, can be passed by name. Pass "
            "the text itself as part of the prompt instead."
        )

    def register_host_call(self, name: str, arity: int) -> None:
        del arity  # see _VARIADIC
        self._pending[name] = self._VARIADIC
        self._settle_registration(name)

    def bind_host_call(self, name: str, function: HostCallable) -> None:
        self._inner.bind_host_call(name, function)
        self._bound.add(name)
        self._settle_registration(name)

    def set_host_call_handler(self, name: str, handler: HostCallable) -> None:
        """The name the current runtime looks for. Same seam, older spelling."""
        self.bind_host_call(name, handler)

    def close(self) -> None:
        if self._open:
            self._open = False
            self._bridge.pop(self)
        self._inner.close()

    def _settle_registration(self, name: str) -> None:
        if name in self._registered or name not in self._pending:
            return
        if name not in self._bound:
            return
        self._inner.register_host_call(name, self._pending[name])
        self._registered.add(name)


def sandbox_factory(
    choice: SandboxChoice = "docker", **options: Any
) -> Callable[[], Sandbox]:
    """A factory for one kind of sandbox, checked now rather than at first use.

    Docker availability is probed here so that an unusable choice fails while
    the caller is still configuring, which is what `SandboxUnavailableError`
    asks for: a sandbox that reports its absence only when the model finally
    writes code has already let somebody set up a long run against a promise it
    cannot keep.

    The returned factory is called once per attempt and once per sub-call, and
    each call must return a fresh environment. A shared sandbox would let a
    depth-two attempt read variables the depth-zero control had bound, which
    would make the control not a control.
    """
    if choice == "docker":
        if not docker_available(str(options.get("binary", "docker"))):
            raise SandboxUnavailableError(
                "the default sandbox is Docker and no usable Docker was found. "
                "Install or start it, or pass sandbox='subprocess' and read "
                "SubprocessSandbox's docstring first: it isolates the "
                "orchestrator from crashes and is not a security boundary, so "
                "it is only safe for a context you wrote yourself."
            )
        return lambda: DockerSandbox(**options)
    if choice == "subprocess":
        return lambda: SubprocessSandbox(**options)
    if choice == "microvm":
        binary = str(options.get("binary", "docker"))
        runtime = str(options.get("runtime", "krun"))
        if not microvm_available(binary, runtime):
            raise SandboxUnavailableError(
                f"no usable {runtime!r} microVM runtime is registered with "
                f"{binary!r}. Configure it first, or choose sandbox='docker'."
            )
        return lambda: MicroVMSandbox(**options)
    raise ValueError(f"unknown sandbox choice {choice!r}")


# -- the budget port -----------------------------------------------------


class _BudgetPort:
    """A budget that also answers the runtime's zero-call probe.

    `runtime.py` reads headroom by reserving zero calls; `RunBudget` refuses
    any reservation below one. Rather than let that crash a run at the first
    escalation decision, the probe is answered from `remaining()`, which takes
    nothing.

    The `granted` flag has to be synthesised. `remaining()` reports False
    because nothing was reserved, and `policy.can_fund_another_attempt` reads
    `granted` first and would then refuse every escalation the project exists
    to make. So a probe is granted when the budget is not exhausted, which is
    the question the policy is actually asking, and the headroom numbers come
    through untouched.
    """

    def __init__(self, inner: Budget) -> None:
        self._inner = inner

    def reserve(self, *, n_calls: int, estimated_tokens: int) -> CallReservation:
        if n_calls < 1:
            return self._probe()
        return self._inner.reserve(
            n_calls=n_calls, estimated_tokens=estimated_tokens
        )

    def settle(self, usage: TokenUsage, cost_usd: float | None) -> None:
        self._inner.settle(usage, cost_usd)

    def release(self, *, n_calls: int) -> None:
        """Return an unused hold when the wrapped budget supports refunds."""
        release = getattr(self._inner, "release", None)
        if callable(release):
            release(n_calls=n_calls)

    def remaining(self) -> CallReservation:
        return self._inner.remaining()

    def summary(self) -> str:
        return self._inner.summary()

    @property
    def exhausted(self) -> bool:
        return self._inner.exhausted

    def _probe(self) -> CallReservation:
        headroom = self._inner.remaining()
        return CallReservation(
            granted=not self._inner.exhausted,
            reason=headroom.reason,
            calls_remaining=headroom.calls_remaining,
            usd_remaining=headroom.usd_remaining,
            seconds_remaining=headroom.seconds_remaining,
        )


# -- prompting, parsing and observation ----------------------------------


@dataclass(frozen=True, slots=True)
class _Turn:
    """One parsed turn, reduced to the two questions the loop asks."""

    code: str | None
    final_answer: str | None
    completion_source: CompletionSource | None = None


@dataclass(slots=True)
class _Frame:
    """The state of one REPL, at one depth, inside one attempt.

    A frame and not a field on the bridge, because a sub-call opens a second
    REPL underneath the first and both are live at once. Holding this state
    flat would let a child's parse overwrite what the parent was doing, and the
    symptom is quiet: the parent's next observation echoes the child's code
    back at it as the block that just ran.
    """

    sandbox: _SandboxPort | None = None
    context_chars: int = 0
    sub_calls: bool = False
    task: str = ""
    code_has_run: bool = False
    last_code: str = ""
    last_rejection: Rejection | None = None
    missing_variable: str = ""


class _Bridge:
    """`prompt`, `parse` and `observation` as the three objects the loop wants.

    All three modules are deliberately stateless functions, and all three ports
    are objects whose signatures carry less than the functions need. The gap is
    made of four things, and this class holds exactly those four and nothing
    else:

    - the size of the context in front of the current REPL, which the system
      prompt states and `Prompter.system` is not given;
    - whether any code has run yet in the current REPL, which `parse_turn` uses
      to reject an answer produced from priors and `TurnParser.parse` is not
      given;
    - the code whose result is being rendered, which the observation echoes
      back and `ObservationFormatter.format` is not given;
    - the sandbox to read a FINAL_VAR answer out of, which nothing in these
      three ports mentions at all.

    All four are per-REPL, and the loop is depth first and single threaded, so
    they are kept as a stack that `_SandboxPort` pushes and pops. The reset
    points are real calls: `opening` starts a REPL, `set_variable` seeds it.
    """

    def __init__(
        self,
        *,
        context_variable: str = CONTEXT_VARIABLE,
        sub_call_chars: int = SUB_CALL_CHAR_BUDGET,
        stdout_cap: int = DEFAULT_STDOUT_CAP,
        stderr_cap: int = DEFAULT_STDERR_CAP,
    ) -> None:
        self.context_variable = context_variable
        self._sub_call_chars = sub_call_chars
        self._stdout_cap = stdout_cap
        self._stderr_cap = stderr_cap
        self._frames: list[_Frame] = []
        self._orphan = _Frame()

    # State the ports do not carry

    @property
    def _frame(self) -> _Frame:
        """The REPL currently being driven.

        The orphan frame is a fallback for a call that arrives with no sandbox
        alive, which the loop should never produce. It keeps a wiring mistake
        as a wrong prompt rather than an IndexError inside somebody else's
        traceback.
        """
        return self._frames[-1] if self._frames else self._orphan

    def push(self, sandbox: _SandboxPort) -> None:
        self._frames.append(_Frame(sandbox=sandbox))

    def pop(self, sandbox: _SandboxPort) -> None:
        for index in range(len(self._frames) - 1, -1, -1):
            if self._frames[index].sandbox is sandbox:
                del self._frames[index]
                return

    def note_context(self, chars: int) -> None:
        self._frame.context_chars = chars

    # Prompter

    def system(self, *, max_depth: int, sub_call_name: str | None) -> str:
        del max_depth  # the prompt is told what exists, never how deep it is
        frame = self._frame
        frame.sub_calls = sub_call_name is not None
        prompt = build_system_prompt(
            ContextShape(total_chars=frame.context_chars),
            sub_calls=frame.sub_calls,
            sub_call_chars=self._sub_call_chars,
        )
        if sub_call_name is None:
            return prompt
        # Only ever added to the sub-call variant, so the depth-zero control is
        # byte identical to what `prompt.py` produces and the comparison the
        # project is built on is unaffected.
        note = _SUB_CALL_HANDLE_NOTE.format(
            name=sub_call_name, context=self.context_variable
        )
        return prompt + "\n\n" + note

    def opening(
        self, *, task: str, context_variable: str, context_chars: int
    ) -> str:
        del context_variable  # fixed at CONTEXT_VARIABLE; see the constant
        frame = self._frame
        frame.task = task
        frame.context_chars = context_chars
        # A new REPL has run nothing, so a final answer on its first turn is
        # the model answering from memory. This is the reset point that keeps
        # that check per-REPL rather than per-process.
        frame.code_has_run = False
        frame.last_code = ""
        frame.last_rejection = None
        frame.missing_variable = ""
        return build_turn_prompt(task, iteration=0)

    def wind_down(self, *, reason: str) -> str:
        return (
            f"The run is stopping now: {reason}\n\n"
            + build_turn_prompt(self._frame.task, wrap_up=True)
        )

    # TurnParser

    def parse(self, text: str) -> _Turn:
        frame = self._frame
        parsed = parse_turn(text, code_has_run=frame.code_has_run)
        frame.last_rejection = parsed.rejection
        frame.missing_variable = ""
        blocks = parsed.executable_blocks
        code = "\n".join(block.code for block in blocks) if blocks else None
        frame.last_code = code or ""
        answer: str | None = None
        if parsed.final is not None:
            if parsed.final.kind is FinalKind.LITERAL:
                answer = parsed.final.value
            else:
                answer = self._read_variable(parsed.final.value)
        return _Turn(
            code=code,
            final_answer=answer,
            completion_source=None if parsed.final is None else parsed.final.source,
        )

    def _read_variable(self, name: str) -> str | None:
        """Resolve FINAL_VAR against the REPL that produced it.

        The port hands the loop an answer, not a variable name, so the read has
        to happen here. It is the mechanism by which an answer longer than an
        output limit gets out, and a bridge that returned the name instead
        would make every FINAL_VAR run report the string "summary" as its
        answer, which no assertion downstream could catch.
        """
        frame = self._frame
        if frame.sandbox is None:  # pragma: no cover - no REPL to read from
            frame.missing_variable = name
            return None
        value = frame.sandbox.get_variable(name)
        if value is None:
            frame.missing_variable = name
        return value

    # ObservationFormatter

    def format(self, result: ExecResult) -> str:
        frame = self._frame
        frame.code_has_run = True
        return format_observation(
            frame.last_code,
            result,
            stdout_cap=self._stdout_cap,
            stderr_cap=self._stderr_cap,
            sub_calls=frame.sub_calls,
        )

    def format_no_action(self, text: str) -> str:
        del text  # the model has its own turn in the transcript already
        frame = self._frame
        if frame.missing_variable:
            return (
                f"You ended with FINAL_VAR({frame.missing_variable}), but no "
                f"variable named {frame.missing_variable} is bound in the "
                "REPL. Bind it, or give the answer with FINAL(...)."
            )
        return _NO_ACTION.get(frame.last_rejection, _NO_ACTION_DEFAULT)


_NO_ACTION_DEFAULT = (
    "That turn ran no code and gave no answer. Write a ```repl block that "
    "does something, or finish with FINAL(...) or FINAL_VAR(...)."
)

_NO_ACTION: dict[Rejection | None, str] = {
    Rejection.NO_CODE_HAS_RUN: (
        "You gave a final answer before running any code, so it came from "
        "memory rather than from the context, and it was not accepted. Run "
        "something first: establish the type and size of `context` and where "
        "the query's terms occur, then answer."
    ),
    Rejection.MALFORMED_VARIABLE: (
        "FINAL_VAR takes a bare variable name and nothing else. Either name a "
        "variable you have bound, or use FINAL(...) with the answer itself."
    ),
    Rejection.EMPTY_ANSWER: (
        "FINAL() was empty. Put the complete answer inside the parentheses."
    ),
    Rejection.CODE_IN_SAME_TURN: (
        "You wrote both code and a final answer in one turn. The code ran and "
        "the answer was held back; repeat the answer on its own next turn if "
        "the output has not changed your mind."
    ),
    Rejection.MALFORMED_PROTOCOL: (
        "RLM0_FINAL_V1 must be one JSON object with protocol_version 1, "
        "status answered, a non-empty answer, an evidence list and an "
        "answer_artifact field. Fix the envelope and try again."
    ),
}


# -- the assembly --------------------------------------------------------


def build_rlm(
    *,
    model: str,
    lm: LMClient | None = None,
    budget: Budget | None = None,
    sandbox: SandboxChoice | Callable[[], Sandbox] = "docker",
    policy: DepthPolicy | None = None,
    sub_model: str | None = None,
    max_iterations: int = 8,
    max_tokens: int = 4096,
    max_attempts: int = 4,
    experimental_depth: bool = False,
    exec_timeout_s: float = 30.0,
    attempt_timeout_s: float | None = None,
    sub_call_chars: int = SUB_CALL_CHAR_BUDGET,
    stdout_cap: int = DEFAULT_STDOUT_CAP,
    stderr_cap: int = DEFAULT_STDERR_CAP,
    clock: Callable[[], float] = time.monotonic,
    sandbox_options: dict[str, Any] | None = None,
) -> RLM:
    """A working `RLM` from ordinary arguments.

    The default configuration is a Docker sandbox, a `RunBudget` bounded on
    cost, calls and wall clock, and the escalating policy, which is depth zero
    first and one rung deeper per failed attempt up to two. All three defaults
    are the safe end of their axis: the only sandbox that is a boundary, a
    ceiling that exists on every dimension this project can measure, and a
    policy that never pays for recursion until the cheap attempt has actually
    failed.

    `lm` defaults to None and must be supplied. There is no default provider
    on purpose: picking one here would be exactly the hardcoded-three-layers-
    down choice that `ports.py` exists to prevent, and it would make a run
    against the wrong account one forgotten argument away.
    """
    if lm is None:
        raise ValueError(
            "build_rlm needs an LMClient. Pass AnthropicClient(), "
            "OpenAIClient(), or FakeClient() for a test; nothing here picks a "
            "provider for you."
        )
    bridge = _Bridge(
        context_variable=CONTEXT_VARIABLE,
        sub_call_chars=sub_call_chars,
        stdout_cap=stdout_cap,
        stderr_cap=stderr_cap,
    )
    make_sandbox = (
        sandbox
        if callable(sandbox)
        else sandbox_factory(sandbox, **(sandbox_options or {}))
    )

    def factory() -> Sandbox:
        return _SandboxPort(make_sandbox(), bridge)

    return RLM(
        lm=lm,
        sandbox_factory=factory,
        budget=_BudgetPort(default_budget() if budget is None else budget),
        prompter=bridge,
        parser=bridge,
        observer=bridge,
        model=model,
        policy=Escalating() if policy is None else policy,
        sub_model=sub_model,
        max_iterations=max_iterations,
        max_tokens=max_tokens,
        max_attempts=max_attempts,
        exec_timeout_s=exec_timeout_s,
        attempt_timeout_s=attempt_timeout_s,
        context_variable=CONTEXT_VARIABLE,
        sub_call_name=SUB_CALL_NAME,
        experimental_depth=experimental_depth,
        clock=clock,
    )
