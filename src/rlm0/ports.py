"""The seams between the layers, so each can be replaced or faked.

These are Protocols rather than base classes because every one of them has a
test double that should not have to inherit anything, and because the point of
naming a seam is that somebody else can implement it. The survey behind this
project found the opposite pattern repeatedly: a provider hardcoded three
layers down, a sandbox chosen by a string compared inside the orchestrator,
and in one case an entire recursion path silently disabled for every remote
environment because the wiring lived in a conditional nobody revisited.

Nothing here holds behaviour. The contracts that matter are stated as
docstrings on the methods, and the constraint that recurs is that every one of
these seams must report what it spent, because a layer that can hide its own
cost makes the run-level accounting a fiction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from rlm0.run import TokenUsage

__all__ = [
    "Budget",
    "CallReservation",
    "DepthPolicy",
    "ExecResult",
    "LMClient",
    "LMResponse",
    "Sandbox",
    "SandboxUnavailableError",
]


class SandboxUnavailableError(RuntimeError):
    """No sandbox could be built, and the runtime will not proceed without one.

    Raised at construction rather than at first execution. A sandbox that
    reports its absence only when the model finally writes code has already
    let the caller set up a long run against a promise it cannot keep.
    """


@dataclass(frozen=True, slots=True)
class LMResponse:
    """One completion, with what it cost attached to it.

    `cost_usd` stays None when the provider returned no price and no table
    covered the model, never zero. `cached_prefix` records whether the
    provider reported reading a cached prefix, which is the fan-out diagnostic
    the run layer aggregates.
    """

    text: str
    usage: TokenUsage
    model: str
    wall_clock_s: float
    cost_usd: float | None = None
    cached_prefix: bool = False
    stop_reason: str = ""

    @property
    def truncated(self) -> bool:
        """Whether the provider stopped on a length limit rather than by choice.

        Worth a property because a silently truncated sub-call answer looks
        exactly like a short one, and the surveyed implementations that fed
        sub-results straight into a variable had no way to tell the difference.
        """
        return self.stop_reason in {"max_tokens", "length"}


@runtime_checkable
class LMClient(Protocol):
    """A provider, reduced to the one call this project makes.

    Implementations must return real provider-reported token counts. Deriving
    usage from string length is not acceptable here, because every cost claim
    the runtime makes is built on these numbers and an estimate that is wrong
    by a constant factor is indistinguishable from a correct one until the
    bill arrives.
    """

    def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, str]],
        model: str,
        max_tokens: int,
        cache_prefix: bool = False,
    ) -> LMResponse:
        """Issue one completion.

        `cache_prefix` asks the provider to mark the system block and the
        stable head of the conversation as cacheable. Implementations that
        cannot do this must ignore it rather than emulate it, and must leave
        `cached_prefix` False, so the diagnostic stays honest.
        """
        ...


@dataclass(frozen=True, slots=True)
class ExecResult:
    """What came back from running one block of model-written code."""

    stdout: str
    stderr: str
    wall_clock_s: float
    ok: bool
    truncated_stdout: bool = False
    variables: tuple[str, ...] = ()
    """Names currently bound in the environment, without their values.

    Names only. The whole point of holding the context in the environment is
    that its contents never enter the model's window, and a variable dump is
    the most direct way to undo that by accident.
    """


@runtime_checkable
class Sandbox(Protocol):
    """Somewhere to run code the model wrote, that the model cannot escape.

    The threat here is sharper than in an ordinary coding agent. The context
    is attacker-controlled text and it shares an interpreter with a model that
    writes code, so prompt injection and code execution are one step apart by
    construction. Implementations are therefore expected to keep the network
    closed and to keep credentials outside the boundary, which is why
    `register_host_call` exists: a sub-call must be serviceable without the
    sandbox ever holding an API key or opening a socket.

    An implementation that cannot hold those properties should raise
    SandboxUnavailableError rather than degrade quietly.
    """

    def execute(self, code: str, *, timeout_s: float) -> ExecResult:
        """Run one block. Must return on timeout rather than hanging."""
        ...

    def set_variable(self, name: str, value: str) -> None:
        """Bind a value inside without routing it through the model."""
        ...

    def get_variable(self, name: str) -> str | None:
        """Read one value out by name, for returning an answer larger than a window."""
        ...

    def register_host_call(self, name: str, arity: int) -> None:
        """Expose a host-serviced function to code running inside.

        The implementation marshals the call back out; it does not hand the
        sandbox the means to make it.
        """
        ...

    def close(self) -> None:
        """Release the sandbox. Must be safe to call twice."""
        ...


@dataclass(frozen=True, slots=True)
class CallReservation:
    """Permission to make one call, taken before the call is made.

    Reserving first is what makes a ceiling real. Checking a counter after the
    fact bounds nothing, and a fan-out that checks between dispatches lands
    half its calls before noticing. A reservation that cannot be granted is
    the signal to wind down, not an error to raise at the model.
    """

    granted: bool
    reason: str = ""
    calls_remaining: int | None = None
    usd_remaining: float | None = None
    seconds_remaining: float | None = None


@runtime_checkable
class Budget(Protocol):
    """A ceiling shared by every call in one run, across every depth.

    Shared rather than per-level. A per-level budget multiplies with fan-out,
    which is precisely how the cost tail in the published work is generated:
    the runs that blow up are the ones that never find the answer and keep
    spawning children to look for it.
    """

    def reserve(self, *, n_calls: int, estimated_tokens: int) -> CallReservation:
        """Ask before dispatching. A batch reserves all of its calls at once."""
        ...

    def settle(self, usage: TokenUsage, cost_usd: float | None) -> None:
        """Record what a granted call actually consumed."""
        ...

    def summary(self) -> str:
        """One line naming the ceilings, for the run record."""
        ...

    @property
    def exhausted(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class EscalationContext:
    """What is known when deciding whether to go deeper.

    Deliberately small. The decision is made after a real depth-zero attempt,
    so the most informative input is simply whether that attempt produced an
    answer and what it cost, and a policy that needs more than this is
    probably reaching for the task label, which is how a router becomes
    overfitted to a benchmark.
    """

    task: str
    context_chars: int
    attempts_so_far: int
    last_outcome: str
    last_answer: str | None
    spent_usd: float | None
    elapsed_s: float
    budget_reservation: CallReservation
    signals: dict[str, float] = field(default_factory=dict)


@runtime_checkable
class DepthPolicy(Protocol):
    """Decides whether to try again deeper, having already tried shallower.

    This exists as a seam because the evidence says the right answer varies by
    task and by model. Recursion is worth a great deal on dense aggregation
    and is measurably harmful on retrieval and on anything that fits the
    window, so a fixed depth is wrong in one direction or the other on most
    queries. Making the policy replaceable is also what lets the harness
    measure policies against each other rather than arguing about them.
    """

    def next_depth(self, context: EscalationContext) -> int | None:
        """Return the bound for the next attempt, or None to stop.

        Returning None when the previous attempt answered is the common case
        and is what keeps the cheap path cheap.
        """
        ...

    def describe(self) -> str:
        """Name the policy, for the run record."""
        ...
