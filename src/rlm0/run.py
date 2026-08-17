"""What a run is, and what it may not be without.

This module is the whole argument of the project expressed as types.

Recursive language models are reported on almost entirely by assertion. The
published literature and every open implementation surveyed for this project
share two claims that neither can support from their own artifacts: that the
recursion helped, and that it was cheap. The first is contested by the source
paper's own depth ablations, by an independent reproduction that found depth
two degrading, and by a critique showing recursion hurting inside the native
context window. The second is undermined by a cost distribution whose standard
deviation runs at roughly double its mean, because the expensive trajectories
are exactly the ones that never find the answer.

Neither claim is hard to check. They go unchecked because nothing forces the
control to be run and nothing forces the cost to be attributed, so both are
left as things a README says.

So a `Run` here cannot be constructed without the depth-zero attempt that would
have answered the same question without recursion, without every model call
attributed to the role and depth that issued it, and without the budget it
executed under. `recursion_verdict` is then a measurement over data the run
already holds rather than a claim somebody makes about the paradigm.

The waiver exists because a rule with no escape gets bypassed rather than
obeyed. It is deliberately expensive to use, and a waived run reports
UNTESTED forever rather than quietly reading as a success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Attempt",
    "BaselineWaiver",
    "CallRecord",
    "Outcome",
    "RecursionVerdict",
    "Role",
    "Run",
    "TokenUsage",
    "Verdict",
]


class Role(StrEnum):
    """Which position in the tree issued a call.

    Kept separate from depth because they answer different questions. Depth
    says how far down the call sat; role says whether it was driving a REPL or
    being handed a string. A depth-one call can be either, and conflating them
    is how a cost table stops being able to say where the money went.
    """

    ROOT = "root"
    SUB = "sub"


class Outcome(StrEnum):
    """How an attempt stopped.

    ANSWERED is the only outcome that produced a usable answer. The rest are
    distinguished because they fail differently: exhausting a budget is a
    controlled stop that the runtime chose, whereas an error is one it did
    not, and reporting them as one number hides how often the bound is
    actually binding.
    """

    ANSWERED = "answered"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ITERATIONS_EXHAUSTED = "iterations_exhausted"
    TIMED_OUT = "timed_out"
    ERRORED = "errored"

    @property
    def produced_answer(self) -> bool:
        return self is Outcome.ANSWERED


class Verdict(StrEnum):
    """Whether recursion earned its cost on this run."""

    HELPED = "helped"
    DID_NOT_HELP = "did_not_help"
    NOT_ATTEMPTED = "not_attempted"
    UNTESTED = "untested"
    """No depth-zero control exists, because the baseline was waived."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Tokens as the provider reported them, never as a length estimate.

    Cache reads and writes are separate fields rather than folded into input
    because they price differently and because their ratio is the single best
    diagnostic available for a recursive workload. Every sub-call in a fan-out
    shares a prefix by construction, so a cache-read count that stays at zero
    across iterations means something is invalidating that prefix, which is a
    bug that otherwise shows up only as a bill.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __post_init__(self) -> None:
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def billed_input(self) -> int:
        """Everything the provider charged for on the way in.

        Cache reads are usually discounted rather than free, so they belong in
        the input total even though they are tracked apart from it.
        """
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def total(self) -> int:
        return self.billed_input + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass(frozen=True, slots=True)
class CallRecord:
    """One model call, attributed.

    `cost_usd` is optional and stays None when the provider returned no price
    and no price table covered the model. That is deliberately not zero. A
    survey of existing implementations found several whose budget ceilings
    could never fire because unpriced calls accumulated as zero, and a cost
    ceiling that silently never triggers is worse than none at all, since it
    is believed.
    """

    role: Role
    depth: int
    model: str
    usage: TokenUsage
    wall_clock_s: float
    cost_usd: float | None = None
    cached_prefix: bool = False

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth cannot be negative")
        if self.wall_clock_s < 0:
            raise ValueError("wall_clock_s cannot be negative")
        if self.cost_usd is not None and self.cost_usd < 0:
            raise ValueError("cost_usd cannot be negative")
        if self.role is Role.ROOT and self.depth != 0:
            # A root call is the one driving the REPL at the top of a tree. A
            # child RLM drives its own REPL but is a sub-call from here, which
            # keeps "how much did the entry point cost" answerable.
            raise ValueError(f"a root call sits at depth 0, got depth {self.depth}")


@dataclass(frozen=True, slots=True)
class Attempt:
    """One bounded try at the task, at a fixed maximum depth.

    An attempt is the unit that can be compared against another attempt,
    which is why the depth bound is a property of it rather than of the run.
    The depth-zero attempt is a real attempt and not a simulation: it uses the
    same environment, the same prompt and the same parsing, with sub-calls
    unavailable. Anything less would make the comparison meaningless, because
    the difference would then include the scaffolding rather than isolating
    the recursion.
    """

    max_depth: int
    outcome: Outcome
    calls: tuple[CallRecord, ...]
    wall_clock_s: float
    answer: str | None = None
    detail: str = ""
    completion_source: str | None = None
    """How an answered attempt crossed the model/runtime boundary.

    ``rlm0_final_v1`` is the versioned envelope used for reportable evaluation.
    Older ``FINAL`` forms remain readable, but preserving their source keeps a
    compatibility path from quietly becoming a benchmark result.
    """

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        if self.wall_clock_s < 0:
            raise ValueError("wall_clock_s cannot be negative")
        if self.outcome.produced_answer and self.answer is None:
            raise ValueError("an attempt that answered must carry its answer")
        if not self.outcome.produced_answer and self.answer is not None:
            raise ValueError(
                f"attempt reports {self.outcome.value} but carries an answer; "
                "an answer that did not survive its own stop condition is not "
                "an answer"
            )
        if self.completion_source is not None and not self.outcome.produced_answer:
            raise ValueError("a completion source belongs only to an answered attempt")
        deepest = max((call.depth for call in self.calls), default=0)
        if deepest > self.max_depth:
            raise ValueError(
                f"attempt bounded at depth {self.max_depth} contains a call at "
                f"depth {deepest}; the bound was not enforced"
            )
        if self.max_depth == 0 and any(c.role is Role.SUB for c in self.calls):
            raise ValueError(
                "a depth-zero attempt cannot contain sub-calls; it is the "
                "control that the rest of the run is measured against"
            )

    @property
    def usage(self) -> TokenUsage:
        total = TokenUsage()
        for call in self.calls:
            total = total + call.usage
        return total

    @property
    def cost_usd(self) -> float | None:
        """None when any call in the attempt was unpriced.

        Deliberately not a partial sum. A total that silently omits the calls
        it could not price reads as complete and is the number a budget would
        be checked against.
        """
        if any(call.cost_usd is None for call in self.calls):
            return None
        return sum(call.cost_usd or 0.0 for call in self.calls)

    @property
    def n_sub_calls(self) -> int:
        return sum(1 for call in self.calls if call.role is Role.SUB)

    @property
    def has_versioned_completion(self) -> bool:
        """Whether this answer used the reportable completion envelope."""
        return self.completion_source == "rlm0_final_v1"

    @property
    def cache_read_ratio(self) -> float | None:
        """Cache reads over everything billed on the way in.

        None when nothing was billed. Zero across a fan-out is the signal
        described on TokenUsage.
        """
        billed = self.usage.billed_input
        if billed == 0:
            return None
        return self.usage.cache_read_tokens / billed


@dataclass(frozen=True, slots=True)
class BaselineWaiver:
    """A recorded, attributed refusal to run the depth-zero control.

    Waiving is legitimate in narrow cases, the honest one being a context so
    far past the window that depth zero cannot be attempted at all. It is not
    legitimate as a way to skip an inconvenient comparison, so the waiver
    carries who allowed it and why, and the resulting run reports UNTESTED
    rather than reading as a success.
    """

    reason: str
    approved_by: str

    def __post_init__(self) -> None:
        if not self.approved_by.strip():
            raise ValueError("a waiver must name who approved it")
        words = {w for w in self.reason.lower().split() if len(w) > 2}
        if len(words) < 6:
            raise ValueError(
                "a waiver reason must actually be a reason. Six distinct words "
                "is not a high bar, and a waiver too tedious to justify is one "
                "that should not be taken."
            )


@dataclass(frozen=True, slots=True)
class RecursionVerdict:
    """Whether recursion paid, measured rather than asserted."""

    verdict: Verdict
    baseline_answer: str | None
    escalated_answer: str | None
    extra_cost_usd: float | None
    extra_wall_clock_s: float
    extra_sub_calls: int

    def __str__(self) -> str:
        if self.verdict is Verdict.UNTESTED:
            return "recursion untested: no depth-zero control was run"
        if self.verdict is Verdict.NOT_ATTEMPTED:
            return "recursion not attempted: depth zero answered"
        cost = (
            "cost unpriced"
            if self.extra_cost_usd is None
            else f"${self.extra_cost_usd:.4f}"
        )
        return (
            f"recursion {self.verdict.value.replace('_', ' ')}: "
            f"{cost}, +{self.extra_wall_clock_s:.1f}s, "
            f"+{self.extra_sub_calls} sub-calls"
        )


@dataclass(frozen=True, slots=True)
class Run:
    """A complete attempt at one task, with its own control attached.

    Construction refuses a run whose first attempt is not the depth-zero
    control, unless a waiver is present. That ordering is not bookkeeping: the
    runtime really does try depth zero first and escalate only on failure, so
    the control is a by-product of how the work is done rather than a second
    experiment somebody has to remember to run. Most published evaluations of
    this idea are missing the control, and the reason is that producing it was
    extra work.
    """

    task: str
    attempts: tuple[Attempt, ...]
    budget_summary: str
    waiver: BaselineWaiver | None = None
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.attempts:
            raise ValueError("a run needs at least one attempt")
        if not self.budget_summary.strip():
            raise ValueError(
                "a run must record the budget it executed under; an unbounded "
                "run is the failure mode this project exists to remove"
            )
        first = self.attempts[0]
        if first.max_depth != 0 and self.waiver is None:
            raise ValueError(
                f"the first attempt is bounded at depth {first.max_depth}, so "
                "this run has no depth-zero control and cannot say whether "
                "recursion helped. Run the control, or attach a BaselineWaiver "
                "and accept an UNTESTED verdict."
            )
        if first.max_depth == 0 and self.waiver is not None:
            raise ValueError(
                "a waiver was attached to a run that has its control anyway; "
                "a waiver on file for a comparison that was made is how the "
                "next one gets skipped without anyone noticing"
            )
        depths = [attempt.max_depth for attempt in self.attempts]
        if depths != sorted(depths):
            raise ValueError(
                f"attempts must escalate, got depth bounds {depths}. Reordering "
                "them is how the cheapest attempt stops being the one that ran "
                "first."
            )

    @property
    def baseline(self) -> Attempt | None:
        """The depth-zero control, or None when it was waived."""
        first = self.attempts[0]
        return first if first.max_depth == 0 else None

    @property
    def final(self) -> Attempt:
        """The attempt whose answer the run reports."""
        for attempt in reversed(self.attempts):
            if attempt.outcome.produced_answer:
                return attempt
        return self.attempts[-1]

    @property
    def answer(self) -> str | None:
        return self.final.answer

    @property
    def calls(self) -> tuple[CallRecord, ...]:
        return tuple(call for attempt in self.attempts for call in attempt.calls)

    @property
    def usage(self) -> TokenUsage:
        total = TokenUsage()
        for attempt in self.attempts:
            total = total + attempt.usage
        return total

    @property
    def cost_usd(self) -> float | None:
        """Total across every attempt, or None when anything was unpriced."""
        if any(attempt.cost_usd is None for attempt in self.attempts):
            return None
        return sum(attempt.cost_usd or 0.0 for attempt in self.attempts)

    @property
    def wall_clock_s(self) -> float:
        return sum(attempt.wall_clock_s for attempt in self.attempts)

    def usage_by_role(self) -> dict[Role, TokenUsage]:
        """Where the tokens went.

        A recursive system that cannot separate root spend from sub spend
        cannot explain its own bill, and root spend is the part that grows
        with conversation length rather than with the work.
        """
        by_role = {Role.ROOT: TokenUsage(), Role.SUB: TokenUsage()}
        for call in self.calls:
            by_role[call.role] = by_role[call.role] + call.usage
        return by_role

    def recursion_verdict(self) -> RecursionVerdict:
        """Did recursion earn its cost here, on this task.

        Deliberately a per-run measurement and not a claim about the paradigm.
        The evidence is that the answer varies by task shape: recursion is
        worth a lot on dense aggregation and is actively harmful on retrieval
        and on anything that fits the window, so a system that answers this
        once and globally is answering the wrong question.
        """
        baseline = self.baseline
        if baseline is None:
            return RecursionVerdict(
                verdict=Verdict.UNTESTED,
                baseline_answer=None,
                escalated_answer=self.answer,
                extra_cost_usd=None,
                extra_wall_clock_s=0.0,
                extra_sub_calls=sum(a.n_sub_calls for a in self.attempts),
            )
        if len(self.attempts) == 1:
            return RecursionVerdict(
                verdict=Verdict.NOT_ATTEMPTED,
                baseline_answer=baseline.answer,
                escalated_answer=None,
                extra_cost_usd=0.0,
                extra_wall_clock_s=0.0,
                extra_sub_calls=0,
            )
        escalated = self.final
        deeper = self.attempts[1:]
        extra_cost: float | None = None
        if all(a.cost_usd is not None for a in deeper):
            extra_cost = sum(a.cost_usd or 0.0 for a in deeper)
        # Helped means the escalation produced an answer the control did not.
        # Whether that answer is correct is a question for the harness, which
        # owns the ground truth; this type owns only what the run can see.
        helped = escalated.outcome.produced_answer and not (
            baseline.outcome.produced_answer
        )
        return RecursionVerdict(
            verdict=Verdict.HELPED if helped else Verdict.DID_NOT_HELP,
            baseline_answer=baseline.answer,
            escalated_answer=escalated.answer,
            extra_cost_usd=extra_cost,
            extra_wall_clock_s=sum(a.wall_clock_s for a in deeper),
            extra_sub_calls=sum(a.n_sub_calls for a in deeper),
        )

    def describe(self) -> str:
        """One line per attempt, then the verdict.

        Printed at the end of every run, because the alternative is that the
        cost of a trajectory is discoverable only from an invoice.
        """
        lines = [f"task: {self.task}"]
        for attempt in self.attempts:
            cost = (
                "unpriced" if attempt.cost_usd is None else f"${attempt.cost_usd:.4f}"
            )
            lines.append(
                f"  depth<={attempt.max_depth}  {attempt.outcome.value:<20} "
                f"{cost:>10}  {attempt.wall_clock_s:6.1f}s  "
                f"{len(attempt.calls):3d} calls "
                f"({attempt.n_sub_calls} sub)"
            )
        lines.append(f"  budget: {self.budget_summary}")
        lines.append(f"  {self.recursion_verdict()}")
        return "\n".join(lines)
