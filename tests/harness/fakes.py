"""Solvers that behave badly on purpose.

A harness is only worth anything if it catches the systems that should not
pass. The two that matter are the one that answers without reading anything,
which every long-context benchmark without evidence scoring will happily rank
first, and the one that runs no control, which is the state of the published
literature. Both are built here so the tests can assert that the harness
notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rlm0.harness.corpus import Corpus, Sample, SolverTask
from rlm0.harness.runner import Attempted, SolverResult
from rlm0.run import (
    Attempt,
    BaselineWaiver,
    CallRecord,
    Outcome,
    Role,
    Run,
    TokenUsage,
)

BUDGET = "max $1.00, 60s, 20 calls"


def root_call(cost: float | None = 0.01) -> CallRecord:
    return CallRecord(
        role=Role.ROOT,
        depth=0,
        model="fake-root-model",
        usage=TokenUsage(input_tokens=1000, output_tokens=50),
        wall_clock_s=1.0,
        cost_usd=cost,
    )


def sub_call(cost: float | None = 0.005) -> CallRecord:
    return CallRecord(
        role=Role.SUB,
        depth=1,
        model="fake-sub-model",
        usage=TokenUsage(input_tokens=200, output_tokens=20, cache_read_tokens=800),
        wall_clock_s=0.5,
        cost_usd=cost,
        cached_prefix=True,
    )


def _attempt(
    depth: int, answer: str | None, calls: tuple[CallRecord, ...], wall: float
) -> Attempt:
    return Attempt(
        max_depth=depth,
        outcome=(
            Outcome.ANSWERED if answer is not None else Outcome.ITERATIONS_EXHAUSTED
        ),
        calls=calls,
        wall_clock_s=wall,
        answer=answer,
    )


@dataclass(frozen=True, slots=True)
class Plan:
    """What a scripted solver should do on one sample."""

    baseline_answer: str | None
    baseline_cites: tuple[str, ...] = ()
    escalate: bool = False
    escalated_answer: str | None = None
    escalated_cites: tuple[str, ...] = ()
    cost: float | None = 0.01
    n_sub_calls: int = 2


@dataclass
class ScriptedSolver:
    """Replays a plan per sample, producing a well-formed Run each time."""

    plans: dict[str, Plan]
    label: str = "scripted"

    def describe(self) -> str:
        return self.label

    def solve(self, task: SolverTask) -> SolverResult:
        plan = self.plans[task.sample_id]
        baseline = _attempt(0, plan.baseline_answer, (root_call(plan.cost),), 1.0)
        attempts = [baseline]
        if plan.escalate:
            calls = (
                root_call(plan.cost),
                *(sub_call(plan.cost) for _ in range(plan.n_sub_calls)),
            )
            attempts.append(_attempt(1, plan.escalated_answer, calls, 2.0))
        run = Run(task=task.question, attempts=tuple(attempts), budget_summary=BUDGET)
        final_answer = run.answer
        final_cites = (
            plan.escalated_cites
            if plan.escalate and plan.escalated_answer is not None
            else plan.baseline_cites
        )
        return SolverResult(
            run=run,
            final=Attempted(answer=final_answer, cited_doc_ids=final_cites),
            baseline=Attempted(
                answer=plan.baseline_answer, cited_doc_ids=plan.baseline_cites
            ),
        )


def perfect_plans(corpus: Corpus) -> dict[str, Plan]:
    """Depth zero answers retrieval; aggregation needs the escalation.

    Shaped after the finding this project keeps pointing at: the cheap path is
    enough for retrieval and recursion is what earns its cost on dense
    aggregation.
    """
    plans: dict[str, Plan] = {}
    for sample in corpus.samples:
        evidence = tuple(sorted(sample.required_doc_ids))
        if sample.regime.value == "retrieval":
            plans[sample.sample_id] = Plan(
                baseline_answer=sample.answer, baseline_cites=evidence
            )
        else:
            plans[sample.sample_id] = Plan(
                baseline_answer=None,
                escalate=True,
                escalated_answer=sample.answer,
                escalated_cites=evidence,
            )
    return plans


@dataclass
class AlwaysWrongSolver:
    """Answers, confidently, with something that is never right.

    The floor of the benchmark. A harness that gives this anything above zero
    is measuring its own leniency.
    """

    answer: str = "0"
    label: str = "always-wrong"

    def describe(self) -> str:
        return self.label

    def solve(self, task: SolverTask) -> SolverResult:
        baseline = _attempt(0, self.answer, (root_call(),), 1.0)
        run = Run(task=task.question, attempts=(baseline,), budget_summary=BUDGET)
        return SolverResult(
            run=run,
            final=Attempted(answer=self.answer, cited_doc_ids=()),
            baseline=Attempted(answer=self.answer, cited_doc_ids=()),
        )


@dataclass
class CheatingSolver:
    """Returns the right answer having read nothing.

    Built with an answer key handed to it from outside, because the runner
    will not give it one: a solver sees a SolverTask, which carries the
    question and the documents and nothing else. It still scores perfectly on
    answer accuracy, which is exactly why answer accuracy on its own is not a
    result.
    """

    answers: dict[str, str]
    label: str = "oracle-lookup"

    def describe(self) -> str:
        return self.label

    def solve(self, task: SolverTask) -> SolverResult:
        answer = self.answers[task.sample_id]
        attempt = Attempt(
            max_depth=0,
            outcome=Outcome.ANSWERED,
            calls=(),
            wall_clock_s=0.0,
            answer=answer,
        )
        run = Run(task=task.question, attempts=(attempt,), budget_summary=BUDGET)
        return SolverResult(
            run=run,
            final=Attempted(answer=answer, cited_doc_ids=()),
            baseline=Attempted(answer=answer, cited_doc_ids=()),
        )


@dataclass
class WaivingSolver:
    """Skips the control, which is what the field does.

    Everything it produces is well formed. The report still refuses it.
    """

    answers: dict[str, str]
    label: str = "no-control"
    waiver_reason: str = (
        "the context exceeds every available model window by several times"
    )

    def describe(self) -> str:
        return self.label

    def solve(self, task: SolverTask) -> SolverResult:
        answer = self.answers[task.sample_id]
        attempt = _attempt(1, answer, (root_call(), sub_call()), 2.0)
        run = Run(
            task=task.question,
            attempts=(attempt,),
            budget_summary=BUDGET,
            waiver=BaselineWaiver(reason=self.waiver_reason, approved_by="tests"),
        )
        return SolverResult(
            run=run,
            final=Attempted(answer=answer, cited_doc_ids=()),
            baseline=None,
        )


@dataclass
class BrokenAccountingSolver:
    """Reports an answer that no attempt in its own run produced."""

    label: str = "broken-accounting"
    cited: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        return self.label

    def solve(self, task: SolverTask) -> SolverResult:
        baseline = _attempt(0, None, (root_call(),), 1.0)
        run = Run(task=task.question, attempts=(baseline,), budget_summary=BUDGET)
        return SolverResult(
            run=run,
            final=Attempted(answer="invented", cited_doc_ids=self.cited),
            baseline=Attempted(answer=None),
        )


def answer_key(corpus: Corpus) -> dict[str, str]:
    return {sample.sample_id: sample.answer for sample in corpus.samples}


def sample_of(corpus: Corpus, family: str) -> Sample:
    for sample in corpus.samples:
        if sample.family.value == family:
            return sample
    raise KeyError(family)
