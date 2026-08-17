"""Bounded depth-zero program search for optional SRLM experiments.

SRLM here means a small, reproducible population of independent depth-zero
RLM runs.  It is deliberately not a confidence vote: models may state a
confidence value in their answer, but the selector never reads it.  Agreement
comes first; equal groups are ordered by observable accounting and then by the
original candidate index.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rlm0.research.contracts import (
    ResearchStage,
    ResearchTrial,
    fingerprint,
    run_to_dict,
)
from rlm0.run import Run

__all__ = [
    "CandidateFactory",
    "CandidateSelection",
    "SRLMError",
    "SRLMResult",
    "search",
    "select_candidate",
]


class SRLMError(ValueError):
    """A candidate cannot be included in bounded depth-zero program search."""


@runtime_checkable
class CandidateFactory(Protocol):
    """Construct one fresh, already-accounted depth-zero candidate run."""

    def __call__(self, task: str, candidate_index: int) -> Run:
        """Return a completed candidate run; calls are made in index order."""
        ...


@dataclass(frozen=True, slots=True)
class CandidateSelection:
    """A selection based on answers and observable execution traces."""

    selected_trial_id: str | None
    answer: str | None
    plurality: int
    eligible_trials: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class SRLMResult:
    """The candidate trials plus the exact deterministic choice made from them."""

    trials: tuple[ResearchTrial, ...]
    selection: CandidateSelection
    selection_stage: ResearchStage

    @property
    def selected(self) -> ResearchTrial | None:
        selected_id = self.selection.selected_trial_id
        return next(
            (trial for trial in self.trials if trial.trial_id == selected_id), None
        )


def _require_depth_zero(run: Run, task: str) -> None:
    if run.task != task:
        raise SRLMError("candidate Run task differs from the requested task")
    if run.baseline is None or len(run.attempts) != 1:
        raise SRLMError("SRLM candidates must be single-attempt depth-zero Runs")


def _candidate_stage(index: int, run: Run) -> ResearchStage:
    record = run_to_dict(run)
    return ResearchStage.create(
        0,
        "candidate",
        {"candidate_index": index, "max_depth": 0},
        metadata={
            "candidate_index": index,
            "run": record,
            "run_fingerprint": fingerprint(record),
            "answer_fingerprint": None
            if run.answer is None
            else fingerprint(run.answer),
            "n_calls": len(run.calls),
            "cost_usd": run.cost_usd,
            "wall_clock_s": run.wall_clock_s,
        },
    )


def search(
    task: str,
    factory: CandidateFactory | Callable[[str, int], Run],
    *,
    candidates: int = 4,
    config: Mapping[str, Any] | None = None,
    budget: Mapping[str, Any] | None = None,
) -> SRLMResult:
    """Run a bounded, ordered depth-zero candidate population.

    Factories are invoked sequentially by index on purpose.  Parallel dispatch
    belongs below a shared budget reservation; accepting an arbitrary factory
    and launching it concurrently here would make budget ownership ambiguous.
    """
    if candidates < 1:
        raise SRLMError("candidates must be at least one")
    if not task.strip():
        raise SRLMError("task must not be empty")
    config_data = {"candidates": candidates, **dict(config or {})}
    trials: list[ResearchTrial] = []
    for index in range(candidates):
        run = factory(task, index)
        _require_depth_zero(run, task)
        trials.append(
            ResearchTrial.create(
                trial_id=f"candidate-{index:04d}",
                strategy="srlm_depth_zero",
                run=run,
                stages=(_candidate_stage(index, run),),
                config={"candidate_index": index, **config_data},
                budget=budget,
            )
        )
    frozen_trials = tuple(trials)
    selection = select_candidate(frozen_trials)
    selection_stage = ResearchStage.create(
        0,
        "plurality_selection",
        {"selector": "plurality_trace_accounting_v1"},
        metadata={
            "selected_trial_id": selection.selected_trial_id,
            "answer": selection.answer,
            "plurality": selection.plurality,
            "eligible_trials": list(selection.eligible_trials),
            "reason": selection.reason,
        },
    )
    return SRLMResult(frozen_trials, selection, selection_stage)


def _trace_key(trial: ResearchTrial, index: int) -> tuple[int, float, float, int]:
    """Order tied candidates by visible work, then their fixed input position."""
    cost = trial.run.cost_usd
    return (
        len(trial.run.calls),
        float("inf") if cost is None else cost,
        trial.run.wall_clock_s,
        index,
    )


def select_candidate(trials: Sequence[ResearchTrial]) -> CandidateSelection:
    """Select an answered candidate through plurality and accounting only."""
    answered = [
        (index, trial) for index, trial in enumerate(trials) if trial.run.answer
    ]
    if not answered:
        return CandidateSelection(None, None, 0, (), "no candidate produced an answer")
    counts = Counter(trial.run.answer for _, trial in answered)
    plurality = max(counts.values())
    winning_answers = {
        answer
        for answer, count in counts.items()
        if count == plurality and answer is not None
    }
    contenders = [
        (index, trial)
        for index, trial in answered
        if trial.run.answer in winning_answers
    ]
    _index, selected = min(
        contenders, key=lambda item: _trace_key(item[1], item[0])
    )
    assert selected.run.answer is not None
    eligible = tuple(trial.trial_id for _, trial in contenders)
    reason = (
        "plurality selected by answer agreement"
        if len(winning_answers) == 1
        else (
            "plurality tie broken by calls, priced cost, wall clock, "
            "then candidate index"
        )
    )
    return CandidateSelection(
        selected_trial_id=selected.trial_id,
        answer=selected.run.answer,
        plurality=plurality,
        eligible_trials=eligible,
        reason=reason,
    )
