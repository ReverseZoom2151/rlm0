"""Evidence-scoped verifier selection for experimental candidate recombination."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from rlm0.research.contracts import ResearchStage, ResearchTrial
from rlm0.research.srlm import CandidateSelection, select_candidate

__all__ = [
    "CandidateEvidence",
    "RecombinationError",
    "RecombinationResult",
    "Verification",
    "Verifier",
    "VerifierVerdict",
    "recombine",
]


class RecombinationError(RuntimeError):
    """No trustworthy verifier decision is available for recombination."""


class VerifierVerdict(StrEnum):
    """A verifier may support, reject, or decline to decide on a candidate."""

    SUPPORTED = "supported"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    """The candidate-visible material a verifier may inspect.

    There is intentionally no gold answer, score, or benchmark label here.
    A verifier only sees the task, proposed answer, cited evidence and the
    candidate's immutable accounting identity.
    """

    trial_id: str
    task: str
    answer: str
    cited_evidence: tuple[str, ...]
    run_fingerprint: str


@dataclass(frozen=True, slots=True)
class Verification:
    """One attributed verifier decision for a candidate."""

    trial_id: str
    verdict: VerifierVerdict
    detail: str = ""


@runtime_checkable
class Verifier(Protocol):
    """A deterministic checker with no access to hidden evaluation labels."""

    def verify(self, candidate: CandidateEvidence) -> Verification:
        """Return a decision for the supplied candidate evidence only."""
        ...


@dataclass(frozen=True, slots=True)
class RecombinationResult:
    """A supported-only selection plus the recorded verifier decisions."""

    selection: CandidateSelection
    decisions: tuple[Verification, ...]
    stage: ResearchStage


def _evidence_from(trial: ResearchTrial) -> tuple[str, ...]:
    """Read only explicit stage metadata, never a hidden harness label."""
    values: list[str] = []
    for stage in trial.stages:
        raw = stage.metadata.get("cited_evidence", ())
        if isinstance(raw, list) and all(isinstance(value, str) for value in raw):
            values.extend(raw)
    return tuple(values)


def _run_fingerprint(trial: ResearchTrial) -> str:
    for stage in trial.stages:
        raw = stage.metadata.get("run_fingerprint")
        if isinstance(raw, str):
            return raw
    raise RecombinationError(f"{trial.trial_id} has no candidate run provenance")


def recombine(
    trials: Sequence[ResearchTrial],
    verifier: Verifier | None,
) -> RecombinationResult:
    """Select only candidates a supplied verifier positively supports.

    Missing, malformed, duplicate, or mismatched decisions fail closed.  A
    ``UNKNOWN`` decision does not enter plurality, even if it is the only
    candidate with an answer.
    """
    if verifier is None:
        raise RecombinationError("recombination requires an explicit verifier")
    expected = [trial for trial in trials if trial.run.answer is not None]
    decisions: list[Verification] = []
    accepted: list[ResearchTrial] = []
    seen: set[str] = set()
    by_id = {trial.trial_id: trial for trial in expected}
    for trial in expected:
        answer = trial.run.answer
        assert answer is not None
        evidence = CandidateEvidence(
            trial_id=trial.trial_id,
            task=trial.run.task,
            answer=answer,
            cited_evidence=_evidence_from(trial),
            run_fingerprint=_run_fingerprint(trial),
        )
        decision = verifier.verify(evidence)
        if decision.trial_id != trial.trial_id:
            raise RecombinationError("verifier decision does not match its candidate")
        if decision.trial_id in seen or decision.trial_id not in by_id:
            raise RecombinationError(
                "verifier returned a duplicate or unknown candidate"
            )
        seen.add(decision.trial_id)
        decisions.append(decision)
        if decision.verdict is VerifierVerdict.SUPPORTED:
            accepted.append(trial)
    selection = select_candidate(tuple(accepted))
    stage = ResearchStage.create(
        0,
        "verifier_recombination",
        {"verifier_required": True, "selector": "supported_plurality_v1"},
        metadata={
            "selected_trial_id": selection.selected_trial_id,
            "decisions": [
                {
                    "trial_id": decision.trial_id,
                    "verdict": decision.verdict.value,
                    "detail": decision.detail,
                }
                for decision in decisions
            ],
        },
    )
    return RecombinationResult(selection, tuple(decisions), stage)
