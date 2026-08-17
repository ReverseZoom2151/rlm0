from __future__ import annotations

import pytest

from rlm0.research.contracts import ResearchStage, ResearchTrial
from rlm0.research.srlm import search
from rlm0.research.verifier import (
    CandidateEvidence,
    RecombinationError,
    Verification,
    VerifierVerdict,
    recombine,
)
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage


def _run(task: str, answer: str) -> Run:
    call = CallRecord(Role.ROOT, 0, "model", TokenUsage(), 0.0, cost_usd=0.0)
    return Run(task, (Attempt(0, Outcome.ANSWERED, (call,), 0.0, answer=answer),), "b")


def _trials() -> tuple[ResearchTrial, ...]:
    result = search(
        "question",
        lambda task, index: _run(task, "right" if index else "wrong"),
        candidates=2,
    )
    enriched: list[ResearchTrial] = []
    for trial in result.trials:
        candidate = trial.stages[0]
        stage = ResearchStage.create(
            0,
            candidate.name,
            candidate.config,
            metadata={**candidate.metadata, "cited_evidence": ["DOC-1"]},
        )
        enriched.append(
            ResearchTrial.create(
                trial.trial_id,
                trial.strategy,
                trial.run,
                stages=(stage,),
                config=trial.config,
            )
        )
    return tuple(enriched)


class _Verifier:
    def __init__(self) -> None:
        self.seen: list[CandidateEvidence] = []

    def verify(self, candidate: CandidateEvidence) -> Verification:
        self.seen.append(candidate)
        trial_id = candidate.trial_id
        answer = candidate.answer
        return Verification(
            trial_id=trial_id,
            verdict=(
                VerifierVerdict.SUPPORTED
                if answer == "right"
                else VerifierVerdict.REJECTED
            ),
        )


def test_recombination_uses_only_supported_candidates_and_no_gold_labels() -> None:
    verifier = _Verifier()
    result = recombine(_trials(), verifier)

    assert result.selection.answer == "right"
    assert result.selection.selected_trial_id == "candidate-0001"
    assert len(verifier.seen) == 2
    assert not hasattr(verifier.seen[0], "gold_answer")
    assert verifier.seen[0].cited_evidence == ("DOC-1",)


def test_missing_verifier_fails_closed() -> None:
    with pytest.raises(RecombinationError, match="requires"):
        recombine(_trials(), None)


def test_unknown_verdict_cannot_win() -> None:
    class Unknown:
        def verify(self, candidate: CandidateEvidence) -> Verification:
            return Verification(candidate.trial_id, VerifierVerdict.UNKNOWN)

    result = recombine(_trials(), Unknown())
    assert result.selection.selected_trial_id is None
    assert result.selection.answer is None


def test_mismatched_verifier_decision_fails_closed() -> None:
    class Mismatch:
        def verify(self, _candidate: CandidateEvidence) -> Verification:
            return Verification("wrong-id", VerifierVerdict.SUPPORTED)

    with pytest.raises(RecombinationError, match="does not match"):
        recombine(_trials(), Mismatch())
