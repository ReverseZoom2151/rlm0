"""Grade the answer and the evidence separately, and never blend them away.

A system that returns the right answer having read the wrong documents got
lucky. On a long-context benchmark that is not a rare event: the answer space
is small, the distractors are near misses of it, and a model that guesses the
shape of the answer from the question will sometimes land on it. Reporting
only answer accuracy makes that indistinguishable from a solve, which is
exactly the number every surveyed implementation would have published if any
of them had published one.

So two numbers, always, and a third that is only awarded when both hold. The
evidence score is set retrieval against the documents the corpus says had to
be read, with precision included so that citing the whole context does not
buy recall for free.

Matching is exact after normalisation, deliberately. The answers here are
opaque identifiers and integers, so there is nothing for a fuzzy match to
usefully forgive, and partial credit for a near miss would reward precisely
the failure mode the corpus was built to punish. Finding something plausible
scores zero.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rlm0.harness.corpus import Regime, Sample, TaskFamily

__all__ = [
    "GradingPolicy",
    "SampleScore",
    "ScoreSummary",
    "grade",
    "summarise",
]

_PUNCT = re.compile(r"[\s,]+")


@dataclass(frozen=True, slots=True)
class GradingPolicy:
    """How strict the evidence requirement is, recorded with every result.

    Carried into the report and compared across rows, because two accuracy
    figures computed under different policies are not a comparison and a table
    that puts them side by side is making a claim it cannot support.
    """

    require_complete_evidence: bool = True
    """A supported solve must cite every document the corpus required."""

    min_evidence_precision: float = 0.5
    """Guards against citing the entire context to guarantee recall."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_evidence_precision <= 1.0:
            raise ValueError("min_evidence_precision must lie in [0, 1]")

    def describe(self) -> str:
        completeness = "complete" if self.require_complete_evidence else "partial"
        return (
            f"evidence {completeness}, precision >= "
            f"{self.min_evidence_precision:.2f}"
        )


def normalise(answer: str) -> str:
    """Fold away formatting that no one would call a different answer.

    Case, surrounding punctuation, thousands separators and a trailing full
    stop. Nothing else. Anything more forgiving starts giving credit for
    being close, and being close is the failure this corpus is designed to
    produce on purpose.
    """
    text = answer.strip()
    for _ in range(3):
        # Quoting and a trailing full stop can nest either way round, and one
        # pass would leave whichever was outermost behind.
        text = text.strip().strip("\"'").rstrip(".")
    text = _PUNCT.sub(" ", text)
    return text.strip().lower()


def _same_answer(given: str, expected: str) -> bool:
    left, right = normalise(given), normalise(expected)
    if left == right:
        return True
    # An integer answer written as "7." or "7,0" has already been folded; this
    # only catches a numeric answer returned as a float, such as "7.0".
    try:
        return float(left) == float(right)
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class SampleScore:
    """One sample, scored on both axes with the composite kept derived.

    `score` is answer correctness gated by evidence quality rather than an
    average of the two, so that a wrong answer can never be lifted by good
    citations and a right answer with no support cannot reach the top.
    """

    sample_id: str
    family: TaskFamily
    answered: bool
    answer_correct: bool
    matched_distractor: bool
    evidence_precision: float
    evidence_recall: float
    evidence_f1: float
    evidence_exact: bool
    n_cited: int
    n_required: int
    supported: bool

    @property
    def regime(self) -> Regime:
        return self.family.regime

    @property
    def score(self) -> float:
        """Zero unless the answer is right; then weighted by the evidence.

        The ordering this guarantees is the one the project cares about: a
        right answer citing the right documents beats a right answer citing
        the wrong ones, which beats nothing at all.
        """
        if not self.answer_correct:
            return 0.0
        return self.evidence_f1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "family": self.family.value,
            "answered": self.answered,
            "answer_correct": self.answer_correct,
            "matched_distractor": self.matched_distractor,
            "evidence_precision": self.evidence_precision,
            "evidence_recall": self.evidence_recall,
            "evidence_f1": self.evidence_f1,
            "evidence_exact": self.evidence_exact,
            "n_cited": self.n_cited,
            "n_required": self.n_required,
            "supported": self.supported,
        }


def score_from_dict(payload: dict[str, Any]) -> SampleScore:
    return SampleScore(
        sample_id=str(payload["sample_id"]),
        family=TaskFamily(payload["family"]),
        answered=bool(payload["answered"]),
        answer_correct=bool(payload["answer_correct"]),
        matched_distractor=bool(payload["matched_distractor"]),
        evidence_precision=float(payload["evidence_precision"]),
        evidence_recall=float(payload["evidence_recall"]),
        evidence_f1=float(payload["evidence_f1"]),
        evidence_exact=bool(payload["evidence_exact"]),
        n_cited=int(payload["n_cited"]),
        n_required=int(payload["n_required"]),
        supported=bool(payload["supported"]),
    )


def grade(
    sample: Sample,
    answer: str | None,
    cited_doc_ids: Iterable[str],
    *,
    policy: GradingPolicy | None = None,
) -> SampleScore:
    """Score one attempt at one sample."""
    policy = policy or GradingPolicy()
    cited = {doc_id for doc_id in cited_doc_ids if doc_id}
    required = set(sample.required_doc_ids)
    hits = cited & required

    precision = len(hits) / len(cited) if cited else 0.0
    recall = len(hits) / len(required) if required else 1.0
    f1 = (
        0.0
        if precision + recall == 0.0
        else 2 * precision * recall / (precision + recall)
    )

    correct = answer is not None and _same_answer(answer, sample.answer)
    plausible = answer is not None and not correct and any(
        _same_answer(answer, distractor) for distractor in sample.distractor_answers
    )

    complete = recall >= 1.0 if policy.require_complete_evidence else recall > 0.0
    supported = (
        correct and complete and precision >= policy.min_evidence_precision
    )

    return SampleScore(
        sample_id=sample.sample_id,
        family=sample.family,
        answered=answer is not None,
        answer_correct=correct,
        matched_distractor=plausible,
        evidence_precision=precision,
        evidence_recall=recall,
        evidence_f1=f1,
        evidence_exact=cited == required,
        n_cited=len(cited),
        n_required=len(required),
        supported=supported,
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    """Aggregates over a set of scores, with the two axes kept apart.

    `answer_accuracy` is the number everyone else publishes and is included
    only so that the gap between it and `supported_accuracy` is visible. That
    gap is the amount of the headline that came from luck.
    """

    n: int
    answer_accuracy: float
    supported_accuracy: float
    evidence_recall: float
    evidence_precision: float
    evidence_f1: float
    mean_score: float
    n_unanswered: int
    n_plausible_wrong: int

    @property
    def luck_gap(self) -> float:
        return self.answer_accuracy - self.supported_accuracy

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "answer_accuracy": self.answer_accuracy,
            "supported_accuracy": self.supported_accuracy,
            "evidence_recall": self.evidence_recall,
            "evidence_precision": self.evidence_precision,
            "evidence_f1": self.evidence_f1,
            "mean_score": self.mean_score,
            "n_unanswered": self.n_unanswered,
            "n_plausible_wrong": self.n_plausible_wrong,
        }


def summarise(scores: Sequence[SampleScore]) -> ScoreSummary:
    return ScoreSummary(
        n=len(scores),
        answer_accuracy=_mean([float(s.answer_correct) for s in scores]),
        supported_accuracy=_mean([float(s.supported) for s in scores]),
        evidence_recall=_mean([s.evidence_recall for s in scores]),
        evidence_precision=_mean([s.evidence_precision for s in scores]),
        evidence_f1=_mean([s.evidence_f1 for s in scores]),
        mean_score=_mean([s.score for s in scores]),
        n_unanswered=sum(1 for s in scores if not s.answered),
        n_plausible_wrong=sum(1 for s in scores if s.matched_distractor),
    )
