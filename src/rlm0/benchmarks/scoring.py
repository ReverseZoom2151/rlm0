"""The official metric, kept beside the harness metric rather than instead of it.

rlm0 grades evidence as well as answers, and that is on purpose: a right
answer from the wrong documents is luck, and the harness exists to say so. But
an evidence-aware score is not comparable to a public leaderboard, and a
number that is quietly not comparable is worse than no number, because it will
be compared anyway.

So a benchmark adapter carries two scores and never merges them. The harness
score is what `run_suite` computes, on the same grading policy as everything
else in this project. The official score is computed here, from the same
answers, by code that follows the benchmark's own scoring function including
its oddities. Each adapter declares which of the two things it is doing to the
official metric, reproducing it or approximating it, and the declaration is
carried into the manifest so nobody has to take it on trust from a README.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "Fidelity",
    "OfficialItem",
    "OfficialResult",
    "OfficialSummary",
    "Scoreboard",
]


class Fidelity(StrEnum):
    """How close a local scorer is to the published one.

    Two values and no middle. "Roughly" is the word that lets an
    incomparable number onto a slide, so an adapter either implements the
    official function including its edge cases, or it says it does not and
    explains what it changed.
    """

    REPRODUCES = "reproduces"
    """Same function, same parsing, same tie-breaking, same rounding."""

    APPROXIMATES = "approximates"
    """Differs somewhere. The difference is named in `fidelity_note`."""


@dataclass(frozen=True, slots=True)
class OfficialItem:
    """One benchmark row's gold answer, in the form the official scorer wants.

    Deliberately not the harness `Sample`. The harness stores a normalised
    answer string because it grades by exact match; the official scorers here
    need the raw field, its declared answer type, and enough of the row to
    reproduce type-dependent behaviour.
    """

    sample_id: str
    raw_answer: str
    answer_type: str
    extra: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class OfficialResult:
    """What the official scorer made of one answer.

    `parsed` and `parse_confidence` are kept because both OOLONG scorers
    return them and because an official score of zero has two very different
    causes: a wrong answer, and an answer the official parser could not find.
    Collapsing them into a single number hides a formatting bug as a capability
    result.
    """

    sample_id: str
    score: float
    parsed: str
    parse_confidence: str
    answered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "score": self.score,
            "parsed": self.parsed,
            "parse_confidence": self.parse_confidence,
            "answered": self.answered,
        }


@dataclass(frozen=True, slots=True)
class OfficialSummary:
    """The aggregate the leaderboard would print, plus the parse diagnostics."""

    metric: str
    fidelity: Fidelity
    fidelity_note: str
    n: int
    score: float
    n_unanswered: int
    n_low_confidence_parse: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "fidelity": self.fidelity.value,
            "fidelity_note": self.fidelity_note,
            "n": self.n,
            "score": self.score,
            "n_unanswered": self.n_unanswered,
            "n_low_confidence_parse": self.n_low_confidence_parse,
        }

    def describe(self) -> str:
        return (
            f"{self.metric} = {self.score:.4f} over {self.n} samples "
            f"({self.fidelity.value} the official metric; "
            f"{self.n_unanswered} unanswered, "
            f"{self.n_low_confidence_parse} low-confidence parses)"
        )


@dataclass(frozen=True, slots=True)
class Scoreboard:
    """Everything needed to compute the official number after a run.

    Held apart from the corpus so that the answer key never travels inside a
    `SolverTask`. The harness already refuses to hand a solver the object that
    holds the answer, and an adapter that reintroduced one through a side door
    would undo that.
    """

    metric: str
    fidelity: Fidelity
    fidelity_note: str
    items: Mapping[str, OfficialItem]
    scorer: Callable[[OfficialItem, str | None], OfficialResult]
    aggregate: Callable[[Sequence[OfficialResult]], float] | None = None
    """Overrides the mean, for metrics that are not a mean of per-item scores."""

    def score_all(
        self, answers: Mapping[str, str | None]
    ) -> tuple[OfficialResult, ...]:
        """Score every item the scoreboard knows about.

        An item with no answer in `answers` is scored as unanswered rather than
        skipped. Dropping it would let a run that crashed halfway report the
        mean of the samples it managed, which is the number a partial run
        should never be allowed to print.
        """
        return tuple(
            self.scorer(item, answers.get(sample_id))
            for sample_id, item in self.items.items()
        )

    def summarise(self, results: Sequence[OfficialResult]) -> OfficialSummary:
        if self.aggregate is not None:
            score = self.aggregate(results)
        elif results:
            score = sum(r.score for r in results) / len(results)
        else:
            score = 0.0
        return OfficialSummary(
            metric=self.metric,
            fidelity=self.fidelity,
            fidelity_note=self.fidelity_note,
            n=len(results),
            score=score,
            n_unanswered=sum(1 for r in results if not r.answered),
            n_low_confidence_parse=sum(
                1 for r in results if r.parse_confidence == "low"
            ),
        )
