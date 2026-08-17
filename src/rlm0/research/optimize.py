"""Development-only prompt search with split and regression guardrails.

This module generates immutable *candidate* prompts.  It has no import of the
runtime prompt, no file-writing API, and no "apply" operation.  Selecting a
candidate for a shipped runtime is a separate, human-reviewed release change.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from random import Random

__all__ = [
    "CandidateEvaluation",
    "EvaluationBudget",
    "EvolutionConfig",
    "MetricEstimate",
    "OptimizationError",
    "OptimizationReport",
    "ParetoFrontier",
    "PromptCandidate",
    "PromptSections",
    "RegressionGate",
    "Split",
    "SplitGuard",
    "evolve",
    "pareto_frontier",
]


class OptimizationError(ValueError):
    """An optimization request would violate the development-only contract."""


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SplitGuard:
    """Names disjoint dataset bytes and forbids test-set prompt selection."""

    train_fingerprint: str
    validation_fingerprint: str
    test_fingerprint: str

    def __post_init__(self) -> None:
        values = (
            self.train_fingerprint,
            self.validation_fingerprint,
            self.test_fingerprint,
        )
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in values
        ):
            raise OptimizationError(
                "split fingerprints must be lowercase SHA-256 values"
            )
        if len(set(values)) != len(values):
            raise OptimizationError(
                "train, validation, and test splits must be disjoint"
            )

    def fingerprint_for(self, split: Split) -> str:
        return {
            Split.TRAIN: self.train_fingerprint,
            Split.VALIDATION: self.validation_fingerprint,
            Split.TEST: self.test_fingerprint,
        }[split]

    def allow_selection(self, split: Split, fingerprint: str) -> None:
        if split is not Split.VALIDATION:
            raise OptimizationError(
                "prompt selection may use validation data only, never train or test"
            )
        if fingerprint != self.validation_fingerprint:
            raise OptimizationError(
                "evaluation bytes do not match the pinned validation split"
            )


@dataclass(frozen=True, slots=True)
class PromptSections:
    """Typed prompt parts, kept separate so mutations have legible provenance."""

    system: str
    method: str
    output: str

    def __post_init__(self) -> None:
        if any(not part.strip() for part in (self.system, self.method, self.output)):
            raise OptimizationError(
                "every prompt section must contain non-whitespace text"
            )

    def to_dict(self) -> dict[str, str]:
        return {"system": self.system, "method": self.method, "output": self.output}


@dataclass(frozen=True, slots=True)
class PromptCandidate:
    """A content-addressed prompt proposal, never a mutable runtime setting."""

    sections: PromptSections
    lineage: tuple[str, ...] = ()
    operation: str = "seed"

    @property
    def fingerprint(self) -> str:
        return _fingerprint({"sections": self.sections.to_dict()})

    def render(self) -> str:
        return "\n\n".join(
            (self.sections.system, self.sections.method, self.sections.output)
        )


@dataclass(frozen=True, slots=True)
class MetricEstimate:
    """A metric mean plus enough paired uncertainty to gate a change."""

    mean: float
    standard_error: float
    n: int

    def __post_init__(self) -> None:
        if not self.n or self.standard_error < 0:
            raise OptimizationError(
                "metrics need a positive sample count and nonnegative error"
            )


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """One bounded validation evaluation, tied to prompt and data identities."""

    candidate_fingerprint: str
    split: Split
    split_fingerprint: str
    metrics: Mapping[str, MetricEstimate]
    cost_usd: float

    def __post_init__(self) -> None:
        if not self.metrics or self.cost_usd < 0:
            raise OptimizationError("evaluation needs metrics and a nonnegative cost")


@dataclass(frozen=True, slots=True)
class EvaluationBudget:
    max_candidates: int
    max_cost_usd: float

    def __post_init__(self) -> None:
        if self.max_candidates < 1 or self.max_cost_usd < 0:
            raise OptimizationError(
                "evaluation budget must allow candidates and nonnegative cost"
            )

    def check(self, evaluations: Sequence[CandidateEvaluation]) -> None:
        if len(evaluations) > self.max_candidates:
            raise OptimizationError(
                "candidate evaluation count exceeds the declared budget"
            )
        if sum(item.cost_usd for item in evaluations) > self.max_cost_usd:
            raise OptimizationError(
                "candidate evaluation cost exceeds the declared budget"
            )


@dataclass(frozen=True, slots=True)
class RegressionGate:
    """Reject a candidate if any field regresses or lacks a credible gain."""

    fields: tuple[str, ...]
    max_regression: float = 0.0
    min_improvement: float = 0.0
    z_score: float = 1.96

    def __post_init__(self) -> None:
        if not self.fields or self.max_regression < 0 or self.min_improvement < 0:
            raise OptimizationError("gate needs fields and nonnegative thresholds")

    def approve(
        self, incumbent: CandidateEvaluation, proposal: CandidateEvaluation
    ) -> bool:
        if incumbent.split_fingerprint != proposal.split_fingerprint:
            raise OptimizationError(
                "cannot compare evaluations from different validation bytes"
            )
        improvements: list[float] = []
        for field in self.fields:
            old, new = incumbent.metrics.get(field), proposal.metrics.get(field)
            if old is None or new is None:
                raise OptimizationError(f"regression gate field {field!r} is missing")
            delta = new.mean - old.mean
            uncertainty = (
                self.z_score * (old.standard_error**2 + new.standard_error**2) ** 0.5
            )
            lower_bound = delta - uncertainty
            if lower_bound < -self.max_regression:
                return False
            improvements.append(lower_bound)
        return any(delta >= self.min_improvement for delta in improvements)


def _dominates(left: CandidateEvaluation, right: CandidateEvaluation) -> bool:
    if set(left.metrics) != set(right.metrics):
        raise OptimizationError("Pareto comparisons require identical metric fields")
    qualities = [
        left.metrics[key].mean >= right.metrics[key].mean for key in left.metrics
    ]
    strict_quality = [
        left.metrics[key].mean > right.metrics[key].mean for key in left.metrics
    ]
    return (
        all(qualities)
        and left.cost_usd <= right.cost_usd
        and (any(strict_quality) or left.cost_usd < right.cost_usd)
    )


def pareto_frontier(
    evaluations: Sequence[CandidateEvaluation],
) -> tuple[CandidateEvaluation, ...]:
    """Return stable nondominated validation candidates."""
    return tuple(
        item
        for item in evaluations
        if not any(other != item and _dominates(other, item) for other in evaluations)
    )


@dataclass(frozen=True, slots=True)
class ParetoFrontier:
    evaluations: tuple[CandidateEvaluation, ...]

    @classmethod
    def from_evaluations(cls, values: Sequence[CandidateEvaluation]) -> ParetoFrontier:
        return cls(pareto_frontier(values))


Evaluator = Callable[[PromptCandidate, Split, str], CandidateEvaluation]
Mutator = Callable[[PromptSections, Random], PromptSections]


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    """The review artifact. ``recommended`` is deliberately not applied."""

    candidates: tuple[PromptCandidate, ...]
    evaluations: tuple[CandidateEvaluation, ...]
    frontier: ParetoFrontier
    recommended: str | None
    split_fingerprint: str


def _evaluate(
    candidates: Sequence[PromptCandidate],
    evaluator: Evaluator,
    guard: SplitGuard,
    budget: EvaluationBudget,
    gate: RegressionGate,
    incumbent: CandidateEvaluation | None,
) -> OptimizationReport:
    unique = {candidate.fingerprint: candidate for candidate in candidates}
    if len(unique) > budget.max_candidates:
        raise OptimizationError(
            "candidate generation exceeds the declared evaluation budget"
        )
    evaluations: list[CandidateEvaluation] = []
    approved: list[CandidateEvaluation] = []
    for candidate in unique.values():
        result = evaluator(candidate, Split.VALIDATION, guard.validation_fingerprint)
        if result.candidate_fingerprint != candidate.fingerprint:
            raise OptimizationError("evaluator returned a result for another candidate")
        guard.allow_selection(result.split, result.split_fingerprint)
        evaluations.append(result)
        if incumbent is None or gate.approve(incumbent, result):
            approved.append(result)
    budget.check(evaluations)
    frontier = ParetoFrontier.from_evaluations(approved)
    recommended = (
        min(frontier.evaluations, key=lambda item: item.cost_usd).candidate_fingerprint
        if frontier.evaluations
        else None
    )
    return OptimizationReport(
        tuple(unique.values()),
        tuple(evaluations),
        frontier,
        recommended,
        guard.validation_fingerprint,
    )


@dataclass(frozen=True, slots=True)
class EvolutionConfig:
    seed: int = 0
    population_size: int = 4
    generations: int = 1

    def __post_init__(self) -> None:
        if self.population_size < 2 or self.generations < 1:
            raise OptimizationError(
                "evolution needs at least two parents and one generation"
            )


def _crossover(
    left: PromptCandidate, right: PromptCandidate, random: Random
) -> PromptCandidate:
    a, b = left.sections, right.sections
    sections = PromptSections(
        system=a.system if random.randrange(2) else b.system,
        method=a.method if random.randrange(2) else b.method,
        output=a.output if random.randrange(2) else b.output,
    )
    return PromptCandidate(sections, (left.fingerprint, right.fingerprint), "crossover")


def evolve(
    seeds: Sequence[PromptCandidate],
    *,
    mutate: Mutator,
    evaluator: Evaluator,
    guard: SplitGuard,
    budget: EvaluationBudget,
    gate: RegressionGate,
    config: EvolutionConfig | None = None,
    incumbent: CandidateEvaluation | None = None,
) -> OptimizationReport:
    """Run deterministic mutation/crossover search on validation only."""
    config = EvolutionConfig() if config is None else config
    if len(seeds) < 2:
        raise OptimizationError("evolution needs at least two seed candidates")
    random = Random(config.seed)
    population = list(seeds[: config.population_size])
    for _ in range(config.generations):
        children: list[PromptCandidate] = []
        for index, parent in enumerate(population):
            partner = population[(index + 1) % len(population)]
            crossed = _crossover(parent, partner, random)
            mutated = mutate(crossed.sections, random)
            children.append(PromptCandidate(mutated, crossed.lineage, "mutation"))
        population.extend(children)
        population = list({item.fingerprint: item for item in population}.values())[
            : budget.max_candidates
        ]
    return _evaluate(population, evaluator, guard, budget, gate, incumbent)
