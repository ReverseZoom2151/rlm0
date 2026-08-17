from __future__ import annotations

from random import Random

import pytest

from rlm0.research.optimize import (
    CandidateEvaluation,
    EvaluationBudget,
    EvolutionConfig,
    MetricEstimate,
    OptimizationError,
    PromptCandidate,
    PromptSections,
    RegressionGate,
    Split,
    SplitGuard,
    evolve,
    pareto_frontier,
)


def _guard() -> SplitGuard:
    return SplitGuard("a" * 64, "b" * 64, "c" * 64)


def _candidate(method: str = "method") -> PromptCandidate:
    return PromptCandidate(PromptSections("system", method, "output"))


def _evaluation(
    candidate: PromptCandidate, score: float, cost: float = 1.0
) -> CandidateEvaluation:
    return CandidateEvaluation(
        candidate.fingerprint,
        Split.VALIDATION,
        "b" * 64,
        {
            "accuracy": MetricEstimate(score, 0.001, 20),
            "evidence": MetricEstimate(score, 0.001, 20),
        },
        cost,
    )


def test_split_guard_refuses_overlap_train_selection_and_test_data() -> None:
    with pytest.raises(OptimizationError, match="disjoint"):
        SplitGuard("a" * 64, "a" * 64, "c" * 64)
    guard = _guard()
    with pytest.raises(OptimizationError, match="validation"):
        guard.allow_selection(Split.TEST, "c" * 64)
    with pytest.raises(OptimizationError, match="pinned validation"):
        guard.allow_selection(Split.VALIDATION, "a" * 64)


def test_candidate_is_stable_content_addressed_and_has_no_apply_operation() -> None:
    first, second = _candidate(), _candidate()
    assert first.fingerprint == second.fingerprint
    assert "apply" not in dir(first)
    assert "method" in first.render()


def test_gate_rejects_any_regression_and_needs_significant_improvement() -> None:
    baseline = _evaluation(_candidate("base"), 0.8)
    gate = RegressionGate(
        ("accuracy", "evidence"), max_regression=0.01, min_improvement=0.01
    )
    assert gate.approve(baseline, _evaluation(_candidate("better"), 0.83))
    assert not gate.approve(baseline, _evaluation(_candidate("worse"), 0.75))
    assert not gate.approve(baseline, _evaluation(_candidate("noise"), 0.805))


def test_pareto_keeps_cost_quality_tradeoff_and_drops_dominated() -> None:
    fast = _evaluation(_candidate("fast"), 0.8, 0.1)
    accurate = _evaluation(_candidate("accurate"), 0.9, 1.0)
    dominated = _evaluation(_candidate("dominated"), 0.7, 1.5)
    result = pareto_frontier((fast, accurate, dominated))
    assert {item.candidate_fingerprint for item in result} == {
        fast.candidate_fingerprint,
        accurate.candidate_fingerprint,
    }


def test_evolution_is_deterministic_bounded_and_validation_only() -> None:
    calls: list[tuple[Split, str]] = []

    def evaluator(
        candidate: PromptCandidate, split: Split, fingerprint: str
    ) -> CandidateEvaluation:
        calls.append((split, fingerprint))
        score = 0.9 if "+" in candidate.sections.method else 0.8
        return CandidateEvaluation(
            candidate.fingerprint,
            split,
            fingerprint,
            {"accuracy": MetricEstimate(score, 0.001, 30)},
            0.1,
        )

    def mutate(sections: PromptSections, _: Random) -> PromptSections:
        return PromptSections(sections.system, sections.method + "+", sections.output)

    report = evolve(
        (_candidate("one"), _candidate("two")),
        mutate=mutate,
        evaluator=evaluator,
        guard=_guard(),
        budget=EvaluationBudget(4, 1.0),
        gate=RegressionGate(("accuracy",), min_improvement=0.0),
        config=EvolutionConfig(seed=7, population_size=2, generations=1),
    )
    assert len(report.evaluations) == 4
    assert report.recommended is not None
    assert calls and all(call == (Split.VALIDATION, "b" * 64) for call in calls)


def test_evaluator_cannot_lie_about_candidate_data_or_budget() -> None:
    seed = _candidate()

    def wrong_candidate(
        _: PromptCandidate, split: Split, fingerprint: str
    ) -> CandidateEvaluation:
        return CandidateEvaluation(
            "f" * 64,
            split,
            fingerprint,
            {"accuracy": MetricEstimate(1.0, 0.0, 1)},
            0.0,
        )

    with pytest.raises(OptimizationError, match="another candidate"):
        evolve(
            (seed, _candidate("second")),
            mutate=lambda sections, _: sections,
            evaluator=wrong_candidate,
            guard=_guard(),
            budget=EvaluationBudget(2, 1.0),
            gate=RegressionGate(("accuracy",)),
        )
