from __future__ import annotations

import pytest

from rlm0.research.srlm import SRLMError, search
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage


def _run(task: str, answer: str | None, *, calls: int = 1, cost: float = 0.1) -> Run:
    records = tuple(
        CallRecord(
            role=Role.ROOT,
            depth=0,
            model="model",
            usage=TokenUsage(),
            wall_clock_s=0.1,
            cost_usd=cost,
        )
        for _ in range(calls)
    )
    outcome = Outcome.ANSWERED if answer is not None else Outcome.ITERATIONS_EXHAUSTED
    return Run(
        task, (Attempt(0, outcome, records, calls / 10, answer=answer),), "budget"
    )


def test_search_preserves_factory_order_and_candidate_provenance() -> None:
    seen: list[int] = []

    def factory(task: str, index: int) -> Run:
        seen.append(index)
        return _run(task, "same")

    result = search("task", factory, candidates=3, config={"seed": 4})

    assert seen == [0, 1, 2]
    assert result.selection.answer == "same"
    assert result.selection.plurality == 3
    assert result.selected is result.trials[0]
    metadata = result.trials[1].stages[0].metadata
    assert metadata["candidate_index"] == 1
    assert metadata["run"]["task"] == "task"
    assert len(metadata["run_fingerprint"]) == 64


def test_plurality_beats_self_reported_confidence_text() -> None:
    answers = ["wrong confidence=100%", "right confidence=0%", "right confidence=0%"]

    result = search(
        "task", lambda task, index: _run(task, answers[index]), candidates=3
    )

    assert result.selection.answer == "right confidence=0%"
    assert result.selection.selected_trial_id == "candidate-0001"


def test_answer_tie_uses_trace_then_accounting_then_input_order() -> None:
    runs = [
        _run("task", "a", calls=2, cost=0.01),
        _run("task", "b", calls=1, cost=0.99),
        _run("task", "a", calls=2, cost=0.01),
        _run("task", "b", calls=1, cost=0.99),
    ]
    result = search("task", lambda _task, index: runs[index], candidates=4)

    assert result.selection.answer == "b"
    assert result.selection.reason.startswith("plurality tie")
    assert result.selection.selected_trial_id == "candidate-0001"


def test_rejects_candidate_that_is_not_a_single_depth_zero_run() -> None:
    deep = Run(
        "task",
        (
            Attempt(0, Outcome.ITERATIONS_EXHAUSTED, (), 0.0),
            Attempt(1, Outcome.ANSWERED, (), 0.0, answer="bad"),
        ),
        "budget",
    )
    with pytest.raises(SRLMError, match="single-attempt"):
        search("task", lambda _task, _index: deep)


def test_requires_a_positive_candidate_bound() -> None:
    with pytest.raises(SRLMError, match="at least one"):
        search("task", lambda task, _index: _run(task, "x"), candidates=0)
