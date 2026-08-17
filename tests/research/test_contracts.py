from __future__ import annotations

import pytest

from rlm0.research.contracts import (
    ResearchRun,
    ResearchStage,
    ResearchTrial,
    research_run_from_dict,
)
from rlm0.run import (
    Attempt,
    BaselineWaiver,
    CallRecord,
    Outcome,
    Role,
    Run,
    TokenUsage,
)


def _call(depth: int = 0) -> CallRecord:
    return CallRecord(
        role=Role.ROOT if depth == 0 else Role.SUB,
        depth=depth,
        model="test-model",
        usage=TokenUsage(input_tokens=2, output_tokens=1),
        wall_clock_s=0.1,
        cost_usd=0.001,
    )


def _control() -> Run:
    return Run(
        task="find the answer",
        attempts=(Attempt(0, Outcome.ANSWERED, (_call(),), 0.1, answer="control"),),
        budget_summary="max $1",
    )


def _trial_run() -> Run:
    return Run(
        task="find the answer",
        attempts=(
            Attempt(0, Outcome.ITERATIONS_EXHAUSTED, (_call(),), 0.1),
            Attempt(
                1,
                Outcome.ANSWERED,
                (_call(), _call(1)),
                0.2,
                answer="trial",
                completion_source="rlm0_final_v1",
            ),
        ),
        budget_summary="max $1",
    )


def test_research_record_round_trips_without_a_provider() -> None:
    stage = ResearchStage.create(0, "select", {"candidates": 4})
    trial = ResearchTrial.create(
        "candidate-0",
        "srlm",
        _trial_run(),
        stages=(stage,),
        config={"temperature": 0},
        budget={"max_calls": 20},
    )
    research = ResearchRun.create(
        "research-1",
        _control(),
        (trial,),
        config={"seed": 7},
        budget={"max_calls": 20},
    )

    rebuilt = research_run_from_dict(research.to_dict())

    assert rebuilt == research
    assert rebuilt.trials[0].run.answer == "trial"
    assert rebuilt.config == {"seed": 7}


def test_research_control_must_be_one_depth_zero_attempt() -> None:
    with pytest.raises(ValueError, match="one real depth-zero"):
        ResearchRun.create("bad", _trial_run(), ())


def test_trial_requires_a_control() -> None:
    waived = Run(
        task="t",
        attempts=(Attempt(1, Outcome.ANSWERED, (_call(1),), 0.1, answer="x"),),
        budget_summary="max $1",
        waiver=BaselineWaiver(
            reason="the context exceeds the available depth zero execution window",
            approved_by="reviewer",
        ),
    )
    with pytest.raises(ValueError, match="depth-zero control"):
        ResearchTrial.create("bad", "chained", waived)


def test_tampered_config_fingerprint_is_refused() -> None:
    trial = ResearchTrial.create(
        "trial", "srlm", _trial_run(), budget={"max_calls": 20}
    )
    research = ResearchRun.create(
        "r", _control(), (trial,), config={"seed": 1}, budget={"max_calls": 20}
    )
    payload = research.to_dict()
    payload["config"]["seed"] = 2
    with pytest.raises(ValueError, match="configuration"):
        research_run_from_dict(payload)


def test_stages_must_be_contiguous() -> None:
    stage = ResearchStage.create(1, "late", {})
    with pytest.raises(ValueError, match="contiguous"):
        ResearchTrial.create("x", "srlm", _trial_run(), stages=(stage,))


def test_research_requires_a_nonempty_paired_trial() -> None:
    with pytest.raises(ValueError, match="at least one paired trial"):
        ResearchRun.create("empty", _control(), ())


def test_research_refuses_a_trial_for_another_task() -> None:
    unrelated = Run(
        task="other task",
        attempts=(Attempt(0, Outcome.ANSWERED, (_call(),), 0.1, answer="x"),),
        budget_summary="max $1",
    )
    trial = ResearchTrial.create("other", "srlm", unrelated)
    with pytest.raises(ValueError, match="task must exactly match"):
        ResearchRun.create("r", _control(), (trial,))


def test_research_refuses_a_trial_with_a_different_budget() -> None:
    trial = ResearchTrial.create(
        "candidate", "srlm", _trial_run(), budget={"max_calls": 10}
    )
    with pytest.raises(ValueError, match="budget must exactly match"):
        ResearchRun.create(
            "r", _control(), (trial,), budget={"max_calls": 20}
        )
