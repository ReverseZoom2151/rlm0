from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm0.research.artifacts import ArtifactStore
from rlm0.research.chained import (
    CHAIN_PROTOCOL,
    ChainedArtifactError,
    ChainedLimitError,
    ChainedProtocolError,
    HandoffLimits,
    execute_chained,
    parse_handoff,
)
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage


def answer(
    *,
    summary: str = "found facts",
    blackboard: str = "facts: A, B",
    next_step: str = "check conflict",
    artifact_update: str = "durable evidence note",
    final: bool = False,
) -> str:
    return f"{CHAIN_PROTOCOL}(" + json.dumps(
        {
            "summary": summary,
            "blackboard": blackboard,
            "next": "" if final else next_step,
            "artifact_update": artifact_update,
            "final": final,
        },
        sort_keys=True,
    ) + ")"


def make_run(task: str, result: str) -> Run:
    call = CallRecord(
        role=Role.ROOT,
        depth=0,
        model="fake",
        usage=TokenUsage(input_tokens=2, output_tokens=2),
        wall_clock_s=0.01,
        cost_usd=0.001,
    )
    return Run(
        task=task,
        attempts=(Attempt(0, Outcome.ANSWERED, (call,), 0.01, answer=result),),
        budget_summary="max 10 calls",
    )


class FakeRoot:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    def complete(self, task: str, context: str = "") -> Run:
        self.calls.append((task, context))
        return make_run(task, self.result)


def test_fresh_roots_receive_original_task_and_context_plus_bounded_handoff(
    tmp_path: Path,
) -> None:
    first = FakeRoot(answer())
    second = FakeRoot(answer(summary="checked", final=True))
    roots = [first, second]

    execution = execute_chained(
        research_id="research-1",
        trial_id="chain-1",
        control=make_run("resolve the record", "control"),
        task="resolve the record",
        context="THE ORIGINAL CORPUS",
        fresh_root=lambda: roots.pop(0),
        artifact_store=ArtifactStore(
            tmp_path, max_total_bytes=1000, max_artifact_bytes=500
        ),
        max_roots=2,
    )

    assert len(execution.roots) == 2
    assert roots == []
    assert first.calls == [("resolve the record", "THE ORIGINAL CORPUS")]
    assert second.calls[0][0] == "resolve the record"
    assert "THE ORIGINAL CORPUS" in second.calls[0][1]
    assert execution.research.control.task == "resolve the record"
    assert execution.trial.strategy == "chained"
    assert execution.trial.stages[0].metadata["root_context_sha256"] != ""
    assert execution.trial.stages[0].metadata["original_context_sha256"] != ""
    assert execution.trial.stages[1].metadata["previous_artifact"] == {
        "sha256": execution.artifacts[0].digest,
        "size_bytes": execution.artifacts[0].size_bytes,
        "media_type": execution.artifacts[0].media_type,
    }


def test_each_factory_object_is_used_once_and_handoff_never_includes_store_path(
    tmp_path: Path,
) -> None:
    first = FakeRoot(answer(summary="s", blackboard="b", next_step="n"))
    second = FakeRoot(answer(summary="done", blackboard="b2", final=True))
    roots = iter((first, second))
    store = ArtifactStore(
        tmp_path / "private-artifacts", max_total_bytes=1000, max_artifact_bytes=500
    )

    execute_chained(
        research_id="r",
        trial_id="t",
        control=make_run("task", "control"),
        task="task",
        context="original context",
        fresh_root=lambda: next(roots),
        artifact_store=store,
        max_roots=2,
    )

    assert first.calls == [("task", "original context")]
    assert len(second.calls) == 1
    task, context = second.calls[0]
    assert task == "task"
    assert "original context" in context
    assert "SUMMARY:\ns" in context
    assert "BLACKBOARD:\nb" in context
    assert "NEXT:\nn" in context
    assert first.calls[0][1] != context
    assert str(store.root) not in context
    assert "ARTIFACT_REF:" in context


def test_handoff_needs_a_nonempty_artifact_update_even_for_final() -> None:
    malformed = answer(artifact_update="", final=True)

    with pytest.raises(ChainedProtocolError, match="artifact_update"):
        parse_handoff(malformed, limits=HandoffLimits())


@pytest.mark.parametrize(
    "payload",
    [
        "ordinary final answer",
        f"{CHAIN_PROTOCOL}([])",
        f"{CHAIN_PROTOCOL}({json.dumps({'summary': 's'})})",
        answer(next_step="", final=False),
        answer(summary="x" * 5, final=True),
    ],
)
def test_malformed_handoffs_are_refused(payload: str) -> None:
    limits = HandoffLimits(
        summary_chars=4,
        blackboard_chars=10,
        next_chars=10,
        artifact_update_chars=20,
    )

    with pytest.raises(ChainedProtocolError):
        parse_handoff(payload, limits=limits)


def test_artifact_store_limit_rejects_handoff_before_it_can_progress(
    tmp_path: Path,
) -> None:
    root = FakeRoot(answer(artifact_update="too large"))
    store = ArtifactStore(tmp_path, max_total_bytes=3, max_artifact_bytes=3)

    with pytest.raises(ChainedArtifactError, match="artifact"):
        execute_chained(
            research_id="r",
            trial_id="t",
            control=make_run("task", "control"),
            task="task",
            context="corpus",
            fresh_root=lambda: root,
            artifact_store=store,
        )

    assert root.calls


def test_stage_provenance_contains_artifact_and_each_root_run(tmp_path: Path) -> None:
    root = FakeRoot(answer(final=True))
    store = ArtifactStore(tmp_path, max_total_bytes=1000, max_artifact_bytes=500)

    execution = execute_chained(
        research_id="r",
        trial_id="t",
        control=make_run("task", "control"),
        task="task",
        context="corpus",
        fresh_root=lambda: root,
        artifact_store=store,
    )

    stage = execution.trial.stages[0]
    artifact = stage.metadata["artifact"]
    assert artifact["sha256"] == execution.artifacts[0].digest
    assert stage.metadata["root_run"]["task"] == "task"
    assert stage.metadata["handoff"]["final"] is True
    assert store.contains(execution.artifacts[0])


def test_chain_limit_is_loud_and_control_task_must_match(tmp_path: Path) -> None:
    root = FakeRoot(answer())
    store = ArtifactStore(tmp_path, max_total_bytes=1000, max_artifact_bytes=500)
    with pytest.raises(ChainedLimitError, match="max_roots"):
        execute_chained(
            research_id="r",
            trial_id="t",
            control=make_run("task", "control"),
            task="task",
            context="corpus",
            fresh_root=lambda: root,
            artifact_store=store,
            max_roots=1,
        )
    with pytest.raises(ValueError, match="exactly match"):
        execute_chained(
            research_id="r",
            trial_id="t",
            control=make_run("other", "control"),
            task="task",
            context="corpus",
            fresh_root=lambda: root,
            artifact_store=store,
        )
