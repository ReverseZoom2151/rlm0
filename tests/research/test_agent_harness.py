"""Tests for bounded, artifact-only recursive agent-harness plans."""

from __future__ import annotations

from _thread import LockType
from dataclasses import dataclass, field
from pathlib import Path
from threading import Barrier, Lock

import pytest

from rlm0.research.agent_harness import (
    HarnessLimits,
    HarnessNode,
    HarnessRequest,
    run_agent_harness,
)
from rlm0.research.artifacts import ArtifactRef, ArtifactStore
from rlm0.run import Attempt, Outcome, Run


def _run(node_id: str) -> Run:
    return Run(
        task=node_id,
        attempts=(
            Attempt(
                max_depth=0,
                outcome=Outcome.ANSWERED,
                calls=(),
                wall_clock_s=0.0,
                answer=f"answer:{node_id}",
                completion_source="rlm0_final_v1",
            ),
        ),
        budget_summary="max calls 20",
    )


@dataclass
class _Executor:
    calls: list[tuple[str, int, tuple[ArtifactRef, ...]]] = field(default_factory=list)

    def execute(
        self,
        request: HarnessRequest,
        *,
        inherited_artifacts: tuple[ArtifactRef, ...],
        depth: int,
    ) -> Run:
        self.calls.append((request.node_id, depth, inherited_artifacts))
        return _run(request.node_id)


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(
        tmp_path / "artifacts", max_total_bytes=100_000, max_artifact_bytes=10_000
    )


def test_a_harness_records_fresh_nodes_and_passes_only_artifact_capabilities(
    tmp_path: Path,
) -> None:
    executor = _Executor()
    tree = HarnessNode(
        HarnessRequest("root", "root task", "root context"),
        children=(
            HarnessNode(HarnessRequest("left", "left task", "left context")),
            HarnessNode(HarnessRequest("right", "right task", "right context")),
        ),
    )

    result = run_agent_harness(
        tree,
        executor=executor,
        artifacts=_store(tmp_path),
        limits=HarnessLimits(max_depth=1, max_concurrency=2),
    )

    assert result.root_run.answer == "answer:root"
    assert [node_id for node_id, _ in result.node_runs] == ["root", "left", "right"]
    assert [stage.ordinal for stage in result.trial.stages] == [0, 1, 2]
    assert result.trial.stages[0].metadata["run"]["task"] == "root"
    assert result.trial.stages[0].metadata["output_artifact"]
    children = {node_id: inherited for node_id, _, inherited in executor.calls[1:]}
    assert all(len(inherited) == 1 for inherited in children.values())
    root_artifact = result.trial.stages[0].metadata["output_artifact"]
    assert all(
        reference.digest == root_artifact
        for inherited in children.values()
        for reference in inherited
    )


def test_a_harness_rejects_unsafe_structure_before_the_executor_runs(
    tmp_path: Path,
) -> None:
    executor = _Executor()
    duplicate = HarnessNode(
        HarnessRequest("root", "task", "context"),
        children=(HarnessNode(HarnessRequest("root", "child", "context")),),
    )

    with pytest.raises(ValueError, match="duplicate"):
        run_agent_harness(
            duplicate,
            executor=executor,
            artifacts=_store(tmp_path),
            limits=HarnessLimits(max_depth=1),
        )
    assert executor.calls == []


def test_depth_limit_applies_to_grandchildren_before_execution(tmp_path: Path) -> None:
    executor = _Executor()
    tree = HarnessNode(
        HarnessRequest("root", "task", "context"),
        children=(
            HarnessNode(
                HarnessRequest("child", "task", "context"),
                children=(
                    HarnessNode(HarnessRequest("grandchild", "task", "context")),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="max_depth"):
        run_agent_harness(
            tree,
            executor=executor,
            artifacts=_store(tmp_path),
            limits=HarnessLimits(max_depth=1),
        )
    assert executor.calls == []


def test_nested_siblings_share_the_global_bounded_pool(tmp_path: Path) -> None:
    barrier = Barrier(2, timeout=2)

    @dataclass
    class NestedExecutor:
        active: int = 0
        maximum_active: int = 0
        lock: LockType = field(default_factory=Lock)

        def execute(
            self,
            request: HarnessRequest,
            *,
            inherited_artifacts: tuple[ArtifactRef, ...],
            depth: int,
        ) -> Run:
            del inherited_artifacts
            with self.lock:
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
            try:
                if depth == 2:
                    barrier.wait()
                return _run(request.node_id)
            finally:
                with self.lock:
                    self.active -= 1

    tree = HarnessNode(
        HarnessRequest("root", "root", "context"),
        children=(
            HarnessNode(
                HarnessRequest("parent", "parent", "context"),
                children=(
                    HarnessNode(HarnessRequest("left", "left", "context")),
                    HarnessNode(HarnessRequest("right", "right", "context")),
                ),
            ),
        ),
    )
    executor = NestedExecutor()

    result = run_agent_harness(
        tree,
        executor=executor,
        artifacts=_store(tmp_path),
        limits=HarnessLimits(max_depth=2, max_concurrency=2),
    )

    assert [node_id for node_id, _ in result.node_runs] == [
        "root",
        "parent",
        "left",
        "right",
    ]
    assert executor.maximum_active == 2
