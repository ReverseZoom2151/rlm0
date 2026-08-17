"""A bounded, artifact-only implementation of recursive agent harness plans.

Recursive Agent Harnesses make the recursive unit an entire agent environment,
not merely a model completion.  This module keeps that useful distinction
without handing generated code a host workspace, a credential, or an unlimited
tree.  A caller supplies the agent executor and a declared task tree; this
layer enforces isolation, concurrency, depth, provenance, and the shared
research record.

It does not infer a task tree from prose.  That policy belongs in a separately
evaluated planner, because making one model's planner the runtime's default
would turn a research strategy into invisible scaffolding.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from rlm0.research.artifacts import ArtifactRef, ArtifactStore
from rlm0.research.contracts import ResearchStage, ResearchTrial, run_to_dict
from rlm0.run import Run

__all__ = [
    "AgentHarnessExecutor",
    "HarnessLimits",
    "HarnessNode",
    "HarnessRequest",
    "HarnessResult",
    "run_agent_harness",
]


@dataclass(frozen=True, slots=True)
class HarnessRequest:
    """One agent task and the artifact references it may ask the host to read.

    Artifact references are capabilities, not paths.  An executor must use a
    host-serviced reader to resolve them.  This module deliberately has no
    filesystem-workspace argument, so adding one requires an explicit security
    review instead of being an incidental convenience.
    """

    node_id: str
    task: str
    context: str
    artifact_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not self.node_id.strip() or not self.task.strip():
            raise ValueError("a harness request needs an id and a task")


@dataclass(frozen=True, slots=True)
class HarnessNode:
    """A declared recursive harness tree.

    The tree is data, not code generated inside an unbounded orchestration
    shell.  It is therefore inspectable before a provider call is made and can
    be rejected before its children consume the budget.
    """

    request: HarnessRequest
    children: tuple[HarnessNode, ...] = ()


@dataclass(frozen=True, slots=True)
class HarnessLimits:
    """Hard structural limits independent of an executor's own call budget."""

    max_depth: int = 2
    max_children_per_node: int = 8
    max_nodes: int = 32
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth cannot be negative")
        for name in ("max_children_per_node", "max_nodes", "max_concurrency"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")


@runtime_checkable
class AgentHarnessExecutor(Protocol):
    """The bounded capability given to a concrete full-agent implementation.

    The executor owns its prompt, tools, and sandbox assembly.  It receives no
    credential, no host path, and no opportunity to spawn arbitrary work.  It
    returns a normal ``Run`` so every node retains the existing depth-zero
    control and per-call accounting contract.
    """

    def execute(
        self,
        request: HarnessRequest,
        *,
        inherited_artifacts: tuple[ArtifactRef, ...],
        depth: int,
    ) -> Run:
        """Run one isolated agent harness and return its accounted record."""
        ...


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """The root trial plus child records in stable tree order."""

    trial: ResearchTrial
    node_runs: tuple[tuple[str, Run], ...]

    @property
    def root_run(self) -> Run:
        return self.trial.run


def _validate_tree(node: HarnessNode, limits: HarnessLimits) -> tuple[int, set[str]]:
    """Reject a structurally unsafe plan before its first executor call."""

    seen: set[str] = set()
    count = 0

    def visit(current: HarnessNode, depth: int) -> None:
        nonlocal count
        if depth > limits.max_depth:
            raise ValueError(
                f"harness node {current.request.node_id!r} exceeds max_depth "
                f"{limits.max_depth}"
            )
        if len(current.children) > limits.max_children_per_node:
            raise ValueError(
                f"harness node {current.request.node_id!r} has "
                f"{len(current.children)} children, above "
                f"{limits.max_children_per_node}"
            )
        if current.request.node_id in seen:
            raise ValueError(f"duplicate harness node id {current.request.node_id!r}")
        seen.add(current.request.node_id)
        count += 1
        if count > limits.max_nodes:
            raise ValueError(f"harness plan exceeds max_nodes {limits.max_nodes}")
        for child in current.children:
            visit(child, depth + 1)

    visit(node, 0)
    return count, seen


def _stage(
    *,
    ordinal: int,
    node: HarnessNode,
    depth: int,
    run: Run,
    inherited: Sequence[ArtifactRef],
    produced: ArtifactRef | None,
) -> ResearchStage:
    return ResearchStage.create(
        ordinal,
        "agent_harness_node",
        {
            "node_id": node.request.node_id,
            "depth": depth,
            "n_children": len(node.children),
        },
        metadata={
            "run": run_to_dict(run),
            "input_artifacts": [reference.digest for reference in inherited],
            "output_artifact": None if produced is None else produced.digest,
        },
    )


def run_agent_harness(
    root: HarnessNode,
    *,
    executor: AgentHarnessExecutor,
    artifacts: ArtifactStore,
    limits: HarnessLimits | None = None,
    trial_id: str = "recursive-agent-harness",
) -> HarnessResult:
    """Execute a declared recursive harness plan with isolated child inputs.

    Siblings run in a bounded pool only after their parent completes.  Their
    returned records are sorted by the plan's declaration order, keeping a
    concurrent execution reproducible for reports and replay.
    """

    effective_limits = HarnessLimits() if limits is None else limits
    count, _ = _validate_tree(root, effective_limits)
    del count
    node_runs: list[tuple[str, Run]] = []
    stages: list[ResearchStage] = []

    def execute_node(
        node: HarnessNode,
        depth: int,
        inherited: tuple[ArtifactRef, ...],
    ) -> tuple[Run, ArtifactRef | None]:
        run = executor.execute(node.request, inherited_artifacts=inherited, depth=depth)
        if run.baseline is None:
            raise ValueError(
                f"harness executor returned {node.request.node_id!r} without a "
                "depth-zero control"
            )
        output = (
            None
            if run.answer is None
            else artifacts.put_text(
                run.answer, media_type="text/plain; charset=utf-8"
            )
        )
        node_runs.append((node.request.node_id, run))
        stages.append(
            _stage(
                ordinal=len(stages),
                node=node,
                depth=depth,
                run=run,
                inherited=inherited,
                produced=output,
            )
        )
        child_inherited = (*inherited, *node.request.artifact_refs)
        if output is not None:
            child_inherited = (*child_inherited, output)

        def execute_child(
            child: HarnessNode,
        ) -> tuple[str, list[tuple[str, Run]], list[ResearchStage]]:
            # Child work cannot safely mutate the parent collections while a
            # sibling is running.  Its local execution returns records for the
            # parent to append in plan order below.
            local_runs: list[tuple[str, Run]] = []
            local_stages: list[ResearchStage] = []

            def local(
                node_to_run: HarnessNode,
                local_depth: int,
                local_inherited: tuple[ArtifactRef, ...],
            ) -> None:
                child_run = executor.execute(
                    node_to_run.request,
                    inherited_artifacts=local_inherited,
                    depth=local_depth,
                )
                if child_run.baseline is None:
                    raise ValueError(
                        "harness executor returned "
                        f"{node_to_run.request.node_id!r} without a "
                        "depth-zero control"
                    )
                child_output = (
                    None
                    if child_run.answer is None
                    else artifacts.put_text(
                        child_run.answer, media_type="text/plain; charset=utf-8"
                    )
                )
                local_runs.append((node_to_run.request.node_id, child_run))
                local_stages.append(
                    _stage(
                        ordinal=0,
                        node=node_to_run,
                        depth=local_depth,
                        run=child_run,
                        inherited=local_inherited,
                        produced=child_output,
                    )
                )
                next_inherited = (*local_inherited, *node_to_run.request.artifact_refs)
                if child_output is not None:
                    next_inherited = (*next_inherited, child_output)
                for grandchild in node_to_run.children:
                    local(grandchild, local_depth + 1, next_inherited)

            local(child, depth + 1, child_inherited)
            return child.request.node_id, local_runs, local_stages

        if node.children:
            with ThreadPoolExecutor(
                max_workers=min(effective_limits.max_concurrency, len(node.children))
            ) as pool:
                completed = list(pool.map(execute_child, node.children))
            for _, child_runs, child_stages in completed:
                for child_id, child_run in child_runs:
                    node_runs.append((child_id, child_run))
                for child_stage in child_stages:
                    stages.append(
                        ResearchStage.create(
                            len(stages), child_stage.name, child_stage.config,
                            metadata=child_stage.metadata,
                        )
                    )
        return run, output

    root_run, _ = execute_node(root, 0, ())
    trial = ResearchTrial.create(
        trial_id,
        "recursive_agent_harness",
        root_run,
        stages=tuple(stages),
        config={
            "max_depth": effective_limits.max_depth,
            "max_children_per_node": effective_limits.max_children_per_node,
            "max_nodes": effective_limits.max_nodes,
            "max_concurrency": effective_limits.max_concurrency,
        },
        budget={"summary": root_run.budget_summary},
    )
    return HarnessResult(trial=trial, node_runs=tuple(node_runs))
