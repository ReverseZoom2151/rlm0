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
from threading import BoundedSemaphore
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


@dataclass(frozen=True, slots=True)
class _NodeExecution:
    """One completed node plus its descendants in declared preorder."""

    node: HarnessNode
    depth: int
    inherited: tuple[ArtifactRef, ...]
    run: Run
    output: ArtifactRef | None
    descendants: tuple[_NodeExecution, ...]


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
    execution_slots = BoundedSemaphore(effective_limits.max_concurrency)

    def execute_node(
        node: HarnessNode, depth: int, inherited: tuple[ArtifactRef, ...]
    ) -> _NodeExecution:
        # Nested pools let every sibling set fan out. The one shared semaphore,
        # rather than a separate pool size at each depth, is the global bound on
        # simultaneous concrete agent runs.
        with execution_slots:
            run = executor.execute(
                node.request, inherited_artifacts=inherited, depth=depth
            )
        if run.baseline is None:
            raise ValueError(
                f"harness executor returned {node.request.node_id!r} without a "
                "depth-zero control"
            )
        output = (
            None
            if run.answer is None
            else artifacts.put_text(run.answer, media_type="text/plain; charset=utf-8")
        )
        child_inherited = (*inherited, *node.request.artifact_refs)
        if output is not None:
            child_inherited = (*child_inherited, output)
        descendants: tuple[_NodeExecution, ...] = ()
        if node.children:
            with ThreadPoolExecutor(
                max_workers=min(effective_limits.max_concurrency, len(node.children))
            ) as pool:
                descendants = tuple(
                    pool.map(
                        lambda child: execute_node(child, depth + 1, child_inherited),
                        node.children,
                    )
                )
        return _NodeExecution(node, depth, inherited, run, output, descendants)

    def flatten(execution: _NodeExecution) -> tuple[_NodeExecution, ...]:
        return (
            execution,
            *(nested for child in execution.descendants for nested in flatten(child)),
        )

    root_execution = execute_node(root, 0, ())
    ordered = flatten(root_execution)
    node_runs = tuple((item.node.request.node_id, item.run) for item in ordered)
    stages = tuple(
        _stage(
            ordinal=ordinal,
            node=item.node,
            depth=item.depth,
            run=item.run,
            inherited=item.inherited,
            produced=item.output,
        )
        for ordinal, item in enumerate(ordered)
    )
    root_run = root_execution.run
    trial = ResearchTrial.create(
        trial_id,
        "recursive_agent_harness",
        root_run,
        stages=stages,
        config={
            "max_depth": effective_limits.max_depth,
            "max_children_per_node": effective_limits.max_children_per_node,
            "max_nodes": effective_limits.max_nodes,
            "max_concurrency": effective_limits.max_concurrency,
        },
        budget={"summary": root_run.budget_summary},
    )
    return HarnessResult(trial=trial, node_runs=node_runs)
