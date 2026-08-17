"""Fresh-root chained RLM research strategy.

Each root gets a new injected RLM instance. The chain never shares a REPL,
conversation, host directory, API key, or sandbox mount. It carries only the
immutable original task/context plus a bounded, host-validated handoff. Every
accepted handoff first writes an artifact through :class:`ArtifactStore` and
records both its content address and the root run that produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Protocol

from rlm0.research.artifacts import ArtifactLimitError, ArtifactRef, ArtifactStore
from rlm0.research.contracts import (
    ResearchRun,
    ResearchStage,
    ResearchTrial,
    run_to_dict,
)
from rlm0.run import Run

__all__ = [
    "CHAIN_PROTOCOL",
    "ChainedArtifactError",
    "ChainedExecution",
    "ChainedHandoff",
    "ChainedLimitError",
    "ChainedProtocolError",
    "FreshRoot",
    "FreshRootFactory",
    "HandoffLimits",
    "execute_chained",
    "parse_handoff",
]

CHAIN_PROTOCOL = "RLM0_CHAIN_V1"


class ChainedProtocolError(ValueError):
    """A root did not return an acceptable bounded chain handoff."""


class ChainedArtifactError(RuntimeError):
    """An otherwise valid handoff could not be durably persisted."""


class ChainedLimitError(RuntimeError):
    """The chain consumed its root allowance without producing a final stage."""


@dataclass(frozen=True, slots=True)
class HandoffLimits:
    """Independent bounds for the only data passed between fresh roots."""

    summary_chars: int = 4_000
    blackboard_chars: int = 8_000
    next_chars: int = 1_000
    artifact_update_chars: int = 8_000

    def __post_init__(self) -> None:
        for name in (
            "summary_chars",
            "blackboard_chars",
            "next_chars",
            "artifact_update_chars",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")


@dataclass(frozen=True, slots=True)
class ChainedHandoff:
    """A validated result from one fresh root.

    ``artifact_update`` is text, not a host path. The strategy stores it by
    content address before another root can see or act on this handoff.
    """

    summary: str
    blackboard: str
    next: str
    artifact_update: str
    final: bool

    def validate(self, limits: HandoffLimits) -> None:
        _bounded("summary", self.summary, limits.summary_chars)
        _bounded("blackboard", self.blackboard, limits.blackboard_chars)
        _bounded("artifact_update", self.artifact_update, limits.artifact_update_chars)
        if self.final:
            if self.next:
                raise ChainedProtocolError("a final handoff must leave NEXT empty")
        else:
            _bounded("next", self.next, limits.next_chars)


class FreshRoot(Protocol):
    """The narrow capability handed to each fresh chain root."""

    def complete(self, task: str, context: str = "") -> Run:
        """Run a normal paired RLM attempt for the supplied task and context."""


FreshRootFactory = Callable[[], FreshRoot]


@dataclass(frozen=True, slots=True)
class ChainedExecution:
    """One research record plus every independent root kept for inspection."""

    research: ResearchRun
    roots: tuple[Run, ...]
    handoffs: tuple[ChainedHandoff, ...]
    artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not self.roots or len(self.roots) != len(self.handoffs):
            raise ValueError("a chain needs one root and handoff per stage")
        if len(self.artifacts) != len(self.handoffs):
            raise ValueError("each accepted chain stage needs an artifact")
        if self.research.trials[0].run != self.roots[-1]:
            raise ValueError("the chained trial must report the final fresh root")

    @property
    def trial(self) -> ResearchTrial:
        return self.research.trials[0]


def parse_handoff(answer: str, *, limits: HandoffLimits) -> ChainedHandoff:
    """Parse one strict ``RLM0_CHAIN_V1({...})`` response.

    Legacy free text is intentionally not accepted. A root that cannot form a
    bounded handoff cannot safely influence the next root.
    """

    if not isinstance(answer, str):
        raise ChainedProtocolError("a chain handoff must be text")
    prefix = f"{CHAIN_PROTOCOL}("
    stripped = answer.strip()
    if not stripped.startswith(prefix) or not stripped.endswith(")"):
        raise ChainedProtocolError(f"handoff must use {CHAIN_PROTOCOL}(...)" )
    try:
        decoded = json.loads(stripped[len(prefix) : -1])
    except json.JSONDecodeError as exc:
        raise ChainedProtocolError("handoff body must be JSON") from exc
    expected = {"summary", "blackboard", "next", "artifact_update", "final"}
    if not isinstance(decoded, dict) or set(decoded) != expected:
        raise ChainedProtocolError("handoff must contain exactly the chain fields")
    summary = decoded["summary"]
    blackboard = decoded["blackboard"]
    next_step = decoded["next"]
    artifact_update = decoded["artifact_update"]
    final = decoded["final"]
    text_fields = (summary, blackboard, next_step, artifact_update)
    if not all(isinstance(value, str) for value in text_fields):
        raise ChainedProtocolError("handoff text fields must be strings")
    if not isinstance(final, bool):
        raise ChainedProtocolError("handoff final field must be boolean")
    handoff = ChainedHandoff(summary, blackboard, next_step, artifact_update, final)
    handoff.validate(limits)
    return handoff


def execute_chained(
    *,
    research_id: str,
    trial_id: str,
    control: Run,
    task: str,
    context: str,
    fresh_root: FreshRootFactory,
    artifact_store: ArtifactStore,
    max_roots: int = 3,
    limits: HandoffLimits | None = None,
) -> ChainedExecution:
    """Run fresh roots until a valid final handoff is produced.

    ``control`` is an already-computed single depth-zero Run and remains the
    immutable comparison record. Every experimental root receives the original
    task and original context. Later roots additionally receive only bounded
    handoff text and content-addressed artifact metadata, never a host path.
    """

    if max_roots < 1:
        raise ValueError("max_roots must be at least one")
    actual_limits = limits or HandoffLimits()
    if not task.strip():
        raise ValueError("task must not be empty")
    if control.task != task:
        raise ValueError("control task must exactly match the chained task")
    if control.baseline is None or len(control.attempts) != 1:
        raise ValueError("control must be one real depth-zero Run")
    if not isinstance(context, str):
        raise TypeError("context must be a string")

    roots: list[Run] = []
    handoffs: list[ChainedHandoff] = []
    artifacts: list[ArtifactRef] = []
    stages: list[ResearchStage] = []
    previous: ChainedHandoff | None = None
    previous_artifact: ArtifactRef | None = None

    for ordinal in range(max_roots):
        root = fresh_root()
        root_context = _root_context(context, previous, previous_artifact)
        run = root.complete(task=task, context=root_context)
        if run.task != task:
            raise ChainedProtocolError("fresh root changed the original task")
        if run.answer is None:
            raise ChainedProtocolError("fresh root did not produce a handoff answer")
        handoff = parse_handoff(run.answer, limits=actual_limits)
        try:
            artifact = artifact_store.put_text(
                handoff.artifact_update,
                media_type="text/plain; charset=utf-8",
            )
        except ArtifactLimitError as exc:
            raise ChainedArtifactError(
                "handoff artifact exceeds the configured store"
            ) from exc

        roots.append(run)
        handoffs.append(handoff)
        artifacts.append(artifact)
        stages.append(
            ResearchStage.create(
                ordinal,
                "chained_root",
                {
                    "protocol": CHAIN_PROTOCOL,
                    "max_roots": max_roots,
                    "handoff_limits": asdict(actual_limits),
                },
                metadata={
                    "root_index": ordinal,
                    "original_context_sha256": _hash(context),
                    "root_context_sha256": _hash(root_context),
                    "artifact": _artifact_metadata(artifact),
                    "previous_artifact": (
                        None
                        if previous_artifact is None
                        else _artifact_metadata(previous_artifact)
                    ),
                    "handoff": _handoff_metadata(handoff),
                    "root_run": run_to_dict(run),
                },
            )
        )
        if handoff.final:
            trial = ResearchTrial.create(
                trial_id,
                "chained",
                run,
                stages=tuple(stages),
                config={
                    "protocol": CHAIN_PROTOCOL,
                    "max_roots": max_roots,
                    "handoff_limits": asdict(actual_limits),
                    "original_context_sha256": _hash(context),
                },
                budget={"summary": run.budget_summary},
            )
            research = ResearchRun.create(
                research_id,
                control,
                (trial,),
                config={"strategy": "chained", "protocol": CHAIN_PROTOCOL},
                budget={"summary": control.budget_summary},
            )
            return ChainedExecution(
                research=research,
                roots=tuple(roots),
                handoffs=tuple(handoffs),
                artifacts=tuple(artifacts),
            )
        previous = handoff
        previous_artifact = artifact

    raise ChainedLimitError("chain reached max_roots without a final handoff")


def _root_context(
    original: str,
    previous: ChainedHandoff | None,
    artifact: ArtifactRef | None,
) -> str:
    if previous is None:
        return original
    assert artifact is not None
    return (
        f"{original}\n\n"
        "[rlm0 chained handoff: bounded host-validated text]\n"
        f"SUMMARY:\n{previous.summary}\n\n"
        f"BLACKBOARD:\n{previous.blackboard}\n\n"
        f"NEXT:\n{previous.next}\n\n"
        "ARTIFACT_REF:\n"
        f"sha256:{artifact.digest} ({artifact.size_bytes} bytes; "
        f"{artifact.media_type})\n"
        "Do not assume access to any host path, credential, previous REPL, "
        "or prior conversation."
    )


def _bounded(name: str, value: str, limit: int) -> None:
    if not value.strip():
        raise ChainedProtocolError(f"handoff {name} must not be empty")
    if len(value) > limit:
        raise ChainedProtocolError(
            f"handoff {name} exceeds its {limit}-character limit"
        )


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _artifact_metadata(reference: ArtifactRef) -> dict[str, object]:
    return {
        "sha256": reference.digest,
        "size_bytes": reference.size_bytes,
        "media_type": reference.media_type,
    }


def _handoff_metadata(handoff: ChainedHandoff) -> dict[str, object]:
    return {
        "summary_chars": len(handoff.summary),
        "blackboard_chars": len(handoff.blackboard),
        "next_chars": len(handoff.next),
        "artifact_update_chars": len(handoff.artifact_update),
        "final": handoff.final,
    }
