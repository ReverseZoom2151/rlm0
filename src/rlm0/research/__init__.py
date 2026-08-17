"""Optional research strategies built around the stable RLM runtime."""

from rlm0.research.artifacts import ArtifactLimitError, ArtifactRef, ArtifactStore
from rlm0.research.contracts import ResearchRun, ResearchStage, ResearchTrial
from rlm0.research.events import (
    EventLog,
    EventRecord,
    EventValidationError,
    ResearchReplay,
    read_events,
    replay,
)

__all__ = [
    "ArtifactLimitError",
    "ArtifactRef",
    "ArtifactStore",
    "EventLog",
    "EventRecord",
    "EventValidationError",
    "ResearchReplay",
    "ResearchRun",
    "ResearchStage",
    "ResearchTrial",
    "read_events",
    "replay",
]
