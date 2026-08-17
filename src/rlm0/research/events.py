"""Append-only, hash-chained JSONL events for research sessions."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm0.research.contracts import (
    RESEARCH_SCHEMA_VERSION,
    ResearchRun,
    canonical_json,
    research_run_from_dict,
)

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "EventLog",
    "EventRecord",
    "EventValidationError",
    "ResearchReplay",
    "read_events",
    "replay",
    "write_research_events",
]

EVENT_SCHEMA_VERSION = 1
_GENESIS = "0" * 64


def _digest(
    sequence: int,
    kind: str,
    at: str,
    payload_json: str,
    previous_digest: str,
) -> str:
    body = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "sequence": sequence,
        "kind": kind,
        "at": at,
        "payload": json.loads(payload_json),
        "previous_digest": previous_digest,
    }
    return hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()


class EventValidationError(ValueError):
    """An event log was modified, truncated, or does not describe one session."""


@dataclass(frozen=True, slots=True)
class EventRecord:
    """One hash-chained event with canonical JSON payload."""

    sequence: int
    kind: str
    at: str
    payload_json: str
    previous_digest: str
    digest: str
    schema_version: int = EVENT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        sequence: int,
        kind: str,
        at: str,
        payload: Mapping[str, Any],
        previous_digest: str,
    ) -> EventRecord:
        payload_json = canonical_json(dict(payload))
        digest = _digest(sequence, kind, at, payload_json, previous_digest)
        return cls(sequence, kind, at, payload_json, previous_digest, digest)

    def __post_init__(self) -> None:
        if self.schema_version != EVENT_SCHEMA_VERSION:
            raise EventValidationError("unsupported event schema")
        if self.sequence < 0 or not self.kind.strip() or not self.at.strip():
            raise EventValidationError(
                "event needs a nonnegative sequence, kind, and time"
            )
        if len(self.previous_digest) != 64 or len(self.digest) != 64:
            raise EventValidationError("event digest must be SHA-256")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as error:
            raise EventValidationError("event payload is not JSON") from error
        if not isinstance(payload, dict):
            raise EventValidationError("event payload must be an object")
        expected = _digest(
            self.sequence,
            self.kind,
            self.at,
            canonical_json(payload),
            self.previous_digest,
        )
        if self.digest != expected:
            raise EventValidationError("event digest does not match content")

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self.payload_json)
        assert isinstance(value, dict)
        return value

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "kind": self.kind,
            "at": self.at,
            "payload": self.payload,
            "previous_digest": self.previous_digest,
            "digest": self.digest,
        }


class EventLog:
    """An append-only event file; construction validates its entire history."""

    def __init__(self, path: Path) -> None:
        self.path = path
        events = read_events(path) if path.exists() else ()
        self._next_sequence = len(events)
        self._previous_digest = events[-1].digest if events else _GENESIS

    def append(self, kind: str, payload: Mapping[str, Any], *, at: str) -> EventRecord:
        event = EventRecord.create(
            self._next_sequence, kind, at, payload, self._previous_digest
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(event.to_dict()) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._next_sequence += 1
        self._previous_digest = event.digest
        return event


def _event_from_dict(raw: Mapping[str, Any]) -> EventRecord:
    payload = raw.get("payload")
    if not isinstance(payload, Mapping):
        raise EventValidationError("event payload must be an object")
    return EventRecord(
        sequence=int(raw["sequence"]),
        kind=str(raw["kind"]),
        at=str(raw["at"]),
        payload_json=canonical_json(dict(payload)),
        previous_digest=str(raw["previous_digest"]),
        digest=str(raw["digest"]),
        schema_version=int(raw.get("schema_version", EVENT_SCHEMA_VERSION)),
    )


def read_events(path: Path) -> tuple[EventRecord, ...]:
    """Load and validate a complete JSONL chain without running a model."""
    events: list[EventRecord] = []
    previous = _GENESIS
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise EventValidationError(f"{path}:{line_number}: blank event line")
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise EventValidationError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(raw, Mapping):
            raise EventValidationError(f"{path}:{line_number}: event must be an object")
        event = _event_from_dict(raw)
        if event.sequence != len(events):
            raise EventValidationError(f"{path}:{line_number}: noncontiguous sequence")
        if event.previous_digest != previous:
            raise EventValidationError(f"{path}:{line_number}: broken digest chain")
        events.append(event)
        previous = event.digest
    return tuple(events)


@dataclass(frozen=True, slots=True)
class ResearchReplay:
    """A validated persisted session, reconstructed without provider calls."""

    research: ResearchRun
    events: tuple[EventRecord, ...]


def write_research_events(log: EventLog, research: ResearchRun, *, at: str) -> None:
    """Persist enough events to reconstruct a research record deterministically."""
    log.append(
        "research_started",
        {"research": research.to_dict(), "research_schema": RESEARCH_SCHEMA_VERSION},
        at=at,
    )
    for trial in research.trials:
        log.append(
            "trial_recorded",
            {
                "trial_id": trial.trial_id,
                "config_fingerprint": trial.config_fingerprint,
                "budget_fingerprint": trial.budget_fingerprint,
            },
            at=at,
        )
    log.append("research_complete", {"research_id": research.research_id}, at=at)


def replay(events: Iterable[EventRecord]) -> ResearchReplay:
    """Validate event semantics and recreate the submitted immutable record."""
    event_tuple = tuple(events)
    if not event_tuple or event_tuple[0].kind != "research_started":
        raise EventValidationError("a research log must begin with research_started")
    start = event_tuple[0].payload
    raw_research = start.get("research")
    if not isinstance(raw_research, Mapping):
        raise EventValidationError("research_started event needs a research record")
    research = research_run_from_dict(raw_research)
    trials = {trial.trial_id: trial for trial in research.trials}
    recorded: set[str] = set()
    complete = False
    for event in event_tuple[1:]:
        if event.kind == "trial_recorded":
            trial_id = event.payload.get("trial_id")
            if not isinstance(trial_id, str) or trial_id not in trials:
                raise EventValidationError("trial_recorded names an unknown trial")
            trial = trials[trial_id]
            if trial_id in recorded:
                raise EventValidationError("trial was recorded twice")
            if event.payload.get("config_fingerprint") != trial.config_fingerprint:
                raise EventValidationError(
                    "trial event configuration fingerprint mismatch"
                )
            if event.payload.get("budget_fingerprint") != trial.budget_fingerprint:
                raise EventValidationError("trial event budget fingerprint mismatch")
            recorded.add(trial_id)
        elif event.kind == "research_complete":
            if complete or event.payload.get("research_id") != research.research_id:
                raise EventValidationError("invalid research completion event")
            complete = True
        else:
            raise EventValidationError(f"unknown event kind {event.kind}")
    if recorded != set(trials):
        raise EventValidationError("not every trial was recorded")
    if not complete:
        raise EventValidationError("research log has no completion event")
    return ResearchReplay(research, event_tuple)
