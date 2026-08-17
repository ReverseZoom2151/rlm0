from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm0.research.contracts import ResearchRun, ResearchTrial
from rlm0.research.events import (
    EventLog,
    EventValidationError,
    read_events,
    replay,
    write_research_events,
)
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage


def _run(answer: str, *, depth: int = 0) -> Run:
    root = CallRecord(
        role=Role.ROOT,
        depth=0,
        model="m",
        usage=TokenUsage(),
        wall_clock_s=0.0,
        cost_usd=0.0,
    )
    attempts: tuple[Attempt, ...]
    if depth == 0:
        attempts = (Attempt(0, Outcome.ANSWERED, (root,), 0.0, answer=answer),)
    else:
        child = CallRecord(
            role=Role.SUB,
            depth=1,
            model="m",
            usage=TokenUsage(),
            wall_clock_s=0.0,
            cost_usd=0.0,
        )
        attempts = (
            Attempt(0, Outcome.ITERATIONS_EXHAUSTED, (root,), 0.0),
            Attempt(1, Outcome.ANSWERED, (root, child), 0.0, answer=answer),
        )
    return Run("task", attempts, "budget")


def _research() -> ResearchRun:
    trial = ResearchTrial.create("one", "srlm", _run("trial", depth=1))
    return ResearchRun.create("session", _run("control"), (trial,))


def test_event_log_replays_a_record_without_calling_a_provider(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    research = _research()
    write_research_events(EventLog(path), research, at="2026-08-17T00:00:00Z")

    replayed = replay(read_events(path))

    assert replayed.research == research
    assert [event.kind for event in replayed.events] == [
        "research_started",
        "trial_recorded",
        "research_complete",
    ]


def test_tampering_breaks_the_hash_chain(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    write_research_events(EventLog(path), _research(), at="time")
    rows = path.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["payload"]["research"]["research_id"] = "tampered"
    rows[0] = json.dumps(changed)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(EventValidationError, match="digest"):
        read_events(path)


def test_replay_refuses_missing_trial_record(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = EventLog(path)
    research = _research()
    log.append(
        "research_started",
        {"research": research.to_dict(), "research_schema": 1},
        at="t",
    )
    log.append("research_complete", {"research_id": research.research_id}, at="t")

    with pytest.raises(EventValidationError, match="every trial"):
        replay(read_events(path))
