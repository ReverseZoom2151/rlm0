"""Read-only research CLI commands never build a provider or print context."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm0 import cli
from rlm0.research.contracts import ResearchRun, ResearchTrial
from rlm0.research.events import EventLog, write_research_events
from rlm0.research.peek import MapIdentity, MapStore, build_context_map
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage

_SECRET = "context-that-must-not-reach-the-terminal"


def _run(answer: str, *, recursive: bool = False) -> Run:
    root = CallRecord(
        role=Role.ROOT,
        depth=0,
        model="fake",
        usage=TokenUsage(),
        wall_clock_s=0.0,
        cost_usd=0.0,
    )
    if not recursive:
        attempts: tuple[Attempt, ...] = (
            Attempt(0, Outcome.ANSWERED, (root,), 0.0, answer=answer),
        )
    else:
        sub = CallRecord(
            role=Role.SUB,
            depth=1,
            model="fake",
            usage=TokenUsage(),
            wall_clock_s=0.0,
            cost_usd=0.0,
        )
        attempts = (
            Attempt(0, Outcome.ITERATIONS_EXHAUSTED, (root,), 0.0),
            Attempt(1, Outcome.ANSWERED, (root, sub), 0.0, answer=answer),
        )
    return Run("task", attempts, "budget")


def _events(path: Path) -> None:
    trial = ResearchTrial.create("trial-one", "srlm", _run(_SECRET, recursive=True))
    record = ResearchRun.create("session-one", _run(_SECRET), (trial,))
    write_research_events(EventLog(path), record, at="2026-08-17T00:00:00Z")


def test_research_replay_validates_without_printing_answers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = tmp_path / "session.jsonl"
    _events(events)

    assert cli.main(["research", "replay", str(events)]) == cli.EXIT_OK

    captured = capsys.readouterr()
    assert "research: session-one" in captured.out
    assert "strategy=srlm" in captured.out
    assert _SECRET not in captured.out
    assert not captured.err


def test_research_replay_refuses_tampered_log(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    events = tmp_path / "session.jsonl"
    _events(events)
    rows = events.read_text(encoding="utf-8").splitlines()
    changed = json.loads(rows[0])
    changed["payload"]["research"]["research_id"] = "tampered"
    rows[0] = json.dumps(changed)
    events.write_text("\n".join(rows) + "\n", encoding="utf-8")

    assert cli.main(["research", "replay", str(events)]) == cli.EXIT_CONFIG
    assert "could not be validated" in capsys.readouterr().err


def test_research_screen_shows_verdicts_but_not_checker_detail(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "screen.json"
    report.write_text(
        json.dumps(
            {
                "verdict": "unknown",
                "results": [
                    {"checker": "static", "verdict": "safe", "detail": _SECRET},
                    {
                        "checker": "model-check",
                        "verdict": "unknown",
                        "detail": "unavailable",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert cli.main(["research", "inspect-screen", str(report)]) == cli.EXIT_OK

    captured = capsys.readouterr()
    assert "screen verdict: unknown" in captured.out
    assert "static: safe" in captured.out
    assert _SECRET not in captured.out


def test_research_screen_refuses_inconsistent_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = tmp_path / "screen.json"
    report.write_text(
        '{"verdict":"safe","results":[{"checker":"x","verdict":"unsafe"}]}',
        encoding="utf-8",
    )

    assert cli.main(["research", "inspect-screen", str(report)]) == cli.EXIT_CONFIG
    assert "screen report is invalid" in capsys.readouterr().err


def test_research_map_prints_identity_and_coverage_not_summaries(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    context = "abcdefgh"
    identity = MapIdentity.for_context(
        context,
        builder_id="peek-v1",
        model="fake-map",
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        max_entries=2,
        summary_char_limit=100,
    )
    context_map = build_context_map(
        context,
        identity,
        lambda _span, _index, _total: _SECRET,
    )
    path = MapStore(tmp_path / "maps").save(context_map)

    assert cli.main(["research", "inspect-map", str(path)]) == cli.EXIT_OK

    captured = capsys.readouterr()
    assert f"map key: {identity.key}" in captured.out
    assert "sections: 2/2" in captured.out
    assert _SECRET not in captured.out


def test_research_inspection_refuses_a_symlink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        return

    assert cli.main(["research", "inspect-screen", str(link)]) == cli.EXIT_CONFIG
    assert "not a regular file" in capsys.readouterr().err
