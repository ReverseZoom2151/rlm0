"""Tests for the observation format and for history compaction.

Two properties carry the weight. Values never leave the sandbox, only names.
And output is capped, with a notice that says what to do instead, because a
cap without that notice just looks to the model like a broken environment.
"""

from __future__ import annotations

import pytest

from rlm0.observation import (
    DEFAULT_STDOUT_CAP,
    CompactedHistory,
    HistoryTurn,
    compact_history,
    format_observation,
    truncate,
)
from rlm0.ports import ExecResult


def result(
    stdout: str = "",
    stderr: str = "",
    ok: bool = True,
    variables: tuple[str, ...] = ("context",),
    truncated_stdout: bool = False,
) -> ExecResult:
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        wall_clock_s=0.1,
        ok=ok,
        truncated_stdout=truncated_stdout,
        variables=variables,
    )


# --------------------------------------------------------------------------
# per-turn observation


def test_observation_reports_what_ran_and_what_it_printed() -> None:
    text = format_observation("print(len(context))", result(stdout="33094859\n"))
    assert "print(len(context))" in text
    assert "33094859" in text
    assert "REPL output" in text


def test_variables_are_listed_by_name_only() -> None:
    text = format_observation(
        "x = 1", result(variables=("context", "chunks", "answers"))
    )
    assert "context, chunks, answers" in text
    assert "names only" in text


def test_no_output_says_so_rather_than_showing_a_blank() -> None:
    text = format_observation("x = 1", result())
    assert "(nothing printed)" in text


def test_errors_are_shown_and_the_session_is_said_to_survive() -> None:
    text = format_observation(
        "1/0", result(stderr="ZeroDivisionError: division by zero", ok=False)
    )
    assert "ZeroDivisionError" in text
    assert "survived" in text


def test_long_stderr_keeps_the_end_where_the_exception_is() -> None:
    stderr = "frame\n" * 2000 + "ValueError: the actual problem"
    text = format_observation("boom()", result(stderr=stderr, ok=False))
    assert "ValueError: the actual problem" in text


def test_truncation_fires_and_teaches_the_sub_call() -> None:
    text = format_observation("print(context[0])", result(stdout="x" * 200_000))
    assert len(text) < DEFAULT_STDOUT_CAP + 2_000
    assert "llm_query" in text
    assert "was cut" in text


def test_truncation_message_at_depth_zero_does_not_promise_a_sub_call() -> None:
    text = format_observation(
        "print(context[0])", result(stdout="y" * 200_000), sub_calls=False
    )
    assert "llm_query" not in text
    assert "in code" in text


def test_sandbox_reported_truncation_is_surfaced_too() -> None:
    text = format_observation(
        "print(x)", result(stdout="short", truncated_stdout=True)
    )
    assert "was cut" in text


def test_truncate_keeps_both_ends() -> None:
    body = "HEAD" + "m" * 10_000 + "TAIL"
    cut, was_cut = truncate(body, 1_000)
    assert was_cut
    assert cut.startswith("HEAD")
    assert cut.endswith("TAIL")
    assert "characters cut here" in cut


def test_truncate_leaves_short_text_alone() -> None:
    cut, was_cut = truncate("small", 1_000)
    assert cut == "small"
    assert not was_cut


def test_truncate_refuses_a_nonsense_cap() -> None:
    with pytest.raises(ValueError, match="cap must be positive"):
        truncate("abc", 0)


# --------------------------------------------------------------------------
# history compaction


def turn(index: int, size: int = 100, variables: tuple[str, ...] = ()) -> HistoryTurn:
    return HistoryTurn(
        index=index,
        assistant="a" * size,
        observation="o" * size,
        variables=variables,
    )


def test_a_short_history_is_left_alone() -> None:
    turns = [turn(i) for i in range(3)]
    out = compact_history(turns, char_budget=10_000)
    assert out.turns == tuple(turns)
    assert not out.compacted
    assert out.notice == ""


def test_older_turns_move_into_repl_variables() -> None:
    turns = [turn(i, size=1_000, variables=(f"buf{i}",)) for i in range(6)]
    out = compact_history(turns, char_budget=4_000, keep_recent=2)
    assert out.compacted
    assert len(out.turns) < len(turns)
    assert out.turns[-1].index == 5
    assert out.char_len <= 4_000 + len(out.notice)
    stashed = {s.name for s in out.stashes}
    assert "_turn_000" in stashed


def test_the_stash_holds_the_real_text() -> None:
    turns = [turn(i, size=1_000) for i in range(6)]
    out = compact_history(turns, char_budget=3_000, keep_recent=1)
    first = out.stashes[0]
    assert turns[0].assistant in first.value
    assert turns[0].observation in first.value


def test_the_notice_preserves_the_buffer_inventory() -> None:
    turns = [turn(i, size=1_000, variables=(f"buf{i}", "context")) for i in range(6)]
    out = compact_history(turns, char_budget=3_000, keep_recent=2)
    assert "buf0" in out.notice
    assert out.notice.count("context") == 1
    assert out.notice.startswith("[")
    assert "\n" not in out.notice


def test_recent_turns_are_never_folded_even_over_budget() -> None:
    turns = [turn(i, size=5_000) for i in range(4)]
    out = compact_history(turns, char_budget=100, keep_recent=3)
    assert len(out.turns) == 3
    assert [t.index for t in out.turns] == [1, 2, 3]


def test_compaction_arguments_are_checked() -> None:
    with pytest.raises(ValueError, match="keep_recent"):
        compact_history([turn(0)], keep_recent=0)
    with pytest.raises(ValueError, match="char_budget"):
        compact_history([turn(0)], char_budget=0)


def test_empty_history_compacts_to_nothing() -> None:
    out = compact_history([])
    assert out == CompactedHistory(())
