"""Small hand-built stand-ins for the real datasets.

Deliberately written out by hand rather than trimmed from a download. A
fixture cut from real data is a fixture that cannot be committed, which is how
a test suite acquires a network dependency it did not mean to have. These
carry the same field names and the same value shapes as the published rows,
which is all an adapter reads.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "niah_rows",
    "oolong_real_rows",
    "oolong_synth_rows",
    "write_jsonl",
]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return path


def _window(label_a: str, label_b: str, n: int) -> str:
    lines = [f"The following lines contain {n} text messages, one per line."]
    for index in range(n):
        label = label_a if index % 3 else label_b
        lines.append(f"{index}. message {index} looks like {label}")
    return "\n".join(lines)


def oolong_synth_rows() -> list[dict[str, Any]]:
    """One row per answer type the official scorer branches on.

    The numeric and date rows exist because those are the two branches with
    behaviour beyond string equality, and a fixture that only covers the label
    case would leave the partial-credit formula untested.
    """
    return [
        {
            "id": 110010000,
            "context_window_id": 10000,
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": _window("spam", "ham", 12),
            "context_window_text_with_labels": "",
            "question": "In the above data, which of the labels is the most common?",
            "task_group": "counting",
            "task": "TASK_TYPE.MOST_FREQ",
            "answer": "['spam']",
            "answer_type": "ANSWER_TYPE.LABEL",
            "input_subset": "False",
            "num_labels": 2,
        },
        {
            "id": 110010001,
            "context_window_id": 10000,
            "context_len": 1024,
            "dataset": "spam",
            "context_window_text": _window("spam", "ham", 12),
            "context_window_text_with_labels": "",
            "question": "In the above data, how many messages are labelled ham?",
            "task_group": "counting",
            "task": "TASK_TYPE.COUNT",
            "answer": "[4]",
            "answer_type": "ANSWER_TYPE.NUMERIC",
            "input_subset": "False",
            "num_labels": 2,
        },
        {
            "id": 110010002,
            "context_window_id": 10001,
            "context_len": 2048,
            "dataset": "spam",
            "context_window_text": _window("spam", "ham", 8),
            "context_window_text_with_labels": "",
            "question": "On what date was the first message sent?",
            "task_group": "timeline",
            "task": "TASK_TYPE.FIRST_DATE",
            "answer": "[datetime.date(2020, 1, 2)]",
            "answer_type": "ANSWER_TYPE.DATE",
            "input_subset": "False",
            "num_labels": 2,
        },
    ]


def oolong_real_rows() -> list[dict[str, Any]]:
    """Rows covering the integer, string and list scoring branches."""
    transcript = "\n".join(
        f"SPEAKER {i % 3}: line {i} of the session, a d20 roll of {i % 20}"
        for i in range(40)
    )
    return [
        {
            "id": "3952f2d5-082f-14b2-5ec4-d9cbedd2f865",
            "context_window_id": "e4fb38b9-ffca-0729-d52a-02fffd17610a",
            "context_window_text": transcript,
            "question": "Total number of rolls in this episode?",
            "answer": "84",
            "question_type": "singledoc_rolls",
            "episodes": [1],
            "campaign": "campaign2",
        },
        {
            "id": "0c1d2e3f-4a5b-6c7d-8e9f-0a1b2c3d4e5f",
            "context_window_id": "e4fb38b9-ffca-0729-d52a-02fffd17610a",
            "context_window_text": transcript,
            "question": "Which spells were cast in this episode?",
            "answer": "fireball, shield",
            "question_type": "singledoc_spells",
            "episodes": [1],
            "campaign": "campaign2",
        },
        {
            "id": "9f8e7d6c-5b4a-3210-fedc-ba9876543210",
            "context_window_id": "e4fb38b9-ffca-0729-d52a-02fffd17610a",
            "context_window_text": transcript,
            "question": "Who rolled the highest?",
            "answer": "Caleb",
            "question_type": "singledoc_index",
            "episodes": [1],
            "campaign": "campaign2",
        },
    ]


NEEDLE_VALUE = "8090293"


def _haystack(needle_at: int, total: int) -> str:
    lines: list[str] = []
    for index in range(total):
        if index == needle_at:
            lines.append(
                "One of the special magic numbers for wandering-age is "
                f"{NEEDLE_VALUE}."
            )
        lines.append(
            "The grass is green. The sky is blue. The sun is yellow. Here we go."
        )
    return "\n".join(lines)


def niah_rows() -> list[dict[str, Any]]:
    """Mirror-shaped rows, plus one in RULER's own prompt-blob shape.

    Both shapes are exercised because the second needs the question to be
    recovered from the prompt, and a heuristic with no test is a heuristic that
    will be wrong quietly.
    """
    question = (
        "What is the special magic number for wandering-age mentioned in the "
        "provided text?"
    )
    return [
        {
            "index": 0,
            "context": _haystack(30, 60),
            "question": question,
            "answer_prefix": (
                "The special magic number for wandering-age mentioned in the "
                "provided text is"
            ),
            "answer": [NEEDLE_VALUE],
            "task": "niah_single_1",
            "max_new_tokens": 30,
        },
        {
            "index": 1,
            "context": _haystack(10, 40),
            "question": question,
            "answer_prefix": "",
            "answer": [NEEDLE_VALUE],
            "task": "niah_single_2",
            "max_new_tokens": 30,
        },
        {
            "index": 2,
            "context": _haystack(5, 20),
            "question": question,
            "answer_prefix": "",
            "answer": [NEEDLE_VALUE],
            "task": "niah_multivalue",
            "max_new_tokens": 30,
        },
    ]


def niah_native_row() -> dict[str, Any]:
    """A row in the shape RULER's own prepare.py writes."""
    return {
        "index": 7,
        "input": (
            "Some special magic numbers are hidden within the following text.\n\n"
            f"{_haystack(12, 30)}\n\n"
            "What is the special magic number for wandering-age mentioned in "
            "the provided text?"
        ),
        "outputs": [NEEDLE_VALUE],
        "length": 4096,
    }
