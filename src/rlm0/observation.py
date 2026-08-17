"""What the model is shown after each turn, and what it stops being shown.

The observation is the only channel from the environment back into the window,
so its size is the design. The cap is a forcing function and not a courtesy:
if the model can print a 500K character slice and read it, then the REPL is a
delivery mechanism for context and the whole pattern collapses into one very
long single-shot prompt. With the cap in place, the cheapest way to learn what
a large slice says is to hand it to a sub-model, which is the behaviour the
architecture exists to produce. So when truncation fires, the notice says that
in as many words rather than apologising, because the notice is the only place
the model is taught the lesson at the moment it is needed.

Variables are listed by name and never by value, for the same reason. A single
`print(locals())`, or a runtime that helpfully echoes the environment, puts the
whole context back in the window and nothing about the run looks different
afterwards except the bill.

History compaction follows from the same accounting. Root-side tokens grow
with the length of the conversation rather than with the work, so a long run
pays for its own transcript on every turn. Older turns are moved into REPL
variables, where they are still reachable by code, and are replaced by one
line that names what was moved and which buffers the model owns. The model
keeps its inventory and loses only the verbatim text, which is the part it can
read back if it turns out to matter.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from rlm0.ports import ExecResult

__all__ = [
    "DEFAULT_HISTORY_CHAR_BUDGET",
    "DEFAULT_STDERR_CAP",
    "DEFAULT_STDOUT_CAP",
    "CompactedHistory",
    "HistoryTurn",
    "Stash",
    "compact_history",
    "format_observation",
    "truncate",
]

DEFAULT_STDOUT_CAP = 4_000
"""Characters of stdout shown per turn, about a thousand tokens.

Chosen as the smallest size that still fits the things worth seeing: a chunk
length listing, a few dozen lines of counts or indices, a couple of short
extracts, or a sub-model's answer. Anything a model would want beyond that is
the context itself, and the answer to wanting the context itself is a sub-call.
Raising this to the point where a document fits is the change that quietly
turns the run back into single-shot.
"""

DEFAULT_STDERR_CAP = 2_000
"""Tracebacks are shown from the tail, where the exception is."""

DEFAULT_HISTORY_CHAR_BUDGET = 60_000
"""Transcript size that triggers compaction, roughly fifteen thousand tokens."""

_ELISION = "\n\n[... {n} characters cut here. {advice} ...]\n\n"

_ADVICE_SUB = (
    "output is capped so that reading a large slice happens in a sub-call "
    "rather than in your window. Pass the variable to llm_query with the "
    "question you were going to answer by eye, or print a reduction of it"
)

_ADVICE_NO_SUB = (
    "output is capped so that reading happens through a reduction rather "
    "than in your window. Compute what you need from the variable in code "
    "and print that instead"
)


def truncate(
    text: str,
    cap: int,
    *,
    sub_calls: bool = True,
    from_tail: bool = False,
) -> tuple[str, bool]:
    """Cut `text` to `cap` characters, keeping both ends when it is cut.

    Returns the text and whether anything was removed. Both ends are kept
    because the head says what the output is and the tail says how it ended,
    and a head-only cut hides exactly the loop that ran away.
    """
    if cap <= 0:
        raise ValueError("cap must be positive")
    if len(text) <= cap:
        return text, False
    advice = _ADVICE_SUB if sub_calls else _ADVICE_NO_SUB
    if from_tail:
        marker = _ELISION.format(n=len(text) - cap, advice=advice)
        return marker + text[-cap:], True
    head = (cap * 2) // 3
    tail = cap - head
    cut = len(text) - head - tail
    marker = _ELISION.format(n=cut, advice=advice)
    return text[:head] + marker + text[len(text) - tail :], True


def format_observation(
    code: str,
    result: ExecResult,
    *,
    stdout_cap: int = DEFAULT_STDOUT_CAP,
    stderr_cap: int = DEFAULT_STDERR_CAP,
    sub_calls: bool = True,
) -> str:
    """The user message returned to the model after one block ran.

    Three parts, always in the same order and always all present: what ran,
    what it printed, and which names now exist. Echoing the code back matters
    more than it looks: after compaction the model may no longer be able to
    see what it wrote, and a run that cannot say what produced a buffer cannot
    trust the buffer.
    """
    parts = [f"Code executed:\n```python\n{code.strip()}\n```", ""]
    stdout, cut = truncate(result.stdout, stdout_cap, sub_calls=sub_calls)
    if result.stdout:
        parts.append(f"REPL output:\n{stdout}")
    else:
        parts.append("REPL output: (nothing printed)")
    if cut or result.truncated_stdout:
        parts.append(
            "The output above was cut. Nothing was lost from the REPL itself, "
            "only from your view of it."
        )
    if result.stderr:
        stderr, _ = truncate(
            result.stderr, stderr_cap, sub_calls=sub_calls, from_tail=True
        )
        parts.append(f"Errors:\n{stderr}")
    if not result.ok:
        parts.append(
            "That block did not finish. Fix it or take another route; the "
            "session and every name in it survived."
        )
    names = ", ".join(result.variables) if result.variables else "(none)"
    parts.append("")
    parts.append(f"REPL variables now bound (names only): {names}")
    return "\n".join(parts)


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """One exchange, as the transcript holds it."""

    index: int
    assistant: str
    observation: str
    variables: tuple[str, ...] = ()

    @property
    def char_len(self) -> int:
        return len(self.assistant) + len(self.observation)


@dataclass(frozen=True, slots=True)
class Stash:
    """A turn that moved out of the window and into the REPL."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class CompactedHistory:
    """The result of a compaction pass.

    `stashes` are for the caller to write with `Sandbox.set_variable`. This
    module does not touch the sandbox, so a compaction can be computed and
    inspected without one, which is how it gets tested offline.
    """

    turns: tuple[HistoryTurn, ...]
    notice: str = ""
    stashes: tuple[Stash, ...] = field(default_factory=tuple)

    @property
    def compacted(self) -> bool:
        return bool(self.stashes)

    @property
    def char_len(self) -> int:
        return len(self.notice) + sum(turn.char_len for turn in self.turns)


def compact_history(
    turns: Sequence[HistoryTurn],
    *,
    char_budget: int = DEFAULT_HISTORY_CHAR_BUDGET,
    keep_recent: int = 2,
    stash_prefix: str = "_turn_",
) -> CompactedHistory:
    """Fold the oldest turns into REPL variables until the budget is met.

    The most recent turns are never folded regardless of budget. The model's
    next action is chosen almost entirely from what just happened, and a
    compaction that reaches the current turn makes the run forget what it was
    in the middle of.
    """
    if keep_recent < 1:
        raise ValueError("keep_recent must be at least 1")
    if char_budget <= 0:
        raise ValueError("char_budget must be positive")
    ordered = list(turns)
    total = sum(turn.char_len for turn in ordered)
    if total <= char_budget:
        return CompactedHistory(tuple(ordered))

    folded: list[HistoryTurn] = []
    while len(ordered) > keep_recent and total > char_budget:
        oldest = ordered.pop(0)
        folded.append(oldest)
        total -= oldest.char_len
    if not folded:
        return CompactedHistory(tuple(ordered))

    stashes = tuple(
        Stash(
            name=f"{stash_prefix}{turn.index:03d}",
            value=f"{turn.assistant}\n\n{turn.observation}",
        )
        for turn in folded
    )
    owned: list[str] = []
    for turn in folded:
        for name in turn.variables:
            if name not in owned:
                owned.append(name)
    return CompactedHistory(
        tuple(ordered), _notice(stashes, owned), stashes
    )


def _notice(stashes: tuple[Stash, ...], owned: Sequence[str]) -> str:
    """One line, naming what moved and what the model still owns.

    The variable inventory is the part that has to survive. A model that
    forgets it already built `summaries` builds it again, and rebuilding a
    buffer is the most expensive mistake available to it.
    """
    names = ", ".join(stash.name for stash in stashes)
    buffers = ", ".join(owned) if owned else "none recorded"
    return (
        f"[{len(stashes)} earlier turns were moved out of this transcript to "
        f"keep it short. Their full text is in the REPL as {names}, readable "
        f"with code. Variables you built in them and still own: {buffers}.]"
    )
