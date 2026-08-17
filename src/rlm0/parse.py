"""Turning what the model emitted into code to run and answers to accept.

This is a small module with a bad reputation. Every failure mode handled here
was observed in a surveyed implementation, and all of them are silent: the run
looks fine and the answer is wrong or missing.

- A non-greedy paren match, `FINAL\\((.*?)\\)`, truncates every answer that
  contains a parenthesis, which for prose answers is most of them. The
  argument is matched by balancing here instead.
- A directive matched anywhere in the message fires on a model that wrote
  `FINAL(...)` inside its own code, or inside a string it passed to a
  sub-model, or while quoting the instructions back. Code fences and inline
  backtick spans are masked out before the search, so quoting is safe.
- A sentinel variable, meaning the runtime harvests whatever is bound to a
  well known name such as `final_answer` or `result`, fires whenever the model
  happens to use that name for scratch. There is no sentinel here. A variable
  becomes the answer only when the model names it in FINAL_VAR.
- A final answer accepted on a turn where nothing has run yet is the model
  answering from priors. It is rejected, with a reason the caller can log,
  rather than dropped.

Nothing here raises on malformed input. A model that emits nonsense should get
another turn, not a traceback in the orchestrator.
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "EXECUTABLE_LANGUAGES",
    "CodeBlock",
    "CompletionSource",
    "FinalAnswer",
    "FinalKind",
    "ParsedTurn",
    "Rejection",
    "extract_code_blocks",
    "find_final_answer",
    "parse_turn",
]

EXECUTABLE_LANGUAGES: frozenset[str] = frozenset(
    {"repl", "python", "python3", "py", ""}
)
"""Fence tags whose contents are run.

`repl` is what the prompt asks for. The rest are here because models reach for
`python` habitually, and refusing to run a block over its tag teaches the model
nothing except that its code vanished.
"""


class FinalKind(StrEnum):
    LITERAL = "literal"
    """The answer text was written inline in FINAL(...)."""

    VARIABLE = "variable"
    """FINAL_VAR(name) named a REPL variable holding the answer."""

    PROTOCOL = "protocol"
    """A structured, versioned completion envelope supplied the answer."""


class CompletionSource(StrEnum):
    """How an answer crossed the model/runtime boundary.

    ``RLM0_FINAL_V1`` is the reportable protocol. ``FINAL`` and ``FINAL_VAR``
    remain accepted so existing trajectories do not become unreadable, but a
    caller can distinguish their recovered answer from a structured one.
    """

    V1 = "rlm0_final_v1"
    RECOVERED_FINAL = "recovered_final"
    RECOVERED_FINAL_VAR = "recovered_final_var"


class Rejection(StrEnum):
    """Why a directive that was present did not become the answer."""

    NO_CODE_HAS_RUN = "no_code_has_run"
    CODE_IN_SAME_TURN = "code_in_same_turn"
    MALFORMED_VARIABLE = "malformed_variable"
    EMPTY_ANSWER = "empty_answer"
    MALFORMED_PROTOCOL = "malformed_protocol"


@dataclass(frozen=True, slots=True)
class CodeBlock:
    """One fenced block, with where it sat so callers can report on it."""

    code: str
    language: str
    start: int
    end: int
    terminated: bool = True

    @property
    def executable(self) -> bool:
        return self.language in EXECUTABLE_LANGUAGES


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """A final-answer directive, already validated."""

    kind: FinalKind
    value: str
    """The answer text for LITERAL, the variable name for VARIABLE."""

    terminated: bool = True
    """False when the directive was never closed and the tail was taken."""

    source: CompletionSource = CompletionSource.RECOVERED_FINAL
    protocol_version: int | None = None
    evidence: tuple[str, ...] = ()
    answer_artifact: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedTurn:
    """Everything one assistant message means to the runtime."""

    code_blocks: tuple[CodeBlock, ...]
    final: FinalAnswer | None = None
    rejection: Rejection | None = None
    """Set when a directive was found and refused. `final` is then None."""

    @property
    def executable_blocks(self) -> tuple[CodeBlock, ...]:
        return tuple(b for b in self.code_blocks if b.executable)

    @property
    def is_empty(self) -> bool:
        """No code and no answer, so the turn did nothing at all."""
        return not self.executable_blocks and self.final is None


_FENCE_OPEN = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^\n`]*)$")
_DIRECTIVE = re.compile(r"(?<![0-9A-Za-z_])FINAL(?P<var>_VAR)?[ \t]*\(")
_V1_DIRECTIVE = re.compile(r"(?<![0-9A-Za-z_])RLM0_FINAL_V1[ \t]*\(")
_INLINE_CODE = re.compile(r"(?P<ticks>`+)(?P<body>[^\n]*?)(?P=ticks)")


def extract_code_blocks(text: str) -> tuple[CodeBlock, ...]:
    """Every fenced block in the message, in order.

    Fences are matched by run length and by character, so a nested fence
    inside a longer one is content rather than a terminator. An unterminated
    fence runs to the end of the message and is flagged, because the usual
    cause is the model hitting its output limit mid block and the code is
    still worth seeing.
    """
    blocks: list[CodeBlock] = []
    lines = text.splitlines(keepends=True)
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line)

    i = 0
    while i < len(lines):
        opened = _FENCE_OPEN.match(lines[i].rstrip("\r\n"))
        if opened is None:
            i += 1
            continue
        fence = opened.group("fence")
        info = opened.group("info").strip().lower()
        language = info.split()[0] if info.split() else ""
        body_start = offsets[i] + len(lines[i])
        j = i + 1
        closed_at: int | None = None
        while j < len(lines):
            stripped = lines[j].strip()
            if (
                stripped.startswith(fence[0] * len(fence))
                and set(stripped) == {fence[0]}
            ):
                closed_at = j
                break
            j += 1
        if closed_at is None:
            blocks.append(
                CodeBlock(
                    code=text[body_start:],
                    language=language,
                    start=offsets[i],
                    end=len(text),
                    terminated=False,
                )
            )
            break
        blocks.append(
            CodeBlock(
                code=text[body_start : offsets[closed_at]],
                language=language,
                start=offsets[i],
                end=offsets[closed_at] + len(lines[closed_at]),
            )
        )
        i = closed_at + 1
    return tuple(blocks)


def _mask_code(text: str) -> str:
    """The message with every code region blanked, offsets preserved.

    Blanking rather than deleting keeps every index in the masked copy usable
    against the original, which is what lets the directive search run on the
    masked text and the answer be sliced out of the real one.
    """
    masked = list(text)
    for block in extract_code_blocks(text):
        for k in range(block.start, block.end):
            if masked[k] != "\n":
                masked[k] = " "
    joined = "".join(masked)
    out = list(joined)
    for span in _INLINE_CODE.finditer(joined):
        for k in range(span.start(), span.end()):
            if out[k] != "\n":
                out[k] = " "
    return "".join(out)


def _argument_span(masked: str, open_idx: int) -> tuple[int, bool]:
    """Where the argument of a directive ends, by balancing parentheses.

    Returns the index just past the closing paren and whether one was found.
    Quotes are not tracked, deliberately. An answer is prose, and prose is
    full of apostrophes, so treating a quote as a region delimiter mangles far
    more answers than it saves.
    """
    depth = 0
    for k in range(open_idx, len(masked)):
        char = masked[k]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return k, True
    return len(masked), False


def _last_unmatched_close(masked: str, start: int) -> int | None:
    """The last top-level `)` in the tail, if the tail has one.

    This repairs the other half of the paren problem. `FINAL(the answer is
    (a) or (b) :) ...` balances early on a stray close, and the honest reading
    is that the model wrote an unbalanced answer rather than that it stopped
    there. If the remaining text closes more parens than it opens, the answer
    extends to the last of those closes.
    """
    depth = 0
    found: int | None = None
    for k in range(start, len(masked)):
        char = masked[k]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                found = k
            else:
                depth -= 1
    return found


def find_final_answer(text: str) -> tuple[FinalAnswer | None, Rejection | None]:
    """The final-answer directive in a message, if there is a usable one.

    Searches only the prose. When several directives appear, the last wins,
    since the earlier ones are usually the model narrating what it is about to
    do.
    """
    masked = _mask_code(text)
    protocol_matches = list(_V1_DIRECTIVE.finditer(masked))
    legacy_matches = list(_DIRECTIVE.finditer(masked))
    matches = [(match.start(), "v1", match) for match in protocol_matches]
    matches.extend((match.start(), "legacy", match) for match in legacy_matches)
    if not matches:
        return None, None
    _, family, match = max(matches, key=lambda item: item[0])
    open_idx = match.end() - 1
    close_idx, terminated = _argument_span(masked, open_idx)
    if terminated:
        extended = _last_unmatched_close(masked, close_idx + 1)
        if extended is not None:
            close_idx = extended
    raw = text[open_idx + 1 : close_idx]
    value = raw.strip()
    if family == "v1":
        return _parse_v1(value, terminated)
    if match.group("var"):
        name = value.strip("`")
        if not name.isidentifier() or keyword.iskeyword(name):
            return None, Rejection.MALFORMED_VARIABLE
        return (
            FinalAnswer(
                FinalKind.VARIABLE,
                name,
                terminated,
                CompletionSource.RECOVERED_FINAL_VAR,
            ),
            None,
        )
    if not value:
        return None, Rejection.EMPTY_ANSWER
    return (
        FinalAnswer(
            FinalKind.LITERAL,
            value,
            terminated,
            CompletionSource.RECOVERED_FINAL,
        ),
        None,
    )


def _parse_v1(
    value: str, terminated: bool
) -> tuple[FinalAnswer | None, Rejection | None]:
    """Parse the one completion shape that may back a reportable result.

    The envelope intentionally has no permissive defaults. A malformed version
    marker is a model mistake to correct next turn, not an opportunity to turn
    arbitrary prose into a clean structured answer.
    """
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return None, Rejection.MALFORMED_PROTOCOL
    if not isinstance(payload, dict):
        return None, Rejection.MALFORMED_PROTOCOL
    required = {"protocol_version", "status", "answer", "evidence", "answer_artifact"}
    if set(payload) != required:
        return None, Rejection.MALFORMED_PROTOCOL
    if payload["protocol_version"] != 1 or payload["status"] != "answered":
        return None, Rejection.MALFORMED_PROTOCOL
    answer = payload["answer"]
    evidence = payload["evidence"]
    artifact = payload["answer_artifact"]
    if not isinstance(answer, str) or not answer.strip():
        return None, Rejection.EMPTY_ANSWER
    if not isinstance(evidence, list) or not all(
        isinstance(item, str) for item in evidence
    ):
        return None, Rejection.MALFORMED_PROTOCOL
    if artifact is not None and not isinstance(artifact, str):
        return None, Rejection.MALFORMED_PROTOCOL
    return (
        FinalAnswer(
            FinalKind.PROTOCOL,
            answer,
            terminated,
            CompletionSource.V1,
            protocol_version=1,
            evidence=tuple(evidence),
            answer_artifact=artifact,
        ),
        None,
    )


def parse_turn(text: str, *, code_has_run: bool = True) -> ParsedTurn:
    """Read one assistant message.

    `code_has_run` is the runtime saying whether anything has executed in this
    attempt yet. When it has not, a final answer is refused: the model has not
    read a character of the context, so whatever it wrote came from its priors
    and the first-turn safeguard in the prompt was ignored.

    A message carrying both runnable code and a directive runs the code and
    refuses the directive. The two orderings are not symmetric: deferring the
    answer costs one turn and the model can repeat it, whereas accepting it
    throws away code the model wrote because it thought it still needed it.
    """
    blocks = extract_code_blocks(text)
    final, rejection = find_final_answer(text)
    executable = tuple(b for b in blocks if b.executable)
    if final is not None and not code_has_run:
        return ParsedTurn(blocks, None, Rejection.NO_CODE_HAS_RUN)
    if final is not None and executable:
        return ParsedTurn(blocks, None, Rejection.CODE_IN_SAME_TURN)
    return ParsedTurn(blocks, final, rejection)
