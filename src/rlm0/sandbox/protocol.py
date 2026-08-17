"""The wire between the host and the code the model wrote.

The format is newline delimited JSON and it is JSON for one reason: the host
process must never deserialize anything the sandbox produced using a format
that can execute code. A surveyed implementation calls `pickle.loads` on
sandbox output in the host process, which hands the host away for free the
moment anything achieves execution inside the boundary. JSON cannot construct
an object, so a hostile sandbox can at worst return a wrong string.

`json.dumps` escapes newlines inside strings, so one message really is one
line, and `ensure_ascii` keeps the frame bytes pure ASCII no matter what the
attacker put in the context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any, Final

__all__ = [
    "PROTOCOL_VERSION",
    "STDOUT_CHAR_CAP",
    "HostCallable",
    "ProtocolError",
    "decode",
    "encode",
    "scrub_secrets",
    "truncation_notice",
]

PROTOCOL_VERSION: Final = 1

STDOUT_CHAR_CAP: Final = 8_000
"""How much stdout survives one execution, in characters.

Roughly two thousand tokens. The number is chosen to be generous enough for
the output a person would actually write on purpose, a count, a table of
matches, a summary, and far too small for the output a model reaches for when
it is avoiding the work: printing a slice of the context so it can read it.
That move is what this runtime exists to prevent, because the context is held
in the environment precisely so that its contents never enter the window, and
a cap that comfortably fits a chunk of the context is not a cap. When it
fires, the message says what to do instead, so truncation teaches rather than
merely annoys.
"""

HostCallable = Callable[..., object]


class ProtocolError(RuntimeError):
    """The other end sent something this end cannot make sense of.

    Always fatal for the channel. A framing error means the two sides no
    longer agree on where messages start, and continuing to parse is how a
    desynchronized stream turns into a plausible looking wrong answer.
    """


def encode(message: Mapping[str, Any]) -> bytes:
    """Frame one message as a single ASCII line."""
    text = json.dumps(message, ensure_ascii=True, separators=(",", ":"))
    return text.encode("ascii") + b"\n"


def decode(line: bytes) -> dict[str, Any]:
    """Parse one framed line, refusing anything that is not an object."""
    try:
        parsed = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"unparseable frame: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ProtocolError(f"expected an object, got {type(parsed).__name__}")
    return parsed


def truncation_notice(*, cap: int, dropped: int) -> str:
    """The message appended when stdout hit the cap.

    Written at the model rather than at the operator, because the model is who
    reads it and who can act on it.
    """
    return (
        f"\n[rlm0: stdout truncated at {cap} chars, {dropped} more dropped. "
        "Printing a slice copies it into the context window, which is the "
        "cost this runtime exists to avoid. Leave the text in its variable "
        "and pass the variable to a sub-call, or return it by name.]"
    )


_SECRET_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "[rlm0:redacted]"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "[rlm0:redacted]"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "[rlm0:redacted]"),
    (re.compile(r"xox[abprs]-[A-Za-z0-9\-]{10,}"), "[rlm0:redacted]"),
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "[rlm0:redacted]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{20,}"), "Bearer [rlm0:redacted]"),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b(\s*[:=]\s*)"
            r"[\"']?[^\s\"']{8,}"
        ),
        r"\1\2[rlm0:redacted]",
    ),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        "[rlm0:redacted private key]",
    ),
)


def scrub_secrets(text: str) -> str:
    """Blank out secret shaped substrings on their way back to the model.

    This is a backstop and is described as one. The control that matters is
    that no credential is ever inside the boundary in the first place, which
    is why the sandbox gets a scrubbed environment and no network. But the
    context is attacker controlled and a run may execute on a host that has
    keys in files, so text leaving the boundary is worth a second look before
    it lands in a transcript that gets logged, cached and replayed.

    It will occasionally redact something innocent. That is the correct
    direction to be wrong in.
    """
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
