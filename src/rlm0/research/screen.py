"""Conservative context screening for experimental research strategies.

This is a small, deliberately boring seam for checks inspired by RLM-JB-style
context screening.  It does not execute code, inspect a sandbox, or make a
sandbox unnecessary.  It only turns deterministic, injected checks into a
strict three-state decision that callers can audit.

``UNKNOWN`` is intentional.  A malformed model response, a checker exception,
or a missing checker is not evidence that a context is safe.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

__all__ = [
    "ContextCheck",
    "ScreenReport",
    "ScreenResult",
    "ScreenVerdict",
    "parse_screen_response",
    "screen_context",
]


class ScreenVerdict(StrEnum):
    """The only outcomes a context screen may report."""

    SAFE = "safe"
    UNSAFE = "unsafe"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScreenResult:
    """The result from one named context check."""

    verdict: ScreenVerdict
    checker: str
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.checker.strip():
            raise ValueError("checker must name the source of a screen result")


@dataclass(frozen=True, slots=True)
class ScreenReport:
    """An auditable aggregate decision.

    The aggregation is conservative: one ``UNSAFE`` result blocks the context;
    otherwise one ``UNKNOWN`` result keeps the decision unknown.  Only an
    nonempty all-safe set is safe.
    """

    verdict: ScreenVerdict
    results: tuple[ScreenResult, ...]

    def __post_init__(self) -> None:
        expected = _aggregate(result.verdict for result in self.results)
        if self.verdict is not expected:
            raise ValueError(
                f"report verdict {self.verdict.value!r} disagrees with "
                f"its evidence ({expected.value!r})"
            )

    @property
    def is_approved(self) -> bool:
        """Whether all configured checks positively marked the context safe."""

        return self.verdict is ScreenVerdict.SAFE


# The public extension seam intentionally permits an arbitrary return. A
# malformed checker must result in UNKNOWN at runtime, not be excluded only by
# static typing.
ContextCheck: TypeAlias = Callable[[str], object]


def parse_screen_response(response: str, *, checker: str) -> ScreenResult:
    """Parse a strict, version-free JSON checker response.

    The accepted shape is exactly ``{"verdict": "safe|unsafe|unknown",
    "detail": "..."}``; ``detail`` may be omitted.  Every parse or shape
    failure becomes ``UNKNOWN`` so a caller cannot accidentally treat a broken
    checker as approval.
    """

    try:
        decoded = json.loads(response)
    except (TypeError, json.JSONDecodeError):
        return _unknown(checker, "checker returned malformed JSON")
    if not isinstance(decoded, dict):
        return _unknown(checker, "checker response must be a JSON object")
    if set(decoded) - {"verdict", "detail"} or "verdict" not in decoded:
        return _unknown(checker, "checker response has an unsupported shape")
    raw_verdict = decoded["verdict"]
    if not isinstance(raw_verdict, str):
        return _unknown(checker, "checker verdict must be a string")
    try:
        verdict = ScreenVerdict(raw_verdict)
    except ValueError:
        return _unknown(checker, "checker verdict is not recognised")
    detail = decoded.get("detail", "")
    if not isinstance(detail, str):
        return _unknown(checker, "checker detail must be a string")
    return ScreenResult(verdict=verdict, checker=checker, detail=detail)


def screen_context(context: str, checks: Iterable[ContextCheck]) -> ScreenReport:
    """Run injected deterministic checks and conservatively aggregate them.

    Checks do not receive host credentials or a sandbox handle.  Their only
    input is the supplied context string.  Exceptions and malformed return
    values are retained in the report as ``UNKNOWN`` evidence.
    """

    if not isinstance(context, str):
        raise TypeError("context must be a string")

    results: list[ScreenResult] = []
    for index, check in enumerate(checks):
        checker = _checker_name(check, index)
        try:
            returned = check(context)
        except Exception as exc:  # Checks are untrusted extension points.
            results.append(_unknown(checker, f"checker raised {type(exc).__name__}"))
            continue
        if isinstance(returned, ScreenResult):
            results.append(returned)
        elif isinstance(returned, str):
            results.append(parse_screen_response(returned, checker=checker))
        else:
            results.append(_unknown(checker, "checker returned an unsupported value"))

    frozen_results = tuple(results)
    return ScreenReport(
        verdict=_aggregate(result.verdict for result in frozen_results),
        results=frozen_results,
    )


def _aggregate(verdicts: Iterable[ScreenVerdict]) -> ScreenVerdict:
    values = tuple(verdicts)
    if ScreenVerdict.UNSAFE in values:
        return ScreenVerdict.UNSAFE
    if not values or ScreenVerdict.UNKNOWN in values:
        return ScreenVerdict.UNKNOWN
    return ScreenVerdict.SAFE


def _checker_name(check: ContextCheck, index: int) -> str:
    name = getattr(check, "__name__", "")
    if isinstance(name, str) and name.strip() and name != "<lambda>":
        return name
    return f"check_{index}"


def _unknown(checker: str, detail: str) -> ScreenResult:
    return ScreenResult(ScreenVerdict.UNKNOWN, checker, detail)
