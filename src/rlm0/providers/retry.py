"""Backoff around one provider call, because a single 429 should not end a run.

Not one of the surveyed implementations of this idea puts retry in the library
layer. The consequence is predictable and was observed: a benchmark sweep that
fans out across a long context hits a rate limit somewhere in the middle, one
sub-call raises, and the whole run dies hours in, having already spent the
money for everything before it.

Two properties matter more than the backoff curve.

The first is that `Retry-After` is honoured when the provider sends it. A
client that ignores it and applies its own exponential curve is guessing
against a number the server already computed, and guessing low turns one rate
limit into several.

The second is that a retried call cannot double-count. That property is
structural here rather than something this module has to remember: a failed
attempt raises and produces no response, so there is nothing to add, and usage
is read exactly once from the single response that finally came back. The
alternative shape, where a caller accumulates usage per attempt inside the
loop, is what makes retry and honest accounting look like they are in tension.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TypeVar

from rlm0.providers.payload import field

__all__ = [
    "RETRYABLE_STATUS_CODES",
    "RetryPolicy",
    "call_with_retry",
    "is_retryable",
    "retry_after_seconds",
]

_log = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
"""Statuses worth trying again.

429 and the 5xx family are the obvious ones. 529 is Anthropic's overloaded
signal. 409 is included because both SDKs classify it as retryable, and 408 and
425 are timeouts on the wire rather than rejections of the request. Everything
else, notably 400 and 401, is a request that will fail identically forever, and
retrying it only spends time before reporting the same error.
"""

_CONNECTION_ERROR_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "APIConnectionTimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
)
"""SDK exception class names treated as retryable when no status is attached.

Matching on class name is unpleasant, and it is here because the SDKs are
optional imports: this package cannot reference `anthropic.APIConnectionError`
without making anthropic mandatory. The status-code check above runs first and
handles everything that carries one, so this is the fallback for transport
failures that never reached a response.
"""


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How many times, how long between, and how long is too long to be told to wait.

    `max_retry_after_s` exists because a provider under load can return a
    Retry-After measured in minutes. Sleeping through it inside a fan-out is
    usually worse than failing the sub-call and letting the run layer decide,
    since the run layer is the thing that holds a wall-clock budget.
    """

    max_attempts: int = 4
    initial_backoff_s: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff_s: float = 30.0
    respect_retry_after: bool = True
    max_retry_after_s: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_backoff_s < 0:
            raise ValueError("initial_backoff_s cannot be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.max_backoff_s < 0:
            raise ValueError("max_backoff_s cannot be negative")
        if self.max_retry_after_s < 0:
            raise ValueError("max_retry_after_s cannot be negative")

    def backoff_for(self, attempt: int) -> float:
        """Seconds to wait after a failed attempt, counting attempts from 1."""
        delay = self.initial_backoff_s * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_backoff_s)


def _status_code(exc: BaseException) -> int | None:
    """The HTTP status on an SDK exception, whether it is direct or on a response."""
    raw = field(exc, "status_code", None)
    if raw is None:
        response = field(exc, "response", None)
        if response is not None:
            raw = field(response, "status_code", None)
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    return raw


def _headers(exc: BaseException) -> Mapping[str, str] | None:
    """Response headers off an SDK exception, if it carries a response at all."""
    for holder in (exc, field(exc, "response", None)):
        if holder is None:
            continue
        headers = field(holder, "headers", None)
        if isinstance(headers, Mapping):
            return {str(k).lower(): str(v) for k, v in headers.items()}
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """How long the provider asked us to wait, in seconds, or None if it did not.

    Both the seconds form and the millisecond form are read, because providers
    send both and the millisecond header is the more precise of the two when it
    is present. HTTP-date values are deliberately not parsed: they are rare on
    these APIs, and a wrong clock would turn a two second wait into a wait
    measured against somebody else's idea of now.
    """
    headers = _headers(exc)
    if headers is None:
        return None
    milliseconds = headers.get("retry-after-ms")
    if milliseconds is not None:
        try:
            return max(float(milliseconds) / 1000.0, 0.0)
        except ValueError:
            pass
    seconds = headers.get("retry-after")
    if seconds is not None:
        try:
            return max(float(seconds), 0.0)
        except ValueError:
            return None
    return None


def is_retryable(exc: BaseException) -> bool:
    """Whether trying the identical request again could plausibly succeed."""
    status = _status_code(exc)
    if status is not None:
        return status in RETRYABLE_STATUS_CODES
    if type(exc).__name__ in _CONNECTION_ERROR_NAMES:
        return True
    return isinstance(exc, TimeoutError | ConnectionError)


def call_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    retryable: Callable[[BaseException], bool] = is_retryable,
) -> T:
    """Run one operation, retrying transient failures until the policy runs out.

    `sleep` is injected rather than called directly so that tests can assert
    the delays without spending them. A retry test that actually sleeps is a
    test people delete.

    The operation is called for its result and nothing is accumulated across
    attempts, which is what keeps usage single-counted: the caller reads usage
    from the value this returns, and the values from failed attempts do not
    exist to be read.
    """
    last_error: Exception | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as exc:
            if not retryable(exc) or attempt == policy.max_attempts:
                raise
            last_error = exc
            delay = policy.backoff_for(attempt)
            if policy.respect_retry_after:
                asked = retry_after_seconds(exc)
                if asked is not None:
                    if asked > policy.max_retry_after_s:
                        raise
                    delay = asked
            _log.warning(
                "provider call failed with %s on attempt %d of %d; "
                "retrying in %.2fs",
                type(exc).__name__,
                attempt,
                policy.max_attempts,
                delay,
            )
            if delay > 0:
                sleep(delay)
    # Unreachable: the loop either returns, or raises on the final attempt.
    raise AssertionError("retry loop exited without returning") from last_error
