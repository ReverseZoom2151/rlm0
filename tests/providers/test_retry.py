"""Retry: honours the server's number, gives up eventually, and never doubles up."""

from __future__ import annotations

import pytest
from doubles import RecordingSleep, StubHTTPError

from rlm0.providers import RetryPolicy, call_with_retry, is_retryable
from rlm0.providers.retry import retry_after_seconds


def test_retry_after_header_beats_the_local_backoff_curve() -> None:
    """A guessed exponential delay is a guess against a number the server sent."""
    sleep = RecordingSleep()
    attempts: list[int] = []

    def operation() -> str:
        attempts.append(1)
        if len(attempts) == 1:
            raise StubHTTPError(429, {"retry-after": "7"})
        return "ok"

    result = call_with_retry(
        operation,
        policy=RetryPolicy(initial_backoff_s=0.5),
        sleep=sleep,
    )
    assert result == "ok"
    assert sleep.delays == [7.0]


def test_retry_after_ms_header_is_read_too() -> None:
    sleep = RecordingSleep()
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) == 1:
            raise StubHTTPError(429, {"retry-after-ms": "250"})
        return "ok"

    call_with_retry(operation, policy=RetryPolicy(), sleep=sleep)
    assert sleep.delays == [0.25]


def test_backoff_is_exponential_when_no_header_is_sent() -> None:
    sleep = RecordingSleep()
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        if len(calls) < 4:
            raise StubHTTPError(503)
        return "ok"

    call_with_retry(
        operation,
        policy=RetryPolicy(
            max_attempts=4, initial_backoff_s=0.5, backoff_multiplier=2.0
        ),
        sleep=sleep,
    )
    assert sleep.delays == [0.5, 1.0, 2.0]


def test_attempts_are_capped() -> None:
    sleep = RecordingSleep()
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        raise StubHTTPError(429)

    with pytest.raises(StubHTTPError):
        call_with_retry(operation, policy=RetryPolicy(max_attempts=3), sleep=sleep)
    assert len(calls) == 3
    assert len(sleep.delays) == 2


def test_a_permanent_failure_is_not_retried() -> None:
    """A 400 will fail identically forever; retrying only delays the report."""
    sleep = RecordingSleep()
    calls: list[int] = []

    def operation() -> str:
        calls.append(1)
        raise StubHTTPError(400)

    with pytest.raises(StubHTTPError):
        call_with_retry(operation, policy=RetryPolicy(max_attempts=5), sleep=sleep)
    assert len(calls) == 1
    assert sleep.delays == []


def test_an_absurd_retry_after_is_refused_rather_than_slept_through() -> None:
    """Blocking a fan-out for ten minutes is the run layer's decision, not ours."""
    sleep = RecordingSleep()

    def operation() -> str:
        raise StubHTTPError(429, {"retry-after": "600"})

    with pytest.raises(StubHTTPError):
        call_with_retry(
            operation,
            policy=RetryPolicy(max_attempts=5, max_retry_after_s=60.0),
            sleep=sleep,
        )
    assert sleep.delays == []


def test_first_success_returns_without_sleeping() -> None:
    sleep = RecordingSleep()
    assert call_with_retry(lambda: 42, policy=RetryPolicy(), sleep=sleep) == 42
    assert sleep.delays == []


def test_retryable_classification() -> None:
    assert is_retryable(StubHTTPError(429))
    assert is_retryable(StubHTTPError(529))
    assert is_retryable(StubHTTPError(500))
    assert not is_retryable(StubHTTPError(400))
    assert not is_retryable(StubHTTPError(401))
    assert is_retryable(TimeoutError("no response"))
    assert not is_retryable(ValueError("a bug in our own code"))


def test_a_malformed_retry_after_falls_back_to_the_curve() -> None:
    assert retry_after_seconds(StubHTTPError(429, {"retry-after": "soon"})) is None
    assert retry_after_seconds(StubHTTPError(429)) is None


def test_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="backoff_multiplier"):
        RetryPolicy(backoff_multiplier=0.5)
