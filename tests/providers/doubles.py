"""Doubles for the SDK boundary, so nothing here needs a key, a net, or an SDK.

Mocking is done at the outermost seam, the `client.messages.create` and
`client.chat.completions.create` calls, and responses are plain dicts. Both
choices are deliberate. Mocking further in would test our own wrapper against
itself, and building response objects that mirror the SDK's class hierarchy
would encode this author's belief about that hierarchy rather than the
hierarchy, which is the usual way a mocked test passes against an API that
changed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class StubHTTPError(Exception):
    """An SDK-shaped failure: a status code, and optionally response headers."""

    def __init__(
        self,
        status_code: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(f"stub http error status={status_code}")
        if status_code is not None:
            self.status_code = status_code
        if headers is not None:
            self.response = _StubResponse(headers)


class _StubResponse:
    def __init__(self, headers: Mapping[str, str]) -> None:
        self.headers = dict(headers)


class RecordingCreate:
    """A `create` callable that records its kwargs and serves scripted results."""

    def __init__(self, results: Sequence[object]) -> None:
        self._results = list(results)
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> object:
        self.requests.append(dict(kwargs))
        if not self._results:
            raise AssertionError("stub SDK called more times than it was scripted")
        result = self._results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def last_request(self) -> dict[str, Any]:
        return self.requests[-1]


class StubAnthropicSDK:
    """Shaped like `anthropic.Anthropic()`: a `.messages.create`."""

    def __init__(self, results: Sequence[object]) -> None:
        self.create = RecordingCreate(results)
        self.messages = _Namespace(create=self.create)


class StubOpenAISDK:
    """Shaped like `openai.OpenAI()`: a `.chat.completions.create`."""

    def __init__(self, results: Sequence[object]) -> None:
        self.create = RecordingCreate(results)
        self.chat = _Namespace(completions=_Namespace(create=self.create))


class _Namespace:
    def __init__(self, **attributes: Any) -> None:
        for name, value in attributes.items():
            setattr(self, name, value)


class RecordingSleep:
    """Stands in for `time.sleep`, so a retry test costs nothing to run."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def anthropic_response(
    *,
    text: str = "answer",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 1000,
    output_tokens: int = 50,
    cache_read: int | None = None,
    cache_write: int | None = None,
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    """A Messages API response body.

    `cache_read` and `cache_write` default to absent rather than zero, because
    a response that omits the fields entirely is the case that has to leave the
    cache counters at zero without raising.
    """
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    return {
        "model": model,
        "stop_reason": stop_reason,
        "content": [{"type": "text", "text": text}],
        "usage": usage,
    }


def openai_response(
    *,
    text: str = "answer",
    model: str = "gpt-4o",
    prompt_tokens: int = 1000,
    completion_tokens: int = 50,
    cached_tokens: int | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    """A chat completions response body."""
    usage: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if cached_tokens is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
    return {
        "model": model,
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": usage,
    }
