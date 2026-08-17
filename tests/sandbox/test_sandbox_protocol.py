"""The wire format, tested apart from anything that spawns a process."""

from __future__ import annotations

import pytest

from rlm0.sandbox.protocol import (
    STDOUT_CHAR_CAP,
    ProtocolError,
    decode,
    encode,
    scrub_secrets,
    truncation_notice,
)


def test_a_message_is_exactly_one_line_even_with_newlines_inside() -> None:
    frame = encode({"kind": "exec", "code": "print('a')\nprint('b')\n"})
    assert frame.count(b"\n") == 1
    assert frame.endswith(b"\n")
    assert decode(frame)["code"] == "print('a')\nprint('b')\n"


def test_frames_stay_ascii_whatever_the_context_contained() -> None:
    frame = encode({"kind": "set_var", "value": "你好 \U0001f600"})
    frame.decode("ascii")
    assert decode(frame)["value"] == "你好 \U0001f600"


def test_a_frame_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(ProtocolError):
        decode(b"[1, 2, 3]\n")
    with pytest.raises(ProtocolError):
        decode(b"not json at all\n")


def test_scrubbing_catches_the_common_credential_shapes() -> None:
    text = (
        "key sk-abcdefghijklmnopqrstuvwx here\n"
        "aws AKIAIOSFODNN7EXAMPLE here\n"
        "api_key=hunter2hunter2 here\n"
        "Authorization: Bearer abcdefghijklmnopqrstuvwxyz123\n"
    )
    scrubbed = scrub_secrets(text)
    assert "sk-abcdefghijklmnopqrstuvwx" not in scrubbed
    assert "AKIAIOSFODNN7EXAMPLE" not in scrubbed
    assert "hunter2hunter2" not in scrubbed
    assert "abcdefghijklmnopqrstuvwxyz123" not in scrubbed
    assert "redacted" in scrubbed


def test_scrubbing_leaves_ordinary_prose_alone() -> None:
    text = "The document mentions a key finding on page 12 about tokens."
    assert scrub_secrets(text) == text


def test_the_truncation_message_teaches_rather_than_scolds() -> None:
    notice = truncation_notice(cap=STDOUT_CHAR_CAP, dropped=1234)
    assert "sub-call" in notice
    assert "1234" in notice
    assert str(STDOUT_CHAR_CAP) in notice
