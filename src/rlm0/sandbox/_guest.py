"""The half of the sandbox that runs inside the boundary.

This file is never imported by the host. It is read as text and shipped over
stdin into a Python that has none of this project installed, which is why it
duplicates the tiny framing helpers from `protocol` instead of importing them:
the alternative to a few duplicated lines is mounting host code into the
container, and a writable path from the host into the sandbox is exactly the
kind of seam the threat model is about.

Two structural decisions carry most of the safety here.

The first is that the protocol channel is moved off file descriptors 0 and 1
before any model written code runs. The model shares this interpreter with the
attacker controlled context, so if the channel stayed on fd 1 then a single
`os.write(1, ...)` could forge a result message, and a single read of fd 0
could steal the host's reply to a sub-call. After the swap, fd 0 and fd 1 point
at the null device and the channel lives on descriptors nothing inside knows
the number of.

The second is that a deadline is raised as a BaseException on a repeating
timer, so that neither `except Exception` nor a loop that swallows everything
can hold the interpreter past its budget. The host still keeps a hard kill in
reserve, because a C level loop will not notice a Python signal handler at all.
"""

from __future__ import annotations

import io
import json
import os
import signal
import sys
import traceback
from types import FrameType
from typing import Any, BinaryIO, Final, TextIO, cast

PROTOCOL_VERSION: Final = 1

_REARM_INTERVAL_S: Final = 0.25
"""How often the deadline re-fires once it has fired.

A single shot alarm is enough for well behaved code and useless against code
that catches it in a loop. Re-arming means every attempt to continue past the
deadline is interrupted again, which turns swallowing the deadline from an
escape into a busy wait that the host's hard kill then ends.
"""


_SET_TIMER: Final = getattr(signal, "setitimer", None)
_GET_TIMER: Final = getattr(signal, "getitimer", None)
_ITIMER_REAL: Final = getattr(signal, "ITIMER_REAL", 0)
_SIGALRM: Final = getattr(signal, "SIGALRM", 0)
"""Looked up rather than imported, because this file runs on two platforms.

The sandbox is a Linux container and the interval timer is always there. The
fallback sandbox runs wherever the developer is, and on Windows none of these
names exist. A `sys.platform` test would be the obvious spelling, but a type
checker then proves one branch dead on whichever platform it happens to run
on, which makes the file impossible to check honestly on both.
"""


class _Deadline(BaseException):
    """The execution budget expired.

    Deliberately not an Exception, so that model written code cannot swallow
    it with the `except Exception` that model written code always has.
    """


class _CappedText(io.TextIOBase):
    """A sink that keeps the first `cap` characters and counts the rest.

    Capping here rather than on the host is what keeps a runaway print from
    becoming a multi gigabyte line on the pipe. The dropped count is kept so
    the host can say how much was lost, which is the part that tells the model
    its approach was wrong rather than merely noisy.
    """

    def __init__(self, cap: int) -> None:
        super().__init__()
        self._cap = max(cap, 0)
        self._parts: list[str] = []
        self._used = 0
        self.dropped = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str, /) -> int:
        room = self._cap - self._used
        if room > 0:
            kept = s[:room]
            self._parts.append(kept)
            self._used += len(kept)
            self.dropped += len(s) - len(kept)
        else:
            self.dropped += len(s)
        return len(s)

    def value(self) -> str:
        return "".join(self._parts)


def _take_channel() -> tuple[BinaryIO, BinaryIO]:
    """Move the protocol off fd 0 and fd 1 and point those at the null device.

    See the module docstring: after this, code running inside cannot forge a
    protocol message or intercept a host reply by writing to a well known
    descriptor number.
    """
    channel_in = os.dup(0)
    channel_out = os.dup(1)
    null = os.open(os.devnull, os.O_RDWR)
    os.dup2(null, 0)
    os.dup2(null, 1)
    os.close(null)
    reader = cast(BinaryIO, os.fdopen(channel_in, "rb"))
    writer = cast(BinaryIO, os.fdopen(channel_out, "wb"))
    return reader, writer


def _as_str(message: dict[str, Any], key: str, default: str = "") -> str:
    value = message.get(key, default)
    return value if isinstance(value, str) else str(value)


def _as_float(message: dict[str, Any], key: str, default: float) -> float:
    value = message.get(key, default)
    try:
        return float(cast(float, value))
    except (TypeError, ValueError):
        return default


def _as_int(message: dict[str, Any], key: str, default: int) -> int:
    value = message.get(key, default)
    try:
        return int(cast(int, value))
    except (TypeError, ValueError):
        return default


class _Guest:
    """One long lived REPL, driven a message at a time."""

    def __init__(self) -> None:
        self._in, self._out = _take_channel()
        self._env: dict[str, Any] = {"__name__": "rlm0_env"}
        self._host_calls: set[str] = set()
        self._call_seq = 0

    def send(self, message: dict[str, Any]) -> None:
        line = json.dumps(message, ensure_ascii=True, separators=(",", ":"))
        self._out.write(line.encode("ascii") + b"\n")
        self._out.flush()

    def read(self) -> dict[str, Any] | None:
        line = self._in.readline()
        if not line:
            return None
        parsed = json.loads(line.decode("utf-8"))
        return parsed if isinstance(parsed, dict) else None

    def run(self) -> None:
        self.send({"kind": "ready", "protocol": PROTOCOL_VERSION})
        while True:
            message = self.read()
            if message is None:
                return
            kind = _as_str(message, "kind")
            if kind == "shutdown":
                return
            if kind == "exec":
                try:
                    self._do_exec(message)
                except _Deadline:
                    # The repeating timer can land in the narrow window
                    # between catching the first one and disarming. Without
                    # this the guest would die there and the host would have
                    # to rebuild the whole environment over a race.
                    self._disarm_deadline()
                    self.send(
                        {
                            "kind": "result",
                            "id": _as_str(message, "id"),
                            "ok": False,
                            "stdout": "",
                            "stdout_dropped": 0,
                            "stderr": "rlm0: execution exceeded its budget.",
                            "stderr_dropped": 0,
                            "variables": self._variables(),
                        }
                    )
            elif kind == "set_var":
                self._do_set_var(message)
            elif kind == "get_var":
                self._do_get_var(message)
            elif kind == "register":
                self._do_register(message)
            elif kind == "variables":
                self.send(
                    {
                        "kind": "names",
                        "id": _as_str(message, "id"),
                        "ok": True,
                        "variables": self._variables(),
                    }
                )
            else:
                self.send(
                    {
                        "kind": "ack",
                        "id": _as_str(message, "id"),
                        "ok": False,
                        "error": f"unknown message kind {kind!r}",
                    }
                )

    def _variables(self) -> list[str]:
        """Names, never values.

        Private names and the injected host call shims are omitted because
        they are the runtime's furniture rather than the model's state.
        """
        return sorted(
            name
            for name in self._env
            if not name.startswith("_") and name not in self._host_calls
        )

    def _do_set_var(self, message: dict[str, Any]) -> None:
        self._env[_as_str(message, "name")] = _as_str(message, "value")
        self.send({"kind": "ack", "id": _as_str(message, "id"), "ok": True})

    def _do_get_var(self, message: dict[str, Any]) -> None:
        name = _as_str(message, "name")
        present = name in self._env
        value = self._env.get(name)
        if present and not isinstance(value, str):
            # A model that built its answer as a list should still be able to
            # get it out, and str() is the coercion it would have written.
            value = str(value)
        self.send(
            {
                "kind": "value",
                "id": _as_str(message, "id"),
                "ok": True,
                "found": present,
                "value": value if present else None,
            }
        )

    def _do_register(self, message: dict[str, Any]) -> None:
        name = _as_str(message, "name")
        arity = _as_int(message, "arity", 0)
        self._env[name] = self._make_host_call(name, arity)
        self._host_calls.add(name)
        self.send({"kind": "ack", "id": _as_str(message, "id"), "ok": True})

    def _make_host_call(self, name: str, arity: int) -> Any:
        """Build the shim that model code calls to reach a host service.

        The shim marshals the call out and blocks. It holds no credential and
        opens no socket, which is the whole reason the network can stay off.
        """

        def call(*args: object) -> object:
            if arity >= 0 and len(args) != arity:
                raise TypeError(f"{name}() takes {arity} arguments, got {len(args)}")
            try:
                payload = json.loads(json.dumps(list(args)))
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    f"{name}() arguments must be JSON serializable: {exc}"
                ) from exc
            self._call_seq += 1
            call_id = f"hc{self._call_seq}"
            remaining = self._pause_deadline()
            try:
                self.send(
                    {
                        "kind": "host_call",
                        "id": call_id,
                        "name": name,
                        "args": payload,
                    }
                )
                reply = self._await_host_result(call_id)
            finally:
                self._resume_deadline(remaining)
            if not reply.get("ok", False):
                raise RuntimeError(f"{name}() failed on the host: {reply.get('error')}")
            return reply.get("value")

        call.__name__ = name
        return call

    def _await_host_result(self, call_id: str) -> dict[str, Any]:
        while True:
            message = self.read()
            if message is None:
                raise RuntimeError("the host closed the channel during a host call")
            if _as_str(message, "kind") != "host_result":
                raise RuntimeError(
                    f"expected host_result, got {_as_str(message, 'kind')!r}"
                )
            if _as_str(message, "id") == call_id:
                return message

    def _on_alarm(self, signum: int, frame: FrameType | None) -> None:
        raise _Deadline

    def _arm_deadline(self, timeout_s: float) -> None:
        """Arm the guest's own deadline where the platform has one.

        Windows has no interval timer, so there the host's hard kill is the
        only bound. That is the worse experience, because the environment does
        not survive it, and it is stated rather than hidden. The default
        sandbox is a Linux container, where this path always exists.
        """
        if _SET_TIMER is None or timeout_s <= 0:
            return
        signal.signal(_SIGALRM, self._on_alarm)
        _SET_TIMER(_ITIMER_REAL, timeout_s, _REARM_INTERVAL_S)

    def _disarm_deadline(self) -> None:
        if _SET_TIMER is not None:
            _SET_TIMER(_ITIMER_REAL, 0.0, 0.0)

    def _pause_deadline(self) -> float:
        """Stop the clock while the host services a sub-call.

        The execution budget bounds the model's code, not the latency of a
        provider, and charging one against the other would make a timeout
        depend on how busy an API was.
        """
        if _SET_TIMER is None or _GET_TIMER is None:
            return 0.0
        remaining, _ = _GET_TIMER(_ITIMER_REAL)
        _SET_TIMER(_ITIMER_REAL, 0.0, 0.0)
        return float(remaining)

    def _resume_deadline(self, remaining: float) -> None:
        if _SET_TIMER is not None and remaining > 0:
            _SET_TIMER(_ITIMER_REAL, remaining, _REARM_INTERVAL_S)

    def _do_exec(self, message: dict[str, Any]) -> None:
        cap = _as_int(message, "stdout_cap", 8000)
        timeout_s = _as_float(message, "timeout_s", 0.0)
        out = _CappedText(cap)
        err = _CappedText(cap)
        saved_out, saved_err = sys.stdout, sys.stderr
        sys.stdout = cast(TextIO, out)
        sys.stderr = cast(TextIO, err)
        ok = True
        self._arm_deadline(timeout_s)
        try:
            compiled = compile(_as_str(message, "code"), "<model-code>", "exec")
            exec(compiled, self._env, self._env)
        except _Deadline:
            ok = False
            err.write(
                f"rlm0: execution exceeded its {timeout_s:g}s budget and was "
                "interrupted. Variables bound before the deadline survive."
            )
        except SystemExit:
            # Model code calling exit() means it finished, not that the whole
            # sandbox should go away and take the context with it.
            pass
        except Exception:
            ok = False
            err.write(traceback.format_exc())
        finally:
            self._disarm_deadline()
            sys.stdout, sys.stderr = saved_out, saved_err
        self.send(
            {
                "kind": "result",
                "id": _as_str(message, "id"),
                "ok": ok,
                "stdout": out.value(),
                "stdout_dropped": out.dropped,
                "stderr": err.value(),
                "stderr_dropped": err.dropped,
                "variables": self._variables(),
            }
        )


def main() -> None:
    """Entry point, called by the one line loader that shipped this source."""
    guest = _Guest()
    try:
        guest.run()
    except Exception as exc:  # pragma: no cover - last resort diagnostic
        guest.send({"kind": "fatal", "error": f"{type(exc).__name__}: {exc}"})
        raise SystemExit(1) from exc
