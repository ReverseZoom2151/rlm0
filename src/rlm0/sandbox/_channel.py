"""The host half, shared by both sandboxes.

Everything here is about the two ends staying honest with each other while one
of them is assumed hostile. Three properties are worth naming.

Messages are correlated by id. The surveyed implementation this design started
from ran one request at a time with no correlation field and skipped any line
it could not parse, which means a desynchronized stream degrades into plausible
looking wrong answers instead of an error.

Reads happen on a thread feeding a queue, so a deadline is a `queue.get`
timeout rather than a blocking read the host cannot get out of. This is the
part that makes `while True:` in model written code survivable on every
platform rather than only where SIGALRM exists.

Host calls extend the deadline by the time the host itself spent. The budget
bounds the model's code; charging a provider's latency against it would make
timeouts depend on how busy an API was that afternoon.
"""

from __future__ import annotations

import contextlib
import itertools
import queue
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from rlm0.ports import ExecResult, SandboxUnavailableError
from rlm0.sandbox import cleanup
from rlm0.sandbox.protocol import (
    PROTOCOL_VERSION,
    STDOUT_CHAR_CAP,
    HostCallable,
    ProtocolError,
    decode,
    encode,
    scrub_secrets,
    truncation_notice,
)

__all__ = ["ChannelSandbox"]

_GUEST_SOURCE_PATH: Final = Path(__file__).with_name("_guest.py")

LOADER: Final = (
    "import sys;"
    "_n=int(sys.stdin.buffer.readline());"
    '_g={"__name__":"rlm0_guest"};'
    'exec(compile(sys.stdin.buffer.read(_n).decode("utf-8"),"rlm0_guest",'
    '"exec"),_g);'
    '_g["main"]()'
)
"""A loader small enough to pass as an argument, so the guest can be large.

The guest arrives on stdin rather than on the command line because argument
length limits are low on Windows and because a mounted host path into the
sandbox is a seam this design refuses to open.
"""

_HARD_GRACE_S: Final = 2.0
"""Extra time before the host stops asking and starts killing.

The guest interrupts itself with a repeating signal, which handles a Python
level loop. It cannot handle a loop inside a C extension, because that never
reaches a bytecode boundary. So the host waits a little past the guest's own
deadline and then kills, which is the only bound that holds unconditionally.
"""

_CONTROL_TIMEOUT_S: Final = 30.0
_CONTROL_SECONDS_PER_MB: Final = 1.0
_LARGE_TRANSFER_TIMEOUT_S: Final = 300.0


class SandboxCrashedError(RuntimeError):
    """The child went away mid conversation."""


@dataclass(slots=True)
class Child:
    """One running interpreter and the way to end it hard."""

    proc: subprocess.Popen[bytes]
    terminate: Callable[[], None]
    label: str = ""


class ChannelSandbox:
    """A Sandbox implemented as a long lived child speaking framed JSON.

    Subclasses supply `_spawn`. Everything about framing, deadlines, host
    calls, truncation and restart lives here so that the Docker and subprocess
    implementations cannot drift apart in the parts that matter.
    """

    def __init__(
        self,
        *,
        host_calls: Mapping[str, HostCallable] | None = None,
        stdout_cap: int = STDOUT_CHAR_CAP,
        startup_timeout_s: float = 30.0,
    ) -> None:
        self._host_calls: dict[str, HostCallable] = dict(host_calls or {})
        self._registered: dict[str, int] = {}
        self._stdout_cap = stdout_cap
        self._startup_timeout_s = startup_timeout_s
        self._ids = itertools.count(1)
        self._child: Child | None = None
        self._inbox: queue.Queue[bytes | None] = queue.Queue()
        self._child_stderr: deque[str] = deque(maxlen=64)
        self._closed = False

    # Construction and teardown

    def _spawn(self) -> Child:
        raise NotImplementedError

    def _start(self) -> None:
        """Bring a child up and complete the handshake, or fail loudly.

        The handshake is the probe. A surveyed repo shipped a `check_health`
        method that nothing ever called, so its documented fail closed path
        could never fire; here the only way to have a sandbox object is to
        have watched a real child say hello.
        """
        self._inbox = queue.Queue()
        self._child_stderr = deque(maxlen=64)
        child = self._spawn()
        self._child = child
        threading.Thread(
            target=self._pump_stdout, args=(child,), daemon=True
        ).start()
        threading.Thread(
            target=self._pump_stderr, args=(child,), daemon=True
        ).start()
        source = _GUEST_SOURCE_PATH.read_bytes()
        try:
            self._write(f"{len(source)}\n".encode() + source)
        except SandboxCrashedError as exc:
            self._kill()
            raise SandboxUnavailableError(
                f"{child.label or 'sandbox'} died before it could be loaded: "
                f"{exc}. {self._stderr_tail()}"
            ) from exc
        deadline = time.monotonic() + self._startup_timeout_s
        try:
            hello = self._recv(deadline)
        except SandboxCrashedError as exc:
            self._kill()
            raise SandboxUnavailableError(
                f"{child.label or 'sandbox'} exited during startup: {exc}. "
                f"{self._stderr_tail()}"
            ) from exc
        if hello is None or hello.get("kind") != "ready":
            self._kill()
            raise SandboxUnavailableError(
                f"{child.label or 'sandbox'} never completed the handshake "
                f"within {self._startup_timeout_s:g}s. {self._stderr_tail()}"
            )
        if hello.get("protocol") != PROTOCOL_VERSION:
            self._kill()
            raise SandboxUnavailableError(
                f"guest speaks protocol {hello.get('protocol')!r}, host speaks "
                f"{PROTOCOL_VERSION}"
            )
        cleanup.register(self)

    def close(self) -> None:
        """Safe to call twice, and called for you on exit.

        Idempotence is not politeness here: close runs from `__del__`, from an
        atexit hook and from a signal handler, and those three can and do race
        on the same object.
        """
        if self._closed:
            return
        self._closed = True
        cleanup.unregister(self)
        child = self._child
        if child is not None:
            try:
                self._write(encode({"kind": "shutdown"}))
                child.proc.wait(timeout=2.0)
            except (SandboxCrashedError, subprocess.TimeoutExpired, OSError):
                pass
            self._kill()
        self._child = None

    def _kill(self) -> None:
        child = self._child
        if child is None:
            return
        with contextlib.suppress(Exception):
            child.terminate()
        for stream in (child.proc.stdin, child.proc.stdout, child.proc.stderr):
            if stream is not None:
                with contextlib.suppress(OSError):
                    stream.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter teardown
        with contextlib.suppress(Exception):
            self.close()

    # Framing

    def _pump_stdout(self, child: Child) -> None:
        stream = child.proc.stdout
        if stream is None:  # pragma: no cover - always piped
            return
        try:
            for line in iter(stream.readline, b""):
                self._inbox.put(line)
        except (OSError, ValueError):  # pragma: no cover - closed under us
            pass
        finally:
            self._inbox.put(None)

    def _pump_stderr(self, child: Child) -> None:
        stream = child.proc.stderr
        if stream is None:  # pragma: no cover - always piped
            return
        try:
            for line in iter(stream.readline, b""):
                self._child_stderr.append(line.decode("utf-8", "replace").rstrip())
        except (OSError, ValueError):  # pragma: no cover - closed under us
            pass

    def _stderr_tail(self) -> str:
        if not self._child_stderr:
            return "no diagnostics on the child's stderr"
        return "child stderr: " + " | ".join(self._child_stderr)

    def _write(self, payload: bytes) -> None:
        child = self._child
        if child is None or child.proc.stdin is None:
            raise SandboxCrashedError("no child to write to")
        try:
            child.proc.stdin.write(payload)
            child.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise SandboxCrashedError(f"write failed: {exc}") from exc

    def _send(self, message: Mapping[str, Any]) -> None:
        self._write(encode(message))

    def _recv(self, deadline: float) -> dict[str, Any] | None:
        """One message, or None when the deadline passed."""
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            line = self._inbox.get(timeout=remaining)
        except queue.Empty:
            return None
        if line is None:
            raise SandboxCrashedError("the child closed its output stream")
        return decode(line)

    # The Sandbox protocol

    def execute(self, code: str, *, timeout_s: float) -> ExecResult:
        """Run one block, and return whatever happens.

        Never raises on model error: a traceback is a result the model should
        see and act on, not an exception for the orchestrator to handle. It
        does raise if the sandbox itself cannot be kept alive.
        """
        self._require_open()
        started = time.monotonic()
        request_id = self._next_id()
        try:
            self._send(
                {
                    "kind": "exec",
                    "id": request_id,
                    "code": code,
                    "timeout_s": timeout_s,
                    "stdout_cap": self._stdout_cap,
                }
            )
            payload = self._pump_until_result(request_id, timeout_s)
        except (SandboxCrashedError, ProtocolError) as exc:
            return self._restart_after(exc, started)
        if payload is None:
            return self._restart_after(
                SandboxCrashedError(
                    f"code ran past its {timeout_s:g}s budget and past the "
                    f"{_HARD_GRACE_S:g}s grace period, so the sandbox was "
                    "killed. State did not survive."
                ),
                started,
                timed_out=True,
            )
        return self._to_result(payload, time.monotonic() - started)

    def _pump_until_result(
        self, request_id: str, timeout_s: float
    ) -> dict[str, Any] | None:
        deadline = time.monotonic() + max(timeout_s, 0.0) + _HARD_GRACE_S
        while True:
            message = self._recv(deadline)
            if message is None:
                return None
            kind = message.get("kind")
            if kind == "host_call":
                spent = self._service_host_call(message)
                deadline += spent
                continue
            if kind == "result" and message.get("id") == request_id:
                return message
            if kind == "fatal":
                raise SandboxCrashedError(str(message.get("error")))
            raise ProtocolError(f"unexpected message during execute: {kind!r}")

    def _service_host_call(self, message: Mapping[str, Any]) -> float:
        """Run one sub-call on the host and hand back only its result.

        This is the whole reason the sandbox can keep its network off. The
        credential, the client and the socket all stay here; what crosses the
        boundary is a list of JSON values in and one JSON value out.
        """
        began = time.monotonic()
        name = str(message.get("name"))
        raw_args = message.get("args")
        args = list(raw_args) if isinstance(raw_args, list) else []
        reply: dict[str, Any] = {"kind": "host_result", "id": message.get("id")}
        function = self._host_calls.get(name)
        if function is None:
            reply["ok"] = False
            reply["error"] = f"{name} is not a registered host call"
        else:
            try:
                reply["value"] = function(*args)
                reply["ok"] = True
            except Exception as exc:
                reply["ok"] = False
                reply["error"] = f"{type(exc).__name__}: {exc}"
        try:
            self._send(reply)
        except (TypeError, ValueError) as exc:
            self._send(
                {
                    "kind": "host_result",
                    "id": message.get("id"),
                    "ok": False,
                    "error": f"{name} returned something JSON cannot carry: {exc}",
                }
            )
        return time.monotonic() - began

    def set_variable(self, name: str, value: str) -> None:
        """Bind a large string inside without it passing through the model."""
        self._require_open()
        budget = _CONTROL_TIMEOUT_S + len(value) / 1_000_000 * _CONTROL_SECONDS_PER_MB
        self._round_trip(
            {"kind": "set_var", "name": name, "value": value},
            expect="ack",
            timeout_s=budget,
        )

    def get_variable(self, name: str) -> str | None:
        """Read one value out by name.

        This is how an answer longer than a context window leaves the runtime:
        the model writes it into a variable and names the variable, and the
        caller reads the value without any of it being generated twice or
        travelling through a window.
        """
        self._require_open()
        reply = self._round_trip(
            {"kind": "get_var", "name": name},
            expect="value",
            # Generous, because the value being fetched is by design the one
            # too large to have travelled through a context window, and a
            # dead guest is detected by EOF rather than by this timeout.
            timeout_s=_LARGE_TRANSFER_TIMEOUT_S,
        )
        if not reply.get("found"):
            return None
        value = reply.get("value")
        return value if isinstance(value, str) else None

    def variables(self) -> tuple[str, ...]:
        """Names bound inside. Names only, never values."""
        self._require_open()
        reply = self._round_trip(
            {"kind": "variables"}, expect="names", timeout_s=_CONTROL_TIMEOUT_S
        )
        names = reply.get("variables")
        return tuple(str(n) for n in names) if isinstance(names, list) else ()

    def bind_host_call(self, name: str, function: HostCallable) -> None:
        """Give the host a callable to service `name` with.

        Kept separate from `register_host_call` because the Sandbox protocol
        deliberately passes only a name and an arity across the seam. The
        sandbox is told what may be called; it is never handed the means to
        make the call, which is what keeps the key on this side.
        """
        self._host_calls[name] = function

    def register_host_call(self, name: str, arity: int) -> None:
        """Expose a host serviced function to code running inside."""
        self._require_open()
        if name not in self._host_calls:
            raise ValueError(
                f"no host implementation is bound for {name!r}; bind it with "
                "bind_host_call or pass host_calls at construction. Exposing a "
                "name the host cannot service would fail inside the sandbox, "
                "where the model would read it as its own mistake."
            )
        self._round_trip(
            {"kind": "register", "name": name, "arity": arity},
            expect="ack",
            timeout_s=_CONTROL_TIMEOUT_S,
        )
        self._registered[name] = arity

    # Helpers

    def _round_trip(
        self, message: dict[str, Any], *, expect: str, timeout_s: float
    ) -> dict[str, Any]:
        request_id = self._next_id()
        message["id"] = request_id
        self._send(message)
        deadline = time.monotonic() + timeout_s
        while True:
            reply = self._recv(deadline)
            if reply is None:
                raise SandboxCrashedError(
                    f"no {expect} within {timeout_s:g}s. {self._stderr_tail()}"
                )
            if reply.get("kind") == expect and reply.get("id") == request_id:
                if not reply.get("ok", False):
                    raise ProtocolError(str(reply.get("error", "guest refused")))
                return reply
            raise ProtocolError(
                f"expected {expect}, got {reply.get('kind')!r}"
            )

    def _next_id(self) -> str:
        return f"h{next(self._ids)}"

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("this sandbox is closed")
        if self._child is None:
            raise RuntimeError("this sandbox was never started")

    def _to_result(self, payload: Mapping[str, Any], elapsed: float) -> ExecResult:
        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        dropped = int(payload.get("stdout_dropped", 0) or 0)
        if dropped:
            stdout += truncation_notice(cap=self._stdout_cap, dropped=dropped)
        stderr_dropped = int(payload.get("stderr_dropped", 0) or 0)
        if stderr_dropped:
            stderr += f"\n[rlm0: {stderr_dropped} further stderr chars dropped]"
        names = payload.get("variables")
        return ExecResult(
            stdout=scrub_secrets(stdout),
            stderr=scrub_secrets(stderr),
            wall_clock_s=elapsed,
            ok=bool(payload.get("ok", False)),
            truncated_stdout=dropped > 0,
            variables=tuple(str(n) for n in names) if isinstance(names, list) else (),
        )

    def _restart_after(
        self, exc: Exception, started: float, *, timed_out: bool = False
    ) -> ExecResult:
        """Kill what is left, bring a fresh child up, and say so plainly.

        Returning a failed ExecResult rather than raising keeps a hostile or
        merely careless block from ending the run, but the message says the
        state is gone. Silently continuing with an empty environment would let
        a later step read a missing variable as an empty one.
        """
        self._kill()
        note = str(exc)
        try:
            self._start()
        except SandboxUnavailableError as restart_exc:
            self._closed = True
            raise SandboxUnavailableError(
                f"{note} The sandbox could not be restarted: {restart_exc}"
            ) from restart_exc
        for name, arity in list(self._registered.items()):
            self._round_trip(
                {"kind": "register", "name": name, "arity": arity},
                expect="ack",
                timeout_s=_CONTROL_TIMEOUT_S,
            )
        detail = (
            f"rlm0: {note} A fresh sandbox has been started and every variable "
            "bound before this point is gone. Re-seed the context before "
            "continuing."
        )
        if timed_out:
            detail += (
                " A block that does not finish is usually one that is walking "
                "the whole context in Python; slice it and recurse instead."
            )
        return ExecResult(
            stdout="",
            stderr=detail,
            wall_clock_s=time.monotonic() - started,
            ok=False,
            variables=(),
        )
