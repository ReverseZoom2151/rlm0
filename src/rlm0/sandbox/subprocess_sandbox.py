"""A sandbox for machines without Docker, and an honest label on it.

Opt in only, and never selected for you, because the difference between this
and the Docker sandbox is not performance or convenience. It is whether there
is a boundary at all.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from typing import Final

from rlm0.ports import SandboxUnavailableError
from rlm0.sandbox._channel import LOADER, ChannelSandbox, Child
from rlm0.sandbox.protocol import STDOUT_CHAR_CAP, HostCallable

__all__ = ["SubprocessSandbox"]

_ENV_KEEP: Final = (
    "PATH",
    "SYSTEMROOT",
    "COMSPEC",
    "WINDIR",
    "LANG",
    "LC_ALL",
    "TZ",
)
"""Everything else in the environment is dropped on the way in.

An allowlist rather than a denylist of key looking names, because the whole
point is to be right about the variable nobody thought of. Provider keys,
cloud credentials, proxy URLs with basic auth in them and CI tokens all live
in the environment, and the sandbox has no use for any of them: sub-calls are
serviced by the host, which keeps its own environment.
"""


class SubprocessSandbox(ChannelSandbox):
    """A separate Python process. NOT a security boundary.

    Stated plainly because the temptation to treat it as one is the failure
    this project is arguing against. This sandbox runs as the same user, on
    the same filesystem, with the same network access as the orchestrator.
    Code running inside it can read your home directory, read your key files,
    open sockets and delete your work. Against an attacker controlled context,
    which is the case this runtime is built for, it protects nothing.

    What it does provide is real and worth having on a machine with no Docker:
    it isolates the orchestrator from crashes, from a segfaulting extension,
    from memory exhaustion in model written code and from a `while True` that
    would otherwise wedge the process, because the whole interpreter can be
    killed and replaced without the run dying. It also holds the same wire
    protocol as the Docker sandbox, so nothing above it can tell which one it
    got, and it opens no network connection of its own.

    Use it for development, for tests, and for contexts you wrote yourself.
    Do not use it for a context that arrived from somewhere else.
    """

    def __init__(
        self,
        *,
        host_calls: Mapping[str, HostCallable] | None = None,
        stdout_cap: int = STDOUT_CHAR_CAP,
        startup_timeout_s: float = 30.0,
        python: str | None = None,
    ) -> None:
        super().__init__(
            host_calls=host_calls,
            stdout_cap=stdout_cap,
            startup_timeout_s=startup_timeout_s,
        )
        self._python = python or sys.executable
        if not self._python:
            raise SandboxUnavailableError(
                "no Python interpreter to spawn; sys.executable is empty and "
                "no python= was given"
            )
        self._workdir = tempfile.TemporaryDirectory(prefix="rlm0-sbx-")
        self._start()

    def _spawn(self) -> Child:
        argv = [self._python, "-I", "-u", "-c", LOADER]
        env = {k: os.environ[k] for k in _ENV_KEEP if k in os.environ}
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # Its own session on POSIX, so a runaway that forked children can be
        # killed as a group rather than left orphaned holding the CPU.
        new_session = sys.platform != "win32"
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._workdir.name,
                env=env,
                start_new_session=new_session,
            )
        except OSError as exc:
            raise SandboxUnavailableError(
                f"could not spawn {self._python!r}: {exc}"
            ) from exc
        return Child(proc=proc, terminate=_killer(proc), label="subprocess sandbox")

    def close(self) -> None:
        super().close()
        # Windows keeps handles open a moment after the child dies, so a
        # failure here is a temp directory to sweep, not a run to fail.
        with contextlib.suppress(OSError):
            self._workdir.cleanup()


def _killer(proc: subprocess.Popen[bytes]) -> Callable[[], None]:
    """Kill the whole group where the platform has one.

    Killing only the direct child leaves anything it spawned running, which on
    a benchmark sweep is how a machine ends up with a hundred orphaned
    interpreters and no obvious parent to blame.
    """

    def terminate() -> None:
        if proc.poll() is not None:
            return
        if sys.platform != "win32":
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                return
            except OSError:
                pass
        proc.kill()

    return terminate
