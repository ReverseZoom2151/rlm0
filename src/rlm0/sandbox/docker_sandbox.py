"""The default sandbox: one long lived container with the network switched off.

The design constraint that shapes everything here is that `--network none` is
held genuinely. Several surveyed implementations reach the same place by
starting the container with no network and then punching an HTTP hole through
to a proxy on the host so that code inside can make sub-calls. That is not a
smaller hole than a network: it is a socket to an endpoint that holds a
credential, reachable by exactly the code the isolation was meant to contain.

Here a sub-call travels the same pipe as everything else. Code inside calls a
function that writes a JSON line to the channel and blocks; the host runs the
real call with the real key and writes one JSON line back. The container has no
interface, no resolver and no route, and nothing inside it ever learns that a
provider exists.

The rest is ordinary containment, stated because defaults are what people
actually get: a non-root user, a read-only root filesystem with a small noexec
tmpfs for scratch, memory, CPU and pid ceilings, every capability dropped, and
no new privileges.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
import uuid
from collections.abc import Callable, Mapping
from typing import Final

from rlm0.ports import SandboxUnavailableError
from rlm0.sandbox._channel import LOADER, ChannelSandbox, Child
from rlm0.sandbox.protocol import STDOUT_CHAR_CAP, HostCallable

__all__ = ["DEFAULT_IMAGE", "DockerSandbox", "docker_available", "run_argv"]

DEFAULT_IMAGE: Final = "python:3.11-slim"

_PROBE_TIMEOUT_S: Final = 20.0
_PULL_TIMEOUT_S: Final = 600.0
_NOBODY: Final = "65534:65534"


def docker_available(binary: str = "docker") -> bool:
    """Whether a daemon actually answers, not whether a binary exists.

    A `docker` on PATH with no daemon behind it is the common case on a laptop
    that has just rebooted, and it is precisely the case that a which() style
    check gets wrong.
    """
    try:
        _probe(binary)
    except SandboxUnavailableError:
        return False
    return True


def _run(argv: list[str], timeout_s: float) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, capture_output=True, timeout=timeout_s, check=False)


def _probe(binary: str) -> str:
    """Ask the daemon for its version, and translate every failure.

    This runs at construction. The fail closed rule in the port docstring is
    only worth anything if the check is on the path that everybody takes, and
    a surveyed repo demonstrates the alternative: a `check_health` method that
    nothing ever called, so the documented failure mode never once fired.
    """
    path = shutil.which(binary)
    if path is None:
        raise SandboxUnavailableError(
            f"no {binary!r} on PATH. Install Docker, or opt in to "
            "SubprocessSandbox and accept that it is not a security boundary."
        )
    try:
        argv = [path, "version", "--format", "{{.Server.Version}}"]
        done = _run(argv, _PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SandboxUnavailableError(
            f"{binary} did not answer within {_PROBE_TIMEOUT_S:g}s; the daemon "
            "is probably starting or wedged"
        ) from exc
    except OSError as exc:
        raise SandboxUnavailableError(f"could not run {binary}: {exc}") from exc
    if done.returncode != 0:
        detail = done.stderr.decode("utf-8", "replace").strip().splitlines()
        raise SandboxUnavailableError(
            f"{binary} is installed but no daemon answered: "
            f"{detail[-1] if detail else 'no detail given'}"
        )
    return path


def run_argv(
    *,
    binary: str,
    image: str,
    name: str,
    memory: str,
    cpus: str,
    pids_limit: int,
    tmpfs_size: str,
    user: str,
) -> list[str]:
    """The full run line, in one place so it can be read and audited.

    A free function rather than a method so that the flags can be asserted on
    a machine with no daemon. The containment properties are the part of this
    file most likely to be weakened by a well meaning edit, and a test that
    can only run where Docker is installed is a test that mostly does not run.

    Every flag here is load bearing, so none is optional and none is behind a
    convenience switch that a hurried caller would flip.
    """
    return [
        binary,
        "run",
        "--rm",
        "--interactive",
        "--name",
        name,
        # The property this whole design exists to preserve.
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,nodev,size={tmpfs_size}",
        "--memory",
        memory,
        # Without this, memory pressure spills into swap instead of failing,
        # so the ceiling stops being a ceiling.
        "--memory-swap",
        memory,
        "--cpus",
        cpus,
        "--pids-limit",
        str(pids_limit),
        "--user",
        user,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--workdir",
        "/tmp",
        "--env",
        "HOME=/tmp",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        # No other environment crosses. The host keeps its keys.
        image,
        "python3",
        "-I",
        "-u",
        "-c",
        LOADER,
    ]


class DockerSandbox(ChannelSandbox):
    """A container that stays up across executions and never sees the network.

    State persists between `execute` calls because the container holds one
    Python process for its whole life, which is what makes the REPL model work
    at all: a huge context is placed in a variable once and every subsequent
    block operates on it in place.
    """

    def __init__(
        self,
        *,
        host_calls: Mapping[str, HostCallable] | None = None,
        image: str = DEFAULT_IMAGE,
        binary: str = "docker",
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 128,
        tmpfs_size: str = "64m",
        user: str = _NOBODY,
        pull: bool = True,
        stdout_cap: int = STDOUT_CHAR_CAP,
        startup_timeout_s: float = 60.0,
    ) -> None:
        super().__init__(
            host_calls=host_calls,
            stdout_cap=stdout_cap,
            startup_timeout_s=startup_timeout_s,
        )
        self._binary = _probe(binary)
        self._image = image
        self._memory = memory
        self._cpus = cpus
        self._pids_limit = pids_limit
        self._tmpfs_size = tmpfs_size
        self._user = user
        self._container = ""
        self._ensure_image(pull=pull)
        self._start()

    def _ensure_image(self, *, pull: bool) -> None:
        if _run([self._binary, "image", "inspect", self._image], 60.0).returncode == 0:
            return
        if not pull:
            raise SandboxUnavailableError(
                f"image {self._image!r} is not present and pull=False"
            )
        try:
            done = _run([self._binary, "pull", self._image], _PULL_TIMEOUT_S)
        except subprocess.TimeoutExpired as exc:
            raise SandboxUnavailableError(
                f"pulling {self._image!r} took longer than {_PULL_TIMEOUT_S:g}s"
            ) from exc
        if done.returncode != 0:
            tail = done.stderr.decode("utf-8", "replace").strip().splitlines()
            raise SandboxUnavailableError(
                f"could not pull {self._image!r}: "
                f"{tail[-1] if tail else 'no detail given'}"
            )

    def _docker_argv(self, name: str) -> list[str]:
        return run_argv(
            binary=self._binary,
            image=self._image,
            name=name,
            memory=self._memory,
            cpus=self._cpus,
            pids_limit=self._pids_limit,
            tmpfs_size=self._tmpfs_size,
            user=self._user,
        )

    def _spawn(self) -> Child:
        name = f"rlm0-sbx-{uuid.uuid4().hex[:12]}"
        self._container = name
        try:
            proc = subprocess.Popen(
                self._docker_argv(name),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            raise SandboxUnavailableError(
                f"could not start a container from {self._image!r}: {exc}"
            ) from exc
        return Child(
            proc=proc,
            terminate=self._terminator(name, proc),
            label=f"container {name}",
        )

    def _terminator(
        self, name: str, proc: subprocess.Popen[bytes]
    ) -> Callable[[], None]:
        """Kill the container, not just the client that started it.

        `docker run` is a client. Killing it detaches from the container and
        leaves it running, which is how a benchmark sweep quietly leaks one
        container per task until the host runs out of memory.
        """
        binary = self._binary

        def terminate() -> None:
            with contextlib.suppress(OSError, subprocess.TimeoutExpired):
                _run([binary, "rm", "--force", name], 30.0)
            if proc.poll() is None:
                proc.kill()

        return terminate

    @property
    def container_name(self) -> str:
        """The current container, which changes if one had to be replaced."""
        return self._container
