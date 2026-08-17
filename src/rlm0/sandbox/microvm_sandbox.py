"""A microVM backend: the same container, under a runtime that gives it a kernel.

`docs/RELATED_WORK.md` records the 2026 consensus that shared-kernel container
isolation is no longer adequate for model-written code, which makes the Docker
backend a floor rather than a ceiling. This is the ceiling, and the interesting
part of it is how little of it is new.

WHY AN OCI RUNTIME AND NOT `microsandbox` OR FIRECRACKER DIRECTLY
-----------------------------------------------------------------
Both surveyed candidates were rejected for the same underlying reason, which is
the sub-call channel rather than the hypervisor.

`microsandbox` is the better of the two on paper: Apache-2.0, libkrun microVMs,
sub-200ms boot, self-hostable, and it runs OCI images. But its execution model
is an HTTP request to a server process which forwards one snippet into the VM
and returns its output. There is no way for code inside to block mid-execution
and ask the host for something. A sub-call under that model has to be an
outbound call made from inside the VM to something holding a credential, which
is the precise hole `docker_sandbox.py` exists to refuse: a socket to an
endpoint that holds a key, reachable by exactly the code the isolation was
meant to contain. Swapping a shared kernel for a private one while opening that
socket is a net loss.

E2B's open infrastructure is Firecracker, and self-hosting it means standing up
their orchestration stack; the operational cost is an order of magnitude past
what this project should ask of a user. Firecracker on its own is closer, but
it needs a kernel image and a rootfs that we would have to build and ship, it
is Linux and KVM only, and its host-guest channel is a vsock. That last point
is the killer: `_channel.py` speaks over a child process's stdin and stdout, so
a vsock transport means a second transport under the same protocol, and the
instruction that matters most here is to reuse the wire format rather than grow
a second one.

An OCI microVM runtime avoids all of it. `kata-runtime` and libkrun-backed
`crun` both present the ordinary OCI contract: the daemon starts what looks
like a container, its stdio is piped exactly as before, and the OCI config is
honoured inside by a guest agent. So `--network none`, `--read-only`,
`--cap-drop ALL` and the rest mean the same things, the channel is unchanged,
the guest is unchanged, and the delta is one flag plus a private kernel and a
hardware virtualisation boundary underneath. `run_argv` is reused rather than
restated for the same reason: the containment flags stay in one place, so
weakening them cannot weaken one backend without weakening the other.

THE OPERATIONAL COST, STATED PLAINLY
------------------------------------
This is not the default and should not become one. It needs Linux with
`/dev/kvm`, which rules out most CI runners and every nested-virtualisation-free
cloud instance; it needs the runtime installed and registered with the daemon,
which is a root-level host change; and boot is slower than a container even at
libkrun's sub-200ms, which is paid once per sandbox rather than once per
execution. `DockerSandbox` remains the default because it is the one that works
on a laptop, and the honest framing is that this backend is what you move to
when the context is genuinely hostile and you have a host you control.

WHAT IS VERIFIED AND WHAT IS ASSUMED
------------------------------------
Verified at construction: that the daemon answers, that the named runtime is
actually registered with it rather than merely installed, that a guest booted
and completed the handshake, and, on a Linux host, that the kernel inside is
not the kernel outside. That last check is the one that makes the word microVM
mean something. If the guest reports the host's own kernel release, the runtime
silently fell back to a shared-kernel container, and this refuses to build
rather than hand back an object that claims an isolation it does not have.

Assumed, because no probe from this side can establish it: that the hypervisor
boundary itself holds, and that the runtime's guest agent applies the OCI
config faithfully. The network, filesystem, user and capability properties are
asserted by the same tests as the Docker backend, which run inside the sandbox
and observe the result rather than trusting the flag.

Not verifiable here at all: this file was written and reviewed on a Windows
host with no KVM, so every test past the argv assertions skips. The argv tests
and the fail-closed tests run everywhere, which is the same split the Docker
backend chose and for the same reason.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from collections.abc import Mapping
from typing import Final

from rlm0.ports import SandboxUnavailableError
from rlm0.sandbox.docker_sandbox import (
    DEFAULT_IMAGE,
    DockerSandbox,
    run_argv,
)
from rlm0.sandbox.protocol import STDOUT_CHAR_CAP, HostCallable

__all__ = [
    "DEFAULT_MICROVM_RUNTIME",
    "KERNEL_PROBE",
    "MicroVMSandbox",
    "microvm_available",
    "microvm_run_argv",
    "registered_runtimes",
]

DEFAULT_MICROVM_RUNTIME: Final = "kata-runtime"
"""Kata Containers, because it is the one usually already registered.

libkrun through `crun --krun` is the same idea with a smaller and faster
hypervisor, and is what `microsandbox` uses underneath. It is reached from here
by passing `binary="podman", runtime="krun"`, which is why the runtime is a
parameter rather than a constant.
"""

KERNEL_PROBE: Final = "import os; print(os.uname().release)"
"""What the guest is asked in order to prove it has its own kernel.

A free constant so the construction-time check and the test that documents it
cannot drift apart.
"""

_INFO_TIMEOUT_S: Final = 30.0
_KERNEL_PROBE_TIMEOUT_S: Final = 30.0


def microvm_run_argv(
    *,
    binary: str,
    runtime: str,
    image: str,
    name: str,
    memory: str,
    cpus: str,
    pids_limit: int,
    tmpfs_size: str,
    user: str,
) -> list[str]:
    """The Docker run line with the runtime selected, and nothing else changed.

    A free function for the same reason `run_argv` is one: the containment
    configuration has to be assertable on a machine that cannot run any of
    this, or the flags that carry the security properties are audited only
    where the runtime happens to be installed, which is almost nowhere.

    It splices into `run_argv` rather than restating it. Restating would give
    the two backends two copies of the flag list, and a copy is how one of them
    quietly loses `--network none` in a refactor while its own test keeps
    passing.
    """
    base = run_argv(
        binary=binary,
        image=image,
        name=name,
        memory=memory,
        cpus=cpus,
        pids_limit=pids_limit,
        tmpfs_size=tmpfs_size,
        user=user,
    )
    if base[1] != "run":
        raise AssertionError(
            "run_argv no longer puts the subcommand second, so the runtime "
            "flag cannot be spliced in blindly"
        )
    return [base[0], base[1], "--runtime", runtime, *base[2:]]


def registered_runtimes(binary: str = "docker") -> frozenset[str]:
    """Which runtimes the daemon will actually accept, asked of the daemon.

    Installed and registered are different states, and the gap between them is
    the usual way a `--runtime` flag fails: the binary is on the host, nobody
    edited `daemon.json`, and the error arrives at the first `run` rather than
    at setup. Returns an empty set when the daemon cannot be asked at all,
    because the caller's next move is the same either way.
    """
    path = shutil.which(binary)
    if path is None:
        return frozenset()
    try:
        done = subprocess.run(
            [path, "info", "--format", "{{json .Runtimes}}"],
            capture_output=True,
            timeout=_INFO_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if done.returncode != 0:
        return frozenset()
    try:
        parsed = json.loads(done.stdout.decode("utf-8", "replace") or "null")
    except json.JSONDecodeError:
        return frozenset()
    if isinstance(parsed, Mapping):
        return frozenset(str(name) for name in parsed)
    if isinstance(parsed, list):
        return frozenset(str(name) for name in parsed)
    return frozenset()


def microvm_available(
    binary: str = "docker", runtime: str = DEFAULT_MICROVM_RUNTIME
) -> bool:
    """Whether this exact runtime is registered with a daemon that answers.

    Deliberately not a check for the binary. `docker_available` makes the same
    point about daemons, and the failure this adds is the one where a daemon is
    perfectly healthy and simply does not know the runtime exists.
    """
    return runtime in registered_runtimes(binary)


class MicroVMSandbox(DockerSandbox):
    """A container under a microVM runtime, with its own kernel underneath.

    Subclasses `DockerSandbox` rather than reimplementing it, so that every
    property that backend holds is held here by construction: the same run
    flags, the same image handling, the same container teardown that kills the
    container rather than the client that started it, and the same channel with
    sub-calls serviced by the host over the pipe. What is added is the runtime
    selection, a probe that the runtime is registered, and a check that the
    kernel inside really is not the kernel outside.
    """

    def __init__(
        self,
        *,
        host_calls: Mapping[str, HostCallable] | None = None,
        image: str = DEFAULT_IMAGE,
        binary: str = "docker",
        runtime: str = DEFAULT_MICROVM_RUNTIME,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 128,
        tmpfs_size: str = "64m",
        user: str = "65534:65534",
        pull: bool = True,
        stdout_cap: int = STDOUT_CHAR_CAP,
        startup_timeout_s: float = 120.0,
        require_separate_kernel: bool = True,
    ) -> None:
        # Set before the base constructor, because it starts a child and that
        # child's argv is built from this.
        self._runtime = runtime
        self._require_separate_kernel = require_separate_kernel
        self._guest_kernel = ""
        available = registered_runtimes(binary)
        if available and runtime not in available:
            raise SandboxUnavailableError(
                f"the daemon behind {binary!r} does not have a runtime named "
                f"{runtime!r} registered; it offers "
                f"{', '.join(sorted(available))}. Install a microVM runtime "
                "and register it with the daemon, or use DockerSandbox and "
                "accept a shared kernel."
            )
        # An empty set means the daemon could not be asked. That is not
        # swallowed: the base constructor probes it next and produces the
        # better message, naming the daemon rather than the runtime.
        super().__init__(
            host_calls=host_calls,
            image=image,
            binary=binary,
            memory=memory,
            cpus=cpus,
            pids_limit=pids_limit,
            tmpfs_size=tmpfs_size,
            user=user,
            pull=pull,
            stdout_cap=stdout_cap,
            # Generous next to the container default. A microVM boots a kernel
            # and a guest agent before our own guest starts, and a startup
            # timeout tuned for a container turns a working setup into an
            # intermittent one.
            startup_timeout_s=startup_timeout_s,
        )
        self._verify_separate_kernel()

    def _docker_argv(self, name: str) -> list[str]:
        return microvm_run_argv(
            binary=self._binary,
            runtime=self._runtime,
            image=self._image,
            name=name,
            memory=self._memory,
            cpus=self._cpus,
            pids_limit=self._pids_limit,
            tmpfs_size=self._tmpfs_size,
            user=self._user,
        )

    def _verify_separate_kernel(self) -> None:
        """Fail closed when the guest turns out to be sharing the host's kernel.

        This is the difference between this backend and the one it inherits
        from, so it is checked rather than assumed. A runtime that is
        registered but misconfigured can fall back to running an ordinary
        container, and the resulting object would look identical from the
        outside while providing none of the isolation its name promises.

        Only meaningful when the host is Linux. Everywhere else the daemon is
        already inside a VM whose kernel differs from the host's for reasons
        that have nothing to do with this runtime, so the comparison would pass
        for the wrong reason and is skipped rather than trusted.
        """
        result = self.execute(KERNEL_PROBE, timeout_s=_KERNEL_PROBE_TIMEOUT_S)
        self._guest_kernel = result.stdout.strip()
        if not self._require_separate_kernel:
            return
        if not result.ok or not self._guest_kernel:
            self.close()
            raise SandboxUnavailableError(
                "the guest could not be asked for its kernel release, so "
                "there is no evidence it is running under a microVM: "
                f"{result.stderr.strip() or 'no detail given'}"
            )
        if platform.system() != "Linux":
            return
        if self._guest_kernel == platform.release():
            self.close()
            raise SandboxUnavailableError(
                f"the guest reports the host's own kernel release "
                f"({self._guest_kernel!r}), so runtime {self._runtime!r} "
                "started a shared-kernel container rather than a microVM. "
                "Refusing to hand back a sandbox that claims an isolation it "
                "does not have; fix the runtime, or use DockerSandbox, which "
                "at least says what it is."
            )

    @property
    def runtime(self) -> str:
        """The OCI runtime this sandbox asked the daemon for."""
        return self._runtime

    @property
    def guest_kernel(self) -> str:
        """The kernel release reported from inside, for the run record.

        Worth recording rather than merely checking. It is the one piece of
        evidence a reader has, after the fact, that a run labelled as isolated
        actually was.
        """
        return self._guest_kernel
