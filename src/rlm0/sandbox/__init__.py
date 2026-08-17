"""Somewhere to run model written code that the model cannot escape.

Three implementations, and deliberately not a fourth. There is no in-process
`exec` path here and there will not be one. Every surveyed implementation that
offered one made it the default, because it is the one that always works, and
then shipped a filtered `__builtins__` as the sandbox. That filter is not a
boundary; escaping one of them took three lines. The absent option is the
security control.

`DockerSandbox` is the default and the weakest thing here that is a boundary.
`MicroVMSandbox` is the same container under a runtime that gives it its own
kernel, which is the isolation the field moved to and which costs a host you
control; it is opt in because it cannot run on a laptop. `SubprocessSandbox` is
an explicit opt in for machines without Docker and says in its own docstring
that it protects the orchestrator, not the host.
"""

from __future__ import annotations

from rlm0.ports import ExecResult, Sandbox, SandboxUnavailableError
from rlm0.sandbox.docker_sandbox import DEFAULT_IMAGE, DockerSandbox, docker_available
from rlm0.sandbox.microvm_sandbox import (
    DEFAULT_MICROVM_RUNTIME,
    MicroVMSandbox,
    microvm_available,
)
from rlm0.sandbox.protocol import STDOUT_CHAR_CAP, HostCallable, scrub_secrets
from rlm0.sandbox.subprocess_sandbox import SubprocessSandbox

__all__ = [
    "DEFAULT_IMAGE",
    "DEFAULT_MICROVM_RUNTIME",
    "STDOUT_CHAR_CAP",
    "DockerSandbox",
    "ExecResult",
    "HostCallable",
    "MicroVMSandbox",
    "Sandbox",
    "SandboxUnavailableError",
    "SubprocessSandbox",
    "docker_available",
    "microvm_available",
    "scrub_secrets",
]
