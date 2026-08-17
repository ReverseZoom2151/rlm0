"""The microVM backend.

The split is the same one the Docker tests make and for the same reason: the
containment configuration is asserted as a free function on any machine, and
everything that needs a hypervisor is behind one skip marker that asks the
daemon which runtimes it will actually accept rather than looking for a binary.

The argv tests are the load-bearing ones here. This backend inherits its
containment from `run_argv`, so what has to be proved without a runtime present
is that it inherits all of it and changes only the runtime selection. A test
that could only run where a microVM runtime is installed is a test that runs
nowhere.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rlm0.ports import SandboxUnavailableError
from rlm0.sandbox.docker_sandbox import DEFAULT_IMAGE, run_argv
from rlm0.sandbox.microvm_sandbox import (
    DEFAULT_MICROVM_RUNTIME,
    KERNEL_PROBE,
    MicroVMSandbox,
    microvm_available,
    microvm_run_argv,
    registered_runtimes,
)

requires_microvm = pytest.mark.skipif(
    not microvm_available(),
    reason=f"no daemon offering the {DEFAULT_MICROVM_RUNTIME} runtime",
)


def _argv(runtime: str = DEFAULT_MICROVM_RUNTIME) -> list[str]:
    return microvm_run_argv(
        binary="docker",
        runtime=runtime,
        image=DEFAULT_IMAGE,
        name="rlm0-sbx-test",
        memory="512m",
        cpus="1.0",
        pids_limit=128,
        tmpfs_size="64m",
        user="65534:65534",
    )


def test_the_run_line_holds_every_property_the_docker_backend_claims() -> None:
    """Asserted without a runtime, because this is the part that must not rot."""
    joined = " ".join(_argv())
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pids-limit" in joined
    assert "--memory 512m" in joined
    assert "--memory-swap 512m" in joined
    assert "--user 65534:65534" in joined
    assert "/tmp:rw,noexec,nosuid,nodev,size=64m" in joined
    assert DEFAULT_IMAGE in _argv()
    # No credential shaped environment is handed across.
    assert not [a for a in _argv() if "KEY" in a or "TOKEN" in a]


def test_the_run_line_is_the_docker_one_plus_the_runtime_and_nothing_else() -> None:
    """The strongest statement available without a hypervisor.

    Listing the flags again would pass even if this backend had quietly dropped
    one that `run_argv` still carries. Comparing the two lines directly is what
    makes the inheritance real rather than asserted.
    """
    base = run_argv(
        binary="docker",
        image=DEFAULT_IMAGE,
        name="rlm0-sbx-test",
        memory="512m",
        cpus="1.0",
        pids_limit=128,
        tmpfs_size="64m",
        user="65534:65534",
    )
    built = _argv()
    assert built == [*base[:2], "--runtime", DEFAULT_MICROVM_RUNTIME, *base[2:]]


def test_the_runtime_is_selectable_because_libkrun_is_reached_differently() -> None:
    argv = microvm_run_argv(
        binary="podman",
        runtime="krun",
        image=DEFAULT_IMAGE,
        name="rlm0-sbx-test",
        memory="512m",
        cpus="1.0",
        pids_limit=128,
        tmpfs_size="64m",
        user="65534:65534",
    )
    assert argv[:4] == ["podman", "run", "--runtime", "krun"]


def test_the_runtime_flag_precedes_the_image_and_the_command() -> None:
    """A daemon flag placed after the image is an argument to the guest."""
    argv = _argv()
    assert argv.index("--runtime") < argv.index(DEFAULT_IMAGE)


def test_a_missing_daemon_fails_at_construction_not_at_first_execute() -> None:
    """The port promises this, so it is tested with nothing installed."""
    with pytest.raises(SandboxUnavailableError):
        MicroVMSandbox(binary="rlm0-definitely-not-a-real-docker")


def test_an_absent_daemon_reports_no_runtimes_rather_than_raising() -> None:
    """The probe answers a question; the constructor decides what to do about it."""
    assert registered_runtimes("rlm0-definitely-not-a-real-docker") == frozenset()
    assert microvm_available("rlm0-definitely-not-a-real-docker") is False


def test_availability_is_asked_of_the_daemon_not_of_the_path() -> None:
    """Installed and registered are different states, and only one of them works."""
    assert microvm_available("docker", "rlm0-runtime-that-nobody-registered") is False


@pytest.fixture
def sandbox() -> Iterator[MicroVMSandbox]:
    made = MicroVMSandbox(host_calls={"echo": lambda text: f"host saw {text}"})
    try:
        yield made
    finally:
        made.close()


@requires_microvm
def test_the_guest_is_not_running_the_hosts_kernel(
    sandbox: MicroVMSandbox,
) -> None:
    """The one property that distinguishes this backend from the one it extends.

    Checked at construction too, where a match fails closed rather than
    returning a sandbox that claims an isolation it does not have.
    """
    import platform

    assert sandbox.guest_kernel
    if platform.system() == "Linux":
        assert sandbox.guest_kernel != platform.release()


@requires_microvm
def test_the_kernel_probe_is_the_one_the_constructor_uses(
    sandbox: MicroVMSandbox,
) -> None:
    result = sandbox.execute(KERNEL_PROBE, timeout_s=30)
    assert result.stdout.strip() == sandbox.guest_kernel


@requires_microvm
def test_the_microvm_really_has_no_network(sandbox: MicroVMSandbox) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    print('REACHED')\n"
        "except OSError as exc:\n"
        "    print('blocked', type(exc).__name__)\n"
    )
    result = sandbox.execute(code, timeout_s=30)
    assert "REACHED" not in result.stdout
    assert "blocked" in result.stdout


@requires_microvm
def test_dns_does_not_resolve_either(sandbox: MicroVMSandbox) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    print('RESOLVED', socket.gethostbyname('api.anthropic.com'))\n"
        "except OSError:\n"
        "    print('no resolver')\n"
    )
    assert "RESOLVED" not in sandbox.execute(code, timeout_s=30).stdout


@requires_microvm
def test_it_does_not_run_as_root(sandbox: MicroVMSandbox) -> None:
    result = sandbox.execute("import os; print(os.getuid())", timeout_s=30)
    assert result.stdout.strip() == "65534"


@requires_microvm
def test_the_root_filesystem_is_read_only(sandbox: MicroVMSandbox) -> None:
    code = (
        "try:\n"
        "    open('/evidence', 'w').write('x')\n"
        "    print('WROTE')\n"
        "except OSError as exc:\n"
        "    print('refused', type(exc).__name__)\n"
    )
    result = sandbox.execute(code, timeout_s=30)
    assert "WROTE" not in result.stdout
    assert "refused" in result.stdout


@requires_microvm
def test_no_host_environment_crosses_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-cross-the-boundary")
    made = MicroVMSandbox()
    try:
        result = made.execute("import os; print(sorted(os.environ))", timeout_s=30)
        assert "ANTHROPIC_API_KEY" not in result.stdout
    finally:
        made.close()


@requires_microvm
def test_a_sub_call_is_serviced_over_the_pipe_not_a_socket(
    sandbox: MicroVMSandbox,
) -> None:
    """The property the rejected candidates could not hold. See the module docstring."""
    sandbox.register_host_call("echo", 1)
    result = sandbox.execute("print(echo('hello'))", timeout_s=30)
    assert result.stdout.strip() == "host saw hello"


@requires_microvm
def test_state_persists_across_executions(sandbox: MicroVMSandbox) -> None:
    sandbox.set_variable("context", "needle in a haystack " * 1000)
    assert sandbox.execute("hits = context.count('needle')", timeout_s=30).ok
    assert sandbox.execute("print(hits)", timeout_s=30).stdout.strip() == "1000"


@requires_microvm
def test_a_runaway_loop_returns(sandbox: MicroVMSandbox) -> None:
    result = sandbox.execute("while True:\n    pass\n", timeout_s=2.0)
    assert not result.ok
    assert sandbox.execute("print('alive')", timeout_s=30).stdout.strip() == "alive"


@requires_microvm
def test_the_machine_is_gone_after_close() -> None:
    import subprocess

    made = MicroVMSandbox()
    name = made.container_name
    made.close()
    made.close()
    listing = subprocess.run(
        ["docker", "ps", "--all", "--quiet", "--filter", f"name={name}"],
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert listing.stdout.strip() == b""
