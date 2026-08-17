"""The default sandbox.

Everything that needs a daemon is behind one skip marker that asks the daemon
rather than looking for the binary, so these run for real wherever Docker is
real and skip cleanly where it is not. The fail closed tests need no daemon at
all, which is deliberate: the path that refuses to build a sandbox is the one
that must never rot, and a check that only runs on machines that have Docker
would never exercise it.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rlm0.ports import SandboxUnavailableError
from rlm0.sandbox import DockerSandbox, docker_available
from rlm0.sandbox.docker_sandbox import DEFAULT_IMAGE, run_argv

requires_docker = pytest.mark.skipif(
    not docker_available(), reason="no Docker daemon answered"
)


def test_a_missing_docker_fails_at_construction_not_at_first_execute() -> None:
    """The port promises this, so it is tested without a daemon present."""
    with pytest.raises(SandboxUnavailableError) as caught:
        DockerSandbox(binary="rlm0-definitely-not-a-real-docker")
    assert "PATH" in str(caught.value)


def test_the_failure_message_points_at_the_deliberate_alternative() -> None:
    with pytest.raises(SandboxUnavailableError) as caught:
        DockerSandbox(binary="rlm0-definitely-not-a-real-docker")
    assert "SubprocessSandbox" in str(caught.value)


@pytest.fixture
def sandbox() -> Iterator[DockerSandbox]:
    made = DockerSandbox(host_calls={"echo": lambda text: f"host saw {text}"})
    try:
        yield made
    finally:
        made.close()


def test_the_run_line_holds_every_property_we_claim() -> None:
    """Asserted without a daemon, because this is the part that must not rot."""
    argv = run_argv(
        binary="docker",
        image=DEFAULT_IMAGE,
        name="rlm0-sbx-test",
        memory="512m",
        cpus="1.0",
        pids_limit=128,
        tmpfs_size="64m",
        user="65534:65534",
    )
    joined = " ".join(argv)
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pids-limit" in joined
    assert "--memory 512m" in joined
    assert "--memory-swap 512m" in joined
    assert "--user 65534:65534" in joined
    assert "/tmp:rw,noexec,nosuid,nodev,size=64m" in joined
    assert DEFAULT_IMAGE in argv
    # No credential shaped environment is handed across.
    assert not [a for a in argv if "KEY" in a or "TOKEN" in a]


@requires_docker
def test_the_container_really_has_no_network(sandbox: DockerSandbox) -> None:
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


@requires_docker
def test_dns_does_not_resolve_either(sandbox: DockerSandbox) -> None:
    code = (
        "import socket\n"
        "try:\n"
        "    print('RESOLVED', socket.gethostbyname('api.anthropic.com'))\n"
        "except OSError:\n"
        "    print('no resolver')\n"
    )
    assert "RESOLVED" not in sandbox.execute(code, timeout_s=30).stdout


@requires_docker
def test_it_does_not_run_as_root(sandbox: DockerSandbox) -> None:
    result = sandbox.execute("import os; print(os.getuid())", timeout_s=30)
    assert result.stdout.strip() == "65534"


@requires_docker
def test_the_root_filesystem_is_read_only(sandbox: DockerSandbox) -> None:
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


@requires_docker
def test_scratch_space_exists_for_work_that_needs_it(sandbox: DockerSandbox) -> None:
    code = (
        "open('/tmp/scratch', 'w').write('x' * 10)\n"
        "print(open('/tmp/scratch').read())\n"
    )
    assert sandbox.execute(code, timeout_s=30).stdout.strip() == "xxxxxxxxxx"


@requires_docker
def test_no_host_environment_crosses_the_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-never-cross-the-boundary")
    made = DockerSandbox()
    try:
        result = made.execute("import os; print(sorted(os.environ))", timeout_s=30)
        assert "ANTHROPIC_API_KEY" not in result.stdout
    finally:
        made.close()


@requires_docker
def test_state_persists_across_executions(sandbox: DockerSandbox) -> None:
    sandbox.set_variable("context", "needle in a haystack " * 1000)
    assert sandbox.execute("hits = context.count('needle')", timeout_s=30).ok
    assert sandbox.execute("print(hits)", timeout_s=30).stdout.strip() == "1000"


@requires_docker
def test_a_sub_call_is_serviced_over_the_pipe(sandbox: DockerSandbox) -> None:
    sandbox.register_host_call("echo", 1)
    result = sandbox.execute("print(echo('hello'))", timeout_s=30)
    assert result.stdout.strip() == "host saw hello"


@requires_docker
def test_a_runaway_loop_returns(sandbox: DockerSandbox) -> None:
    result = sandbox.execute("while True:\n    pass\n", timeout_s=2.0)
    assert not result.ok
    assert sandbox.execute("print('alive')", timeout_s=30).stdout.strip() == "alive"


@requires_docker
def test_the_container_is_gone_after_close() -> None:
    import subprocess

    made = DockerSandbox()
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
