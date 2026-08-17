"""Fixtures for the tests that run the layers against each other.

Everything here exists to make one thing easy: driving a real sandbox, a real
budget and a real policy with a scripted model, in both sandboxes, from one
test body. The model is the only fake, and it is fake because a test that
needed a key would not run.
"""

from __future__ import annotations

import pytest

from rlm0.assembly import SandboxChoice
from rlm0.sandbox import docker_available

requires_docker = pytest.mark.skipif(
    not docker_available(), reason="no Docker daemon answered"
)


@pytest.fixture(
    params=[
        pytest.param("subprocess", id="subprocess"),
        pytest.param("docker", id="docker", marks=requires_docker),
    ]
)
def sandbox_choice(request: pytest.FixtureRequest) -> SandboxChoice:
    """Both real sandboxes, with the Docker half skipping where there is none.

    Parametrised rather than duplicated because the property being tested is
    that nothing above the sandbox can tell which one it got. Two hand-written
    copies of a test drift, and the first thing they drift on is the assertion
    that would have caught the divergence.
    """
    choice: SandboxChoice = request.param
    return choice
