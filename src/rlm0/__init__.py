"""rlm0: a recursive language model runtime that runs the control first.

The name is the architecture. Depth zero, the REPL with no sub-calls, is
attempted before anything deeper, so every run carries the counterfactual that
says whether recursion was worth its cost on that query. Escalation happens
when the cheap attempt fails, not because a config file said depth equals two.

What is exported here is what a caller needs to run one: `build_rlm` to wire a
model, a sandbox and a budget into an `RLM`, the budgets and policies worth
choosing between, and the `Run` record the whole thing exists to produce.

Two things are deliberately absent. The provider clients are not re-exported,
because importing this package would then import an optional SDK, and the
harness is not re-exported, because it is a heavier dependency on this package
than this package should have on itself. Both are one explicit import away:
`from rlm0.providers import AnthropicClient`, `from rlm0.harness import
run_suite`.
"""

from rlm0.assembly import (
    CONTEXT_VARIABLE,
    SUB_CALL_NAME,
    SandboxChoice,
    build_rlm,
    default_budget,
    sandbox_factory,
)
from rlm0.budget import BudgetSnapshot, RunBudget, Unbounded
from rlm0.policy import Escalating, Fixed, Never
from rlm0.ports import (
    Budget,
    CallReservation,
    DepthPolicy,
    EscalationContext,
    ExecResult,
    LMClient,
    LMResponse,
    Sandbox,
    SandboxUnavailableError,
)
from rlm0.run import (
    Attempt,
    BaselineWaiver,
    CallRecord,
    Outcome,
    RecursionVerdict,
    Role,
    Run,
    TokenUsage,
    Verdict,
)
from rlm0.runtime import RLM, RecursionUnavailableError
from rlm0.sandbox import DockerSandbox, SubprocessSandbox, docker_available

__version__ = "0.1.0"

__all__ = [
    "CONTEXT_VARIABLE",
    "RLM",
    "SUB_CALL_NAME",
    "Attempt",
    "BaselineWaiver",
    "Budget",
    "BudgetSnapshot",
    "CallRecord",
    "CallReservation",
    "DepthPolicy",
    "DockerSandbox",
    "Escalating",
    "EscalationContext",
    "ExecResult",
    "Fixed",
    "LMClient",
    "LMResponse",
    "Never",
    "Outcome",
    "RecursionUnavailableError",
    "RecursionVerdict",
    "Role",
    "Run",
    "RunBudget",
    "Sandbox",
    "SandboxChoice",
    "SandboxUnavailableError",
    "SubprocessSandbox",
    "TokenUsage",
    "Unbounded",
    "Verdict",
    "__version__",
    "build_rlm",
    "default_budget",
    "docker_available",
    "sandbox_factory",
]
