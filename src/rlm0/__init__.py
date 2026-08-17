"""rlm0: a recursive language model runtime that runs the control first.

The name is the architecture. Depth zero, the REPL with no sub-calls, is
attempted before anything deeper, so every run carries the counterfactual that
says whether recursion was worth its cost on that query. Escalation happens
when the cheap attempt fails, not because a config file said depth equals two.
"""

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

__version__ = "0.1.0"

__all__ = [
    "Attempt",
    "BaselineWaiver",
    "CallRecord",
    "Outcome",
    "RecursionVerdict",
    "Role",
    "Run",
    "TokenUsage",
    "Verdict",
    "__version__",
]
