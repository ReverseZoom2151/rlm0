"""The part of rlm0 that makes its two claims checkable.

The field asserts that recursion helped and that it was cheap. Neither claim
survives contact with an artifact that can measure it, and no open
implementation ships one: a survey of twenty found not a single correctness
number, a reference implementation with no grader at all, and a cost
comparison that reports zero tokens for every configuration because it reads a
usage key nothing sets.

So this package is four things. A corpus that is deterministic from a seed,
adversarial by construction and self-verifying. A grader that scores the cited
evidence separately from the answer, because a right answer from the wrong
documents is luck. A report that refuses to print a headline without the
depth-zero row beside it. And a runner that persists raw per-sample records
with the full run accounting, so a published figure can be traced back to the
trajectory that produced it.
"""

from __future__ import annotations

from rlm0.harness.corpus import (
    Corpus,
    CorpusSpec,
    Document,
    GroundTruthError,
    Regime,
    Sample,
    SolverTask,
    TaskFamily,
    generate_corpus,
    verify_sample,
)
from rlm0.harness.grading import (
    GradingPolicy,
    SampleScore,
    ScoreSummary,
    grade,
    summarise,
)
from rlm0.harness.report import (
    HarnessVerdict,
    IntegrityReport,
    ReportRefusalError,
    ReportRow,
    ResultTable,
    SampleRecord,
    VerdictCounts,
    build_report,
    classify,
)
from rlm0.harness.runner import (
    Attempted,
    Solver,
    SolverContractError,
    SolverResult,
    SuiteResult,
    run_suite,
)

__all__ = [
    "Attempted",
    "Corpus",
    "CorpusSpec",
    "Document",
    "GradingPolicy",
    "GroundTruthError",
    "HarnessVerdict",
    "IntegrityReport",
    "Regime",
    "ReportRefusalError",
    "ReportRow",
    "ResultTable",
    "Sample",
    "SampleRecord",
    "SampleScore",
    "ScoreSummary",
    "Solver",
    "SolverContractError",
    "SolverResult",
    "SolverTask",
    "SuiteResult",
    "TaskFamily",
    "VerdictCounts",
    "build_report",
    "classify",
    "generate_corpus",
    "grade",
    "run_suite",
    "summarise",
    "verify_sample",
]
