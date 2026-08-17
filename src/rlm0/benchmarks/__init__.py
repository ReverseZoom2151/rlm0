"""Public benchmarks, mapped onto the harness this project already has.

rlm0's central empirical claim is that recursion pays on dense aggregation and
loses on retrieval. Until now that could only be tested on a corpus the
project generates itself. A self-authored benchmark is necessary, because it
is the only way to control distractor construction and to hold ground truth to
a self-check, and it is not sufficient, because a benchmark written by the
same hand as the system is evidence about their fit to each other.

So this package adapts public data onto the existing interfaces rather than
building a second evaluation path. An adapter returns a `Corpus`, so
`run_suite` runs it unchanged, `grade` scores it with the same evidence
requirement, and `build_report` still refuses to print a headline without the
depth-zero control beside it. The official metric is computed separately, by
code that follows the benchmark's own scoring function, and the two numbers
are reported side by side and never merged.

Three rules hold across every adapter here.

Nothing is vendored and nothing is downloaded during a test. The data lives
wherever the user put it, tests run against small fixtures constructed in the
test file, and the suite passes offline with no API key. A missing dataset
produces one message naming the exact command that would fix it.

Every run records which benchmark, which split, which revision and the digest
of the files actually read. A number that cannot name its inputs is not a
measurement.

Every adapter states plainly whether it reproduces the official metric or
approximates it, and the statement is carried into the manifest. A number that
is quietly not comparable to a leaderboard is worse than no number, because it
will be compared anyway.
"""

from __future__ import annotations

from rlm0.benchmarks.anomalyxl import AnomalyMetrics, AnomalyXL
from rlm0.benchmarks.context import chunk_context, locate
from rlm0.benchmarks.dataset import (
    ENV_ROOT,
    BenchmarkDataError,
    DatasetRequirement,
    DatasetUnavailableError,
    LoadedFiles,
    load_files,
    resolve_root,
)
from rlm0.benchmarks.niah import RulerNiah, string_match_all
from rlm0.benchmarks.oolong import OolongReal, OolongSynth
from rlm0.benchmarks.registry import (
    ADAPTERS,
    NOT_ADAPTED,
    UnadaptedBenchmark,
    describe_catalogue,
    get,
    names,
)
from rlm0.benchmarks.scoring import (
    Fidelity,
    OfficialItem,
    OfficialResult,
    OfficialSummary,
    Scoreboard,
)
from rlm0.benchmarks.suite import (
    BenchmarkAdapter,
    BenchmarkManifest,
    BenchmarkResult,
    BenchmarkSuite,
    run_benchmark,
)

__all__ = [
    "ADAPTERS",
    "ENV_ROOT",
    "NOT_ADAPTED",
    "AnomalyMetrics",
    "AnomalyXL",
    "BenchmarkAdapter",
    "BenchmarkDataError",
    "BenchmarkManifest",
    "BenchmarkResult",
    "BenchmarkSuite",
    "DatasetRequirement",
    "DatasetUnavailableError",
    "Fidelity",
    "LoadedFiles",
    "OfficialItem",
    "OfficialResult",
    "OfficialSummary",
    "OolongReal",
    "OolongSynth",
    "RulerNiah",
    "Scoreboard",
    "UnadaptedBenchmark",
    "chunk_context",
    "describe_catalogue",
    "get",
    "load_files",
    "locate",
    "names",
    "resolve_root",
    "run_benchmark",
    "string_match_all",
]
