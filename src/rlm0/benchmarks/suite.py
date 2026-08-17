"""What an adapter hands back, and what gets written next to a result.

An adapter produces three things and they are kept separate on purpose. A
`Corpus`, so that the existing runner, the existing evidence-aware grader and
the existing refusing report work with no change at all: the whole value of
adapting to public data is lost if the public data gets its own private
pipeline that nobody audits. A `Scoreboard`, so the official number can be
computed from the same answers without contaminating the harness grade. And a
`BenchmarkManifest`, which is the part that makes a figure citable: which
benchmark, which split, which revision, which files, hashed.

`run_benchmark` is a thin wrapper and stays thin deliberately. It calls
`run_suite` unchanged and then writes the two extra files. Anything it did
beyond that would be a second execution path with its own bugs, and the
project's claim is that a public benchmark runs through the same machinery as
the synthetic corpus rather than beside it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rlm0.benchmarks.dataset import DatasetRequirement
from rlm0.benchmarks.scoring import (
    Fidelity,
    OfficialResult,
    OfficialSummary,
    Scoreboard,
)
from rlm0.harness.corpus import Corpus, CorpusSpec
from rlm0.harness.grading import GradingPolicy
from rlm0.harness.runner import Solver, SuiteResult, run_suite

__all__ = [
    "BENCHMARK_FILENAME",
    "OFFICIAL_FILENAME",
    "BenchmarkAdapter",
    "BenchmarkManifest",
    "BenchmarkResult",
    "BenchmarkSuite",
    "corpus_spec_for",
    "run_benchmark",
]

BENCHMARK_FILENAME = "benchmark.json"
OFFICIAL_FILENAME = "official_scores.json"

ADAPTER_VERSION = "0.1.0"


def corpus_spec_for(dataset_hash: str) -> CorpusSpec:
    """A spec for a corpus that was loaded rather than generated.

    `CorpusSpec` describes the synthetic generator, and for loaded data all of
    it is inert except the seed, which is derived from the dataset digest so
    that the harness manifest still changes when the underlying bytes change.
    The honest provenance lives in `benchmark.json` beside it; this exists so
    that two runs on different data can never collide in the harness manifest
    while looking identical.
    """
    return CorpusSpec(seed=int(dataset_hash[:12], 16))


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """Provenance for one benchmark run, written next to the records.

    Every field here answers a question somebody will ask of a published
    number: what was measured, on which half of the data, at which revision,
    over how many samples, against which metric, and how far the local scorer
    is from the published one. `deviations` is the field that keeps this
    honest, because an adapter always deviates somewhere and the alternative
    to writing it down is discovering it in a review.
    """

    benchmark: str
    source: str
    revision: str
    config: str
    split: str
    dataset_hash: str
    files: tuple[str, ...]
    n_samples: int
    official_metric: str
    fidelity: Fidelity
    fidelity_note: str
    deviations: tuple[str, ...]
    adapter_version: str = ADAPTER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "source": self.source,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "dataset_hash": self.dataset_hash,
            "files": list(self.files),
            "n_samples": self.n_samples,
            "official_metric": self.official_metric,
            "fidelity": self.fidelity.value,
            "fidelity_note": self.fidelity_note,
            "deviations": list(self.deviations),
            "adapter_version": self.adapter_version,
        }

    def describe(self) -> str:
        return (
            f"{self.benchmark} {self.config}/{self.split} @ "
            f"{self.revision[:12]} ({self.n_samples} samples, data "
            f"{self.dataset_hash[:12]})"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """A public benchmark, loaded and ready to run through the normal harness."""

    corpus: Corpus
    scoreboard: Scoreboard
    manifest: BenchmarkManifest

    @property
    def n_samples(self) -> int:
        return len(self.corpus.samples)


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """The interface every benchmark in this package satisfies.

    `requirement` is separated from `load` so that a caller can print what is
    needed without having it, which is what makes a missing dataset a
    documented prerequisite rather than a runtime surprise.
    """

    @property
    def name(self) -> str: ...

    def requirement(self, *, split: str) -> DatasetRequirement:
        """What must be on disk before `load` can succeed."""
        ...

    def load(
        self,
        *,
        split: str,
        root: Path | None = None,
        limit: int | None = None,
        expected_hash: str | None = None,
    ) -> BenchmarkSuite:
        """Read the local data and build a suite, or raise saying why not."""
        ...

    def answer_instruction(self) -> str:
        """The output format the official parser expects.

        Not decoration. Both OOLONG scorers key off a specific surface form,
        and a solver that does not produce it scores zero for a formatting
        reason that looks exactly like a capability result.
        """
        ...


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """The harness result and the official number, side by side and unmixed."""

    suite: SuiteResult
    manifest: BenchmarkManifest
    official: OfficialSummary
    official_results: tuple[OfficialResult, ...]

    def describe(self) -> str:
        return f"{self.manifest.describe()}\n{self.official.describe()}"


def run_benchmark(
    suite: BenchmarkSuite,
    solver: Solver,
    out_dir: Path,
    *,
    policy: GradingPolicy | None = None,
    invocation: Sequence[str] | None = None,
    now: Callable[[], str] | None = None,
    resume: bool = True,
) -> BenchmarkResult:
    """Run a loaded benchmark through the unmodified harness, then score it twice.

    The official score is computed from the persisted records rather than from
    anything the solver said on the way past, so it is recomputable from disk
    long after the process has gone, which is the property that separates a
    measurement from an anecdote.
    """
    result = run_suite(
        suite.corpus,
        solver,
        out_dir,
        policy=policy,
        invocation=invocation,
        now=now,
        resume=resume,
    )
    answers: dict[str, str | None] = {
        record.sample_id: record.answer for record in result.records
    }
    official_results = suite.scoreboard.score_all(answers)
    official = suite.scoreboard.summarise(official_results)

    (out_dir / BENCHMARK_FILENAME).write_text(
        json.dumps(suite.manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / OFFICIAL_FILENAME).write_text(
        json.dumps(
            {
                "summary": official.to_dict(),
                "per_sample": [r.to_dict() for r in official_results],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return BenchmarkResult(
        suite=result,
        manifest=suite.manifest,
        official=official,
        official_results=official_results,
    )
