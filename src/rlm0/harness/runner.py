"""Execute a suite against anything that satisfies a minimal solver interface.

The interface is deliberately small, because the point of it is that a
competing implementation can be measured on the same corpus without adopting
any of this project's internals. What it is not is optional: a solver must
hand back a `Run`, which means it must have attempted depth zero, attributed
its calls, and named the budget it executed under. That is the price of
appearing in a table here, and it is the price precisely because every
implementation surveyed for this project could have produced a comparison and
none did.

Records are persisted per sample rather than at the end. A long run that dies
at sample ninety of a hundred should still be worth ninety samples, and the
manifest makes the partial result legible instead of mysterious. The manifest
records the seed, the corpus hash, the models and the exact invocation, so a
number can be traced back to what produced it, which is the property that
makes a benchmark result a measurement rather than an anecdote.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from rlm0.harness.corpus import Corpus, SolverTask
from rlm0.harness.grading import GradingPolicy, grade
from rlm0.harness.report import (
    HarnessVerdict,
    ResultTable,
    SampleRecord,
    build_report,
    classify,
    record_from_dict,
)
from rlm0.run import Attempt, Run, TokenUsage

__all__ = [
    "Attempted",
    "Solver",
    "SolverContractError",
    "SolverResult",
    "SuiteResult",
    "run_suite",
]

HARNESS_VERSION = "0.1.0"
RECORDS_FILENAME = "records.jsonl"
MANIFEST_FILENAME = "manifest.json"
CORPUS_FILENAME = "corpus.json"


class SolverContractError(Exception):
    """A solver returned something its own run accounting does not support.

    Checked rather than trusted. An answer that appears in the result but not
    in the `Run` is an answer that was not paid for, and a harness that
    accepts one cannot claim its cost figures cover the work.
    """


@dataclass(frozen=True, slots=True)
class Attempted:
    """What one attempt concluded, and what it says it read."""

    answer: str | None
    cited_doc_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SolverResult:
    """A solver's output for one sample.

    The `Run` is mandatory and the baseline attempt is reported separately
    from the final one, because the depth-zero row of the report is built from
    it. A solver that will not say what its control answered cannot be placed
    beside one that will.
    """

    run: Run
    final: Attempted
    baseline: Attempted | None = None


@runtime_checkable
class Solver(Protocol):
    """The whole interface a competing implementation has to satisfy."""

    def solve(self, task: SolverTask) -> SolverResult:
        """Answer one task, having seen only the question and the documents."""
        ...

    def describe(self) -> str:
        """Name the system and its configuration, for the manifest."""
        ...


def _check(result: SolverResult, task: SolverTask) -> None:
    run = result.run
    if run.answer is None and result.final.answer is not None:
        raise SolverContractError(
            f"{task.sample_id}: an answer was returned that no attempt in the "
            "run produced, so it was not accounted for"
        )
    if run.answer is not None and result.final.answer != run.answer:
        raise SolverContractError(
            f"{task.sample_id}: the reported answer differs from the answer "
            "the run recorded, so the accounting describes different work"
        )
    if run.baseline is not None and result.baseline is None:
        raise SolverContractError(
            f"{task.sample_id}: the run carries a depth-zero control but the "
            "result does not report what it answered, so no control row can "
            "be built"
        )
    if run.baseline is None and result.baseline is not None:
        raise SolverContractError(
            f"{task.sample_id}: a control result was reported for a run whose "
            "control was waived"
        )
    if (
        run.baseline is not None
        and result.baseline is not None
        and result.baseline.answer != run.baseline.answer
    ):
        raise SolverContractError(
            f"{task.sample_id}: the reported control answer differs from the "
            "one the depth-zero attempt recorded"
        )
    known = set(task.doc_ids)
    unknown = sorted(set(result.final.cited_doc_ids) - known)
    if unknown:
        raise SolverContractError(
            f"{task.sample_id}: cited documents that are not in the context: "
            f"{unknown}"
        )


def _attempt_to_dict(attempt: Attempt) -> dict[str, Any]:
    return {
        "max_depth": attempt.max_depth,
        "outcome": attempt.outcome.value,
        "answer": attempt.answer,
        "detail": attempt.detail,
        "wall_clock_s": attempt.wall_clock_s,
        "cost_usd": attempt.cost_usd,
        "cache_read_ratio": attempt.cache_read_ratio,
        "calls": [
            {
                "role": call.role.value,
                "depth": call.depth,
                "model": call.model,
                "wall_clock_s": call.wall_clock_s,
                "cost_usd": call.cost_usd,
                "cached_prefix": call.cached_prefix,
                "usage": {
                    "input_tokens": call.usage.input_tokens,
                    "output_tokens": call.usage.output_tokens,
                    "cache_read_tokens": call.usage.cache_read_tokens,
                    "cache_write_tokens": call.usage.cache_write_tokens,
                },
            }
            for call in attempt.calls
        ],
    }


def run_to_dict(run: Run) -> dict[str, Any]:
    """Serialise the full run accounting, not a summary of it.

    Every aggregate this harness publishes must be recomputable from the raw
    records, which is only true if the raw records hold everything the
    aggregate was derived from.
    """
    waiver = run.waiver
    return {
        "task": run.task,
        "budget_summary": run.budget_summary,
        "labels": dict(run.labels),
        "waiver": (
            None
            if waiver is None
            else {"reason": waiver.reason, "approved_by": waiver.approved_by}
        ),
        "attempts": [_attempt_to_dict(attempt) for attempt in run.attempts],
        "cost_usd": run.cost_usd,
        "wall_clock_s": run.wall_clock_s,
        "verdict": run.recursion_verdict().verdict.value,
    }


def _models_in(run: Run) -> set[str]:
    return {call.model for call in run.calls}


@dataclass(frozen=True, slots=True)
class SuiteResult:
    """Everything a run of the suite produced, on disk and in memory."""

    corpus_hash: str
    records: tuple[SampleRecord, ...]
    manifest: dict[str, Any]
    out_dir: Path
    policy: GradingPolicy

    def report(self, *, title: str | None = None) -> ResultTable:
        return build_report(
            self.records,
            title=title or f"rlm0 harness  |  {self.manifest['solver']}",
            policy=self.policy,
        )


def _load_existing(path: Path, corpus_hash: str) -> list[SampleRecord]:
    """Read records from a previous, possibly interrupted, run.

    Refuses to resume across a corpus change. Appending results measured on
    different text to one file produces an aggregate that is a measurement of
    nothing, and the failure is silent unless somebody checks.
    """
    if not path.exists():
        return []
    records: list[SampleRecord] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("corpus_hash") != corpus_hash:
            raise SolverContractError(
                f"{path} line {number} was measured on corpus "
                f"{str(payload.get('corpus_hash'))[:12]} but this run is on "
                f"{corpus_hash[:12]}; resuming would pool two different "
                "benchmarks into one number"
            )
        records.append(record_from_dict(payload))
    return records


def _append(path: Path, record: SampleRecord) -> None:
    """Write one record and get it to disk before the next sample starts."""
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_suite(
    corpus: Corpus,
    solver: Solver,
    out_dir: Path,
    *,
    policy: GradingPolicy | None = None,
    invocation: Sequence[str] | None = None,
    now: Callable[[], str] | None = None,
    resume: bool = True,
    only: Iterable[str] | None = None,
) -> SuiteResult:
    """Run every sample, scoring and persisting as it goes.

    `now` is injected rather than read from the clock so that a suite run is
    reproducible byte for byte in a test. `invocation` is recorded verbatim so
    that a published figure names the command that produced it.
    """
    policy = policy or GradingPolicy()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_hash = corpus.content_hash
    records_path = out_dir / RECORDS_FILENAME
    manifest_path = out_dir / MANIFEST_FILENAME

    (out_dir / CORPUS_FILENAME).write_text(corpus.to_json(), encoding="utf-8")

    existing = _load_existing(records_path, corpus_hash) if resume else []
    if not resume and records_path.exists():
        records_path.unlink()
    done = {record.sample_id for record in existing}
    wanted = None if only is None else set(only)

    stamp = now() if now is not None else None
    manifest: dict[str, Any] = {
        "harness_version": HARNESS_VERSION,
        "seed": corpus.spec.seed,
        "corpus_hash": corpus_hash,
        "corpus_spec": {
            key: getattr(corpus.spec, key) for key in corpus.spec.__slots__
        },
        "n_samples": len(corpus.samples),
        "solver": solver.describe(),
        "grading_policy": policy.describe(),
        "invocation": list(invocation) if invocation is not None else [],
        "started_at": stamp,
        "finished_at": None,
        "models": sorted({m for r in existing for m in _models_of_record(r)}),
        "status": "running",
        "n_records": len(existing),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    records = list(existing)
    models = set(manifest["models"])
    for sample in corpus.samples:
        if sample.sample_id in done:
            continue
        if wanted is not None and sample.sample_id not in wanted:
            continue
        task = sample.as_task()
        result = solver.solve(task)
        _check(result, task)

        final_score = grade(
            sample, result.final.answer, result.final.cited_doc_ids, policy=policy
        )
        baseline_attempt = result.run.baseline
        baseline_score = (
            None
            if result.baseline is None
            else grade(
                sample,
                result.baseline.answer,
                result.baseline.cited_doc_ids,
                policy=policy,
            )
        )
        verdict = classify(result.run, final_score, baseline_score)
        record = SampleRecord(
            sample_id=sample.sample_id,
            family=sample.family,
            corpus_hash=corpus_hash,
            final_score=final_score,
            baseline_score=baseline_score,
            harness_verdict=verdict,
            run_verdict=result.run.recursion_verdict().verdict,
            answer=result.final.answer,
            cited_doc_ids=tuple(result.final.cited_doc_ids),
            baseline_answer=None if result.baseline is None else result.baseline.answer,
            cost_usd=result.run.cost_usd,
            wall_clock_s=result.run.wall_clock_s,
            usage=result.run.usage,
            n_calls=len(result.run.calls),
            n_sub_calls=sum(a.n_sub_calls for a in result.run.attempts),
            baseline_cost_usd=(
                None if baseline_attempt is None else baseline_attempt.cost_usd
            ),
            baseline_wall_clock_s=(
                0.0 if baseline_attempt is None else baseline_attempt.wall_clock_s
            ),
            baseline_usage=(
                TokenUsage() if baseline_attempt is None else baseline_attempt.usage
            ),
            baseline_n_calls=(
                0 if baseline_attempt is None else len(baseline_attempt.calls)
            ),
            run=run_to_dict(result.run),
        )
        _append(records_path, record)
        records.append(record)
        models |= _models_in(result.run)

    manifest["models"] = sorted(models)
    manifest["status"] = "complete"
    manifest["n_records"] = len(records)
    manifest["finished_at"] = now() if now is not None else None
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SuiteResult(
        corpus_hash=corpus_hash,
        records=tuple(records),
        manifest=manifest,
        out_dir=out_dir,
        policy=policy,
    )


def _models_of_record(record: SampleRecord) -> set[str]:
    models: set[str] = set()
    for attempt in record.run.get("attempts", []):
        for call in attempt.get("calls", []):
            models.add(str(call["model"]))
    return models


def verdict_counts_of(records: Sequence[SampleRecord]) -> dict[HarnessVerdict, int]:
    counts: dict[HarnessVerdict, int] = {}
    for record in records:
        counts[record.harness_verdict] = counts.get(record.harness_verdict, 0) + 1
    return counts
