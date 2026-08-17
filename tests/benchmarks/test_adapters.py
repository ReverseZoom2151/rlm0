"""Offline contract tests for every public benchmark adapter."""

from __future__ import annotations

from pathlib import Path

from benchmarks.fixtures import (
    niah_rows,
    oolong_real_rows,
    oolong_synth_rows,
    write_jsonl,
)
from rlm0.benchmarks.niah import RulerNiah, string_match_all
from rlm0.benchmarks.oolong import OolongReal, OolongSynth, score_real, score_synth
from rlm0.benchmarks.registry import describe_catalogue, names
from rlm0.benchmarks.scoring import OfficialItem
from rlm0.benchmarks.suite import RESULT_FILENAME, run_benchmark
from rlm0.harness.corpus import SolverTask
from rlm0.harness.runner import Attempted, SolverResult
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage


class _NoAnswerSolver:
    """A fully accounted solver used to test result persistence offline."""

    def describe(self) -> str:
        return "offline no-answer fixture"

    def solve(self, task: SolverTask) -> SolverResult:
        # The runner only needs the task's question.  This fixture purposely
        # returns no answer, which lets the official scorer prove that it
        # records unanswered rows instead of silently dropping them.
        question = task.question
        call = CallRecord(
            role=Role.ROOT,
            depth=0,
            model="fixture",
            usage=TokenUsage(input_tokens=1, output_tokens=1),
            wall_clock_s=0.0,
            cost_usd=0.0,
        )
        attempt = Attempt(
            max_depth=0,
            outcome=Outcome.ITERATIONS_EXHAUSTED,
            calls=(call,),
            wall_clock_s=0.0,
        )
        run = Run(task=question, attempts=(attempt,), budget_summary="fixture")
        return SolverResult(
            run=run,
            final=Attempted(answer=None),
            baseline=Attempted(answer=None),
        )


def test_ruler_metric_matches_its_substring_contract() -> None:
    assert string_match_all(["the answer is blue"], [("blue",)]) == 100.0
    assert string_match_all(["blue"], [("blue", "green")]) == 50.0


def test_ruler_adapter_loads_mirror_shaped_rows(tmp_path: Path) -> None:
    root = tmp_path / "ruler"
    write_jsonl(root / "4096" / "test-fixture.jsonl", niah_rows())
    suite = RulerNiah().load(root=root)
    assert suite.manifest.n_samples == 2
    assert suite.manifest.fidelity_note
    assert suite.corpus.samples[0].required_doc_ids


def test_oolong_synth_adapter_loads_and_scores_offline_rows(tmp_path: Path) -> None:
    root = tmp_path / "oolong-synth"
    write_jsonl(root / "validation.jsonl", oolong_synth_rows())
    suite = OolongSynth().load(root=root)
    assert suite.manifest.n_samples == 3
    item = OfficialItem("x", "[4]", "ANSWER_TYPE.NUMERIC", {})
    assert score_synth(item, "Answer: 4").score == 1.0


def test_oolong_real_adapter_loads_and_scores_offline_rows(tmp_path: Path) -> None:
    root = tmp_path / "oolong-real"
    write_jsonl(root / "dnd" / "validation.jsonl", oolong_real_rows())
    suite = OolongReal().load(root=root)
    assert suite.manifest.n_samples == len(oolong_real_rows())
    item = OfficialItem("x", "84", "int", {})
    assert score_real(item, r"\boxed{84}").score == 1.0


def test_catalogue_lists_only_adapters_with_a_real_loader() -> None:
    assert names() == (
        "anomalyxl-local",
        "oolong-real",
        "oolong-synth",
        "ruler-s-niah",
    )
    catalogue = describe_catalogue()
    assert "considered and not adapted:" in catalogue
    assert "AGGBench" in catalogue


def test_benchmark_writes_one_reproducibility_index(tmp_path: Path) -> None:
    root = tmp_path / "oolong-synth"
    write_jsonl(root / "validation.jsonl", oolong_synth_rows())
    suite = OolongSynth().load(root=root)

    result = run_benchmark(suite, _NoAnswerSolver(), tmp_path / "out")

    payload = (tmp_path / "out" / RESULT_FILENAME).read_text(encoding="utf-8")
    assert '"format": "rlm0-benchmark-result/v1"' in payload
    assert '"official_scores": "official_scores.json"' in payload
    assert result.official.n_unanswered == suite.manifest.n_samples
