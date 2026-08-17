"""The runner has to persist enough that a number can be traced to its cause.

Also the place where the two dishonest solvers are pointed at the harness, on
the principle that a benchmark which cannot fail anything is not measuring
anything.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rlm0.harness.corpus import Corpus, SolverTask
from rlm0.harness.grading import GradingPolicy
from rlm0.harness.report import HarnessVerdict, ReportRefusalError
from rlm0.harness.runner import (
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    SolverContractError,
    SolverResult,
    run_suite,
)

from .fakes import (
    AlwaysWrongSolver,
    BrokenAccountingSolver,
    CheatingSolver,
    Plan,
    ScriptedSolver,
    WaivingSolver,
    answer_key,
    perfect_plans,
)

CLOCK = "2026-01-01T00:00:00Z"


class TestThePerfectSolver:
    def test_it_scores_and_separates_the_regimes(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        solver = ScriptedSolver(perfect_plans(corpus), label="oracle-with-evidence")
        result = run_suite(
            corpus, solver, tmp_path, invocation=["pytest"], now=lambda: CLOCK
        )
        table = result.report()
        text = table.render()

        assert all(record.final_score.supported for record in result.records)
        # Retrieval was answered by the control; aggregation needed the
        # escalation. That split is the thing the corpus exists to expose.
        verdicts = {r.sample_id: r.harness_verdict for r in result.records}
        by_family = {r.sample_id: r.family.regime.value for r in result.records}
        for sample_id, verdict in verdicts.items():
            expected = (
                HarnessVerdict.NOT_ATTEMPTED
                if by_family[sample_id] == "retrieval"
                else HarnessVerdict.HELPED
            )
            assert verdict is expected
        assert "depth 0 (control)" in text
        assert "retrieval:" in text
        assert "aggregation:" in text

    def test_records_and_manifest_are_written(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        solver = ScriptedSolver(perfect_plans(corpus), label="oracle-with-evidence")
        result = run_suite(
            corpus,
            solver,
            tmp_path,
            invocation=["python", "-m", "rlm0.harness", "--seed", "11"],
            now=lambda: CLOCK,
        )
        manifest = json.loads((tmp_path / MANIFEST_FILENAME).read_text("utf-8"))
        assert manifest["seed"] == corpus.spec.seed
        assert manifest["corpus_hash"] == corpus.content_hash
        assert manifest["solver"] == "oracle-with-evidence"
        assert manifest["invocation"][:2] == ["python", "-m"]
        assert manifest["models"] == ["fake-root-model", "fake-sub-model"]
        assert manifest["status"] == "complete"
        assert manifest["started_at"] == CLOCK

        lines = (tmp_path / RECORDS_FILENAME).read_text("utf-8").strip().splitlines()
        assert len(lines) == len(corpus.samples) == len(result.records)
        payload = json.loads(lines[0])
        # The full run accounting, not a summary of it.
        assert payload["run"]["attempts"][0]["calls"][0]["role"] == "root"
        assert payload["run"]["budget_summary"]
        assert "cost_usd" in payload["run"]["attempts"][0]


class TestInterruptionAndResumption:
    def test_a_partial_run_is_resumed_rather_than_repeated(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        plans = perfect_plans(corpus)
        first_two = [s.sample_id for s in corpus.samples[:2]]
        run_suite(
            corpus,
            ScriptedSolver(plans),
            tmp_path,
            only=first_two,
            now=lambda: CLOCK,
        )
        assert (
            len((tmp_path / RECORDS_FILENAME).read_text("utf-8").strip().splitlines())
            == 2
        )

        seen: list[str] = []

        class Counting(ScriptedSolver):
            def solve(self, task: SolverTask) -> SolverResult:
                seen.append(task.sample_id)
                return super().solve(task)

        result = run_suite(corpus, Counting(plans), tmp_path, now=lambda: CLOCK)
        assert set(seen).isdisjoint(first_two)
        assert len(result.records) == len(corpus.samples)

    def test_resuming_across_a_corpus_change_is_refused(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        run_suite(corpus, ScriptedSolver(perfect_plans(corpus)), tmp_path)
        path = tmp_path / RECORDS_FILENAME
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        rows[0]["corpus_hash"] = "0" * 64
        path.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        with pytest.raises(SolverContractError, match="pool two different"):
            run_suite(corpus, ScriptedSolver(perfect_plans(corpus)), tmp_path)


class TestTheHarnessCatchesBadSolvers:
    def test_an_always_wrong_solver_scores_zero(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        result = run_suite(
            corpus, AlwaysWrongSolver(), tmp_path, invocation=["pytest"]
        )
        summary = result.report().rows[-1].summary
        assert summary.answer_accuracy == 0.0
        assert summary.supported_accuracy == 0.0
        assert summary.evidence_recall == 0.0

    def test_a_cheating_solver_is_caught_by_the_evidence_and_the_accounting(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        result = run_suite(
            corpus, CheatingSolver(answer_key(corpus)), tmp_path, invocation=["pytest"]
        )
        table = result.report()
        row = table.rows[-1]
        # Perfect on the number everyone else publishes.
        assert row.summary.answer_accuracy == 1.0
        # Zero on the one that matters.
        assert row.summary.supported_accuracy == 0.0
        assert row.summary.luck_gap == 1.0
        assert table.integrity.n_answers_without_calls == len(corpus.samples)
        assert table.integrity.n_answers_without_citations == len(corpus.samples)
        assert table.integrity.n_correct_but_unsupported == len(corpus.samples)
        text = table.render()
        assert "no model call recorded" in text
        assert "what luck looks like" in text

    def test_a_solver_that_skips_the_control_cannot_be_reported(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        result = run_suite(
            corpus, WaivingSolver(answer_key(corpus)), tmp_path, invocation=["pytest"]
        )
        assert all(
            r.harness_verdict is HarnessVerdict.UNTESTED for r in result.records
        )
        with pytest.raises(ReportRefusalError, match="no depth-zero row"):
            result.report().render()

    def test_an_answer_the_run_never_produced_is_refused(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        with pytest.raises(SolverContractError, match="not accounted for"):
            run_suite(corpus, BrokenAccountingSolver(), tmp_path)

    def test_citing_a_document_that_is_not_in_the_context_is_refused(
        self, corpus: Corpus, tmp_path: Path
    ) -> None:
        plans = perfect_plans(corpus)
        first = corpus.samples[0]
        plans[first.sample_id] = Plan(
            baseline_answer=first.answer, baseline_cites=("DOC-NOT-HERE",)
        )
        with pytest.raises(SolverContractError, match="not in the context"):
            run_suite(corpus, ScriptedSolver(plans), tmp_path)


def test_the_policy_is_carried_into_the_report(
    corpus: Corpus, tmp_path: Path
) -> None:
    policy = GradingPolicy(require_complete_evidence=False, min_evidence_precision=0.9)
    result = run_suite(
        corpus, ScriptedSolver(perfect_plans(corpus)), tmp_path, policy=policy
    )
    assert policy.describe() in result.report().render()
