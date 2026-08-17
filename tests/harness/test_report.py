"""The report's job is to refuse.

Every assertion here is about a table that does not get printed. A caveat
under a number is read as a footnote on a result; the point is that the result
must not exist.
"""

from __future__ import annotations

import pytest

from rlm0.harness.corpus import Corpus
from rlm0.harness.grading import GradingPolicy, grade
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
from rlm0.run import TokenUsage, Verdict

from .fakes import Plan, ScriptedSolver, root_call, sub_call

EMPTY_VERDICTS = VerdictCounts(counts={}, raw_counts={})
EMPTY_INTEGRITY = IntegrityReport(
    n_samples=0,
    n_answers_without_calls=0,
    n_answers_without_citations=0,
    n_correct_but_unsupported=0,
)


def _row(
    corpus: Corpus,
    *,
    label: str,
    is_depth_zero: bool = False,
    corpus_hash: str = "hash-a",
    policy: GradingPolicy | None = None,
    n_samples: int = 2,
    cost: float | None = 0.5,
) -> ReportRow:
    policy = policy or GradingPolicy()
    scores = tuple(
        grade(sample, sample.answer, sorted(sample.required_doc_ids), policy=policy)
        for sample in corpus.samples[:n_samples]
    )
    return ReportRow(
        label=label,
        is_depth_zero=is_depth_zero,
        corpus_hash=corpus_hash,
        policy=policy,
        scores=scores,
        cost_usd=cost,
        n_unpriced=0 if cost is not None else n_samples,
        wall_clock_s=12.5,
        usage=TokenUsage(input_tokens=1000, cache_read_tokens=1000, output_tokens=10),
        n_calls=4,
    )


def _table(*rows: ReportRow) -> ResultTable:
    return ResultTable(
        title="t", rows=rows, verdicts=EMPTY_VERDICTS, integrity=EMPTY_INTEGRITY
    )


class TestTheTableRefuses:
    def test_no_depth_zero_row_means_no_headline(self, corpus: Corpus) -> None:
        table = _table(_row(corpus, label="rlm0"))
        with pytest.raises(ReportRefusalError, match="no depth-zero row"):
            table.render()

    def test_rows_from_different_corpora_are_not_comparable(
        self, corpus: Corpus
    ) -> None:
        table = _table(
            _row(corpus, label="control", is_depth_zero=True, corpus_hash="hash-a"),
            _row(corpus, label="rlm0", corpus_hash="hash-b"),
        )
        with pytest.raises(ReportRefusalError, match="different corpora"):
            table.render()

    def test_rows_graded_differently_are_not_comparable(self, corpus: Corpus) -> None:
        table = _table(
            _row(corpus, label="control", is_depth_zero=True),
            _row(
                corpus,
                label="rlm0",
                policy=GradingPolicy(require_complete_evidence=False),
            ),
        )
        with pytest.raises(ReportRefusalError, match="different policies"):
            table.render()

    def test_rows_over_different_samples_are_not_comparable(
        self, corpus: Corpus
    ) -> None:
        table = _table(
            _row(corpus, label="control", is_depth_zero=True, n_samples=2),
            _row(corpus, label="rlm0", n_samples=3),
        )
        with pytest.raises(ReportRefusalError, match="different samples"):
            table.render()

    def test_an_empty_table_reports_nothing(self) -> None:
        with pytest.raises(ReportRefusalError, match="no rows"):
            _table().render()

    def test_a_row_with_no_samples_is_an_average_over_nothing(
        self, corpus: Corpus
    ) -> None:
        empty = ReportRow(
            label="control",
            is_depth_zero=True,
            corpus_hash="hash-a",
            policy=GradingPolicy(),
            scores=(),
            cost_usd=0.0,
            n_unpriced=0,
            wall_clock_s=0.0,
            usage=TokenUsage(),
            n_calls=0,
        )
        with pytest.raises(ReportRefusalError, match="no scored samples"):
            _table(empty).render()


class TestWhatTheTableSays:
    def test_it_renders_with_the_control_beside_the_headline(
        self, corpus: Corpus
    ) -> None:
        text = _table(
            _row(corpus, label="depth 0 (control)", is_depth_zero=True),
            _row(corpus, label="rlm0 escalating"),
        ).render()
        assert "depth 0 (control)" in text
        assert "rlm0 escalating" in text
        assert "$0.5000" in text
        assert "12.5s" in text
        assert "0.50" in text, "cache read ratio must be visible"

    def test_unpriced_is_a_word_and_never_a_zero(self, corpus: Corpus) -> None:
        row = _row(corpus, label="control", is_depth_zero=True, cost=None)
        text = _table(row).render()
        assert "unpriced" in text
        assert "$0.0000" not in text

    def test_the_verdict_distribution_is_printed(self, corpus: Corpus) -> None:
        table = ResultTable(
            title="t",
            rows=(_row(corpus, label="control", is_depth_zero=True),),
            verdicts=VerdictCounts(
                counts={
                    HarnessVerdict.NOT_ATTEMPTED: 3,
                    HarnessVerdict.HELPED: 1,
                    HarnessVerdict.WASTED: 5,
                    HarnessVerdict.HARMED: 1,
                },
                raw_counts={Verdict.HELPED: 1},
            ),
            integrity=EMPTY_INTEGRITY,
        )
        text = table.render()
        assert "not_attempted" in text
        assert "paid for and did not help: 6 of 10" in text


class TestClassification:
    def test_a_run_that_never_escalated_is_not_attempted(
        self, corpus: Corpus
    ) -> None:
        sample = corpus.samples[0]
        solver = ScriptedSolver(
            {
                sample.sample_id: Plan(
                    baseline_answer=sample.answer,
                    baseline_cites=tuple(sorted(sample.required_doc_ids)),
                )
            }
        )
        result = solver.solve(sample.as_task())
        score = grade(sample, sample.answer, sorted(sample.required_doc_ids))
        assert classify(result.run, score, score) is HarnessVerdict.NOT_ATTEMPTED

    def test_escalation_that_fixed_the_answer_helped(self, corpus: Corpus) -> None:
        sample = corpus.samples[0]
        evidence = tuple(sorted(sample.required_doc_ids))
        solver = ScriptedSolver(
            {
                sample.sample_id: Plan(
                    baseline_answer=None,
                    escalate=True,
                    escalated_answer=sample.answer,
                    escalated_cites=evidence,
                )
            }
        )
        result = solver.solve(sample.as_task())
        good = grade(sample, sample.answer, evidence)
        bad = grade(sample, None, ())
        assert classify(result.run, good, bad) is HarnessVerdict.HELPED

    def test_escalation_that_broke_a_correct_answer_harmed(
        self, corpus: Corpus
    ) -> None:
        sample = corpus.samples[0]
        evidence = tuple(sorted(sample.required_doc_ids))
        solver = ScriptedSolver(
            {
                sample.sample_id: Plan(
                    baseline_answer=sample.answer,
                    baseline_cites=evidence,
                    escalate=True,
                    escalated_answer="nonsense",
                    escalated_cites=evidence,
                )
            }
        )
        result = solver.solve(sample.as_task())
        good = grade(sample, sample.answer, evidence)
        bad = grade(sample, "nonsense", evidence)
        assert classify(result.run, bad, good) is HarnessVerdict.HARMED

    def test_escalation_that_changed_nothing_was_wasted(self, corpus: Corpus) -> None:
        sample = corpus.samples[0]
        evidence = tuple(sorted(sample.required_doc_ids))
        solver = ScriptedSolver(
            {
                sample.sample_id: Plan(
                    baseline_answer=sample.answer,
                    baseline_cites=evidence,
                    escalate=True,
                    escalated_answer=sample.answer,
                    escalated_cites=evidence,
                )
            }
        )
        result = solver.solve(sample.as_task())
        score = grade(sample, sample.answer, evidence)
        assert classify(result.run, score, score) is HarnessVerdict.WASTED


def test_build_report_refuses_when_any_run_lacks_a_control(
    corpus: Corpus,
) -> None:
    sample = corpus.samples[0]
    score = grade(sample, sample.answer, sorted(sample.required_doc_ids))
    record = SampleRecord(
        sample_id=sample.sample_id,
        family=sample.family,
        corpus_hash="hash-a",
        final_score=score,
        baseline_score=None,
        harness_verdict=HarnessVerdict.UNTESTED,
        run_verdict=Verdict.UNTESTED,
        answer=sample.answer,
        cited_doc_ids=tuple(sorted(sample.required_doc_ids)),
        baseline_answer=None,
        cost_usd=0.1,
        wall_clock_s=1.0,
        usage=TokenUsage(input_tokens=10),
        n_calls=2,
        n_sub_calls=1,
        baseline_cost_usd=None,
        baseline_wall_clock_s=0.0,
        baseline_usage=TokenUsage(),
        baseline_n_calls=0,
        run={},
    )
    table = build_report([record], title="t", policy=GradingPolicy())
    with pytest.raises(ReportRefusalError, match="no depth-zero row"):
        table.render()


def test_calls_are_attributed_by_role_in_the_fakes() -> None:
    # Guards the fixtures themselves: a sub-call that reads nothing from cache
    # would make the cache diagnostic in the report vacuous.
    assert root_call().usage.cache_read_tokens == 0
    assert sub_call().usage.cache_read_tokens > 0
