"""Comparative reporting stays tied to raw, paired measurements.

The harness must make it possible to compare depth zero, an escalating RLM,
and an external baseline without ever turning an unmeasured run into a result.
These tests use only scripted local solvers: they exercise report assembly and
its refusals, not a benchmark claim.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rlm0.harness.corpus import Corpus
from rlm0.harness.grading import GradingPolicy, grade
from rlm0.harness.report import (
    CostBand,
    IntegrityReport,
    ReportRefusalError,
    ReportRow,
    ResultTable,
    SampleRecord,
    VerdictCounts,
    build_comparison,
    noise_floor,
)
from rlm0.harness.runner import run_suite
from rlm0.run import TokenUsage

from .fakes import AlwaysWrongSolver, ScriptedSolver, perfect_plans


def _records(corpus: Corpus, out_dir: Path) -> tuple[SampleRecord, ...]:
    return run_suite(
        corpus,
        ScriptedSolver(perfect_plans(corpus), label="scripted rlm"),
        out_dir,
    ).records


def test_depth_zero_and_escalating_rows_are_paired(
    corpus: Corpus, tmp_path: Path
) -> None:
    """One RLM suite yields both rows from the same sample-level runs."""
    result = run_suite(
        corpus,
        ScriptedSolver(perfect_plans(corpus), label="scripted rlm"),
        tmp_path,
    )
    table = result.report()

    assert [row.label for row in table.rows] == [
        "depth 0 (control)",
        "rlm0 escalating",
    ]
    assert table.rows[0].sample_ids == table.rows[1].sample_ids
    assert table.delta("rlm0 escalating").incumbent == "depth 0 (control)"


def test_repeated_runs_supply_a_noise_floor(corpus: Corpus, tmp_path: Path) -> None:
    first = _records(corpus, tmp_path / "first")
    second = _records(corpus, tmp_path / "second")

    floor = noise_floor("scripted rlm", [first, second])

    assert floor.n_samples == len(corpus.samples)
    assert floor.floor == 0.0
    assert floor.n_flipping_samples == 0


def test_noise_floor_rejects_duplicate_or_cross_corpus_records(
    corpus: Corpus, tmp_path: Path
) -> None:
    records = _records(corpus, tmp_path / "records")

    with pytest.raises(ReportRefusalError, match="more than once"):
        noise_floor("duplicate", [records, (*records, records[0])])

    other_corpus = tuple(
        replace(record, corpus_hash="other-corpus") for record in records
    )
    with pytest.raises(ReportRefusalError, match="different corpora"):
        noise_floor("cross-corpus", [records, other_corpus])


def test_external_baseline_has_the_same_matched_cost_hooks(
    corpus: Corpus, tmp_path: Path
) -> None:
    """A baseline joins the table only through the same raw-record route."""
    rlm_records = _records(corpus, tmp_path / "rlm")
    baseline_records = run_suite(
        corpus, AlwaysWrongSolver(), tmp_path / "baseline"
    ).records

    table = build_comparison(
        [
            ("depth 0 external baseline", baseline_records),
            ("rlm0 escalating", rlm_records),
        ],
        title="comparison fixture",
        policy=GradingPolicy(),
        noise=noise_floor("scripted rlm", [rlm_records, rlm_records]),
        band=CostBand(tolerance=0.1),
        reference_label="depth 0 external baseline",
    )

    text = table.render()
    delta = table.delta("rlm0 escalating")
    assert "accuracy at matched cost" in text
    assert delta.challenger == "rlm0 escalating"
    assert not delta.cost_matched


def test_table_rejects_a_duplicate_sample_in_a_row(corpus: Corpus) -> None:
    sample = corpus.samples[0]
    score = grade(sample, sample.answer, sample.required_doc_ids)
    duplicate = ReportRow(
        label="depth 0 (control)",
        is_depth_zero=True,
        corpus_hash="fixture",
        policy=GradingPolicy(),
        scores=(score, score),
        cost_usd=0.01,
        n_unpriced=0,
        wall_clock_s=1.0,
        usage=TokenUsage(),
        n_calls=1,
    )
    table = ResultTable(
        title="fixture",
        rows=(duplicate,),
        verdicts=VerdictCounts(counts={}, raw_counts={}),
        integrity=IntegrityReport(0, 0, 0, 0),
    )
    with pytest.raises(ReportRefusalError, match="more than once"):
        table.render()
