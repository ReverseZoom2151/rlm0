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
    assert names() == ("oolong-real", "oolong-synth", "ruler-s-niah")
    catalogue = describe_catalogue()
    assert "considered and not adapted:" in catalogue
    assert "AGGBench" in catalogue
