"""Offline tests for the local, strict AnomalyXL-compatible adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rlm0.benchmarks.anomalyxl import (
    ANOMALYXL_LOCAL_FORMAT,
    AnomalyXL,
    parse_prediction,
    score_prediction,
)
from rlm0.benchmarks.dataset import BenchmarkDataError


def _write_local_split(root: Path, rows: list[dict[str, object]]) -> None:
    root.mkdir(parents=True)
    data = root / "data.jsonl"
    data.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    digest = hashlib.sha256(data.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": ANOMALYXL_LOCAL_FORMAT,
                "revision": "local-fixture-2026-08-17",
                "split": "test",
                "data_file": "data.jsonl",
                "sha256": digest,
                "n_rows": len(rows),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _rows() -> list[dict[str, object]]:
    return [
        {
            "id": "locate",
            "split": "test",
            "context": "series A: 0, 0, 8, 9, 0, 0",
            "question": "Is there an anomaly, and where?",
            "category": "localize",
            "label": {"present": True, "start": 2, "end": 4, "length": 6},
        },
        {
            "id": "classify",
            "split": "test",
            "context": "series A shifts abruptly at 5",
            "question": "Classify and locate the anomaly.",
            "category": "classify_with_evidence",
            "label": {"kind": "Level shift", "start": 5, "end": 9, "length": 12},
        },
        {
            "id": "magnitude",
            "split": "test",
            "context": "series A has a 2.0 sigma spike",
            "question": "Measure the magnitude.",
            "category": "measure_magnitude",
            "label": {"magnitude_sigma": 2.0},
        },
        {
            "id": "channels",
            "split": "test",
            "context": "channel east spikes at 3, channel west shifts at 8",
            "question": "Locate every channel anomaly.",
            "category": "localize_all_channels",
            "label": {
                "anomalies": [
                    {"channel": "east", "start": 3, "end": 6},
                    {"channel": "west", "start": 8, "end": 12},
                ]
            },
        },
        {
            "id": "lag",
            "split": "test",
            "context": "series A leads B by three samples",
            "question": "Determine direction and lag.",
            "category": "lead_lag_with_magnitude",
            "label": {"direction": "lead", "lag_samples": 3, "length": 100},
        },
    ]


def test_loader_requires_a_manifest_hash_and_keeps_labels_out_of_tasks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "anomalyxl"
    _write_local_split(root, _rows())

    suite = AnomalyXL(chunk_chars=16).load(root=root)

    assert suite.manifest.n_samples == 5
    assert suite.manifest.revision == "local-fixture-2026-08-17"
    assert suite.scoreboard.fidelity_note
    assert '"start":2' not in suite.corpus.samples[0].as_task().context()
    assert suite.corpus.samples[0].required_doc_ids


def test_loader_refuses_data_that_no_longer_matches_its_lock(tmp_path: Path) -> None:
    root = tmp_path / "anomalyxl"
    _write_local_split(root, _rows())
    with (root / "data.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(BenchmarkDataError, match="sha256"):
        AnomalyXL().load(root=root)


def test_strict_parser_rejects_prose_missing_fields_and_nan() -> None:
    assert parse_prediction('{"present": true, "start": 2, "end": 4}', "localize")
    assert (
        parse_prediction('Answer: {"present": true, "start": 2, "end": 4}', "localize")
        is None
    )
    assert parse_prediction('{"present": true}', "localize") is None
    assert parse_prediction('{"magnitude_sigma": NaN}', "measure_magnitude") is None


def test_localization_and_classification_metrics_are_task_specific() -> None:
    localize = score_prediction(
        "localize",
        {"present": True, "start": 10, "end": 20, "length": 100},
        '{"present": true, "start": 15, "end": 25}',
    )
    assert localize.primary == pytest.approx(1 / 3)
    assert localize.values["presence_accuracy"] == 1.0
    assert localize.values["start_mae_fraction"] == pytest.approx(0.05)

    classified = score_prediction(
        "classify_with_evidence",
        {"kind": "Level shift", "start": 10, "end": 20},
        '{"kind": "Level shift", "start": 10, "end": 20}',
    )
    assert classified.primary == 1.0
    wrong_kind = score_prediction(
        "classify_with_evidence",
        {"kind": "Level shift", "start": 10, "end": 20},
        '{"kind": "Spike", "start": 10, "end": 20}',
    )
    assert wrong_kind.primary == 0.0


def test_magnitude_multichannel_and_leadlag_metrics() -> None:
    magnitude = score_prediction(
        "measure_magnitude", {"magnitude_sigma": 2.0}, '{"magnitude_sigma": 2.2}'
    )
    assert magnitude.values["relative_error"] == pytest.approx(0.1)
    assert (
        magnitude.values["within_10pct"] == 0.0
    )  # float representation is deliberately exact
    assert magnitude.primary == pytest.approx(0.8)

    channels = score_prediction(
        "localize_all_channels",
        {
            "anomalies": [
                {"channel": "a", "start": 1, "end": 5},
                {"channel": "b", "start": 7, "end": 9},
            ]
        },
        (
            '{"anomalies": [{"channel": "a", "start": 1, "end": 5}, '
            '{"channel": "wrong", "start": 7, "end": 9}]}'
        ),
    )
    assert channels.values["precision"] == pytest.approx(0.5)
    assert channels.values["recall"] == pytest.approx(0.5)
    assert channels.primary == pytest.approx(0.5)

    lag = score_prediction(
        "lead_lag_with_magnitude",
        {"direction": "lead", "lag_samples": 10, "length": 1000},
        '{"direction": "lead", "lag_samples": 15}',
    )
    assert lag.values["within_1pct"] == 1.0
    assert lag.primary == pytest.approx(0.9)


def test_scoreboard_exposes_metrics_without_claiming_public_parity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "anomalyxl"
    _write_local_split(root, _rows())
    suite = AnomalyXL().load(root=root)
    answers = {
        "anomalyxl-locate": '{"present": true, "start": 2, "end": 4}',
        "anomalyxl-classify": '{"kind": "Level shift", "start": 5, "end": 9}',
        "anomalyxl-magnitude": '{"magnitude_sigma": 2.0}',
        "anomalyxl-channels": (
            '{"anomalies": [{"channel": "east", "start": 3, "end": 6}, '
            '{"channel": "west", "start": 8, "end": 12}]}'
        ),
        "anomalyxl-lag": '{"direction": "lead", "lag_samples": 3}',
    }
    results = suite.scoreboard.score_all(answers)
    assert all(result.score == 1.0 for result in results)
    assert '"f1": 1.0' in results[3].parsed
    assert suite.scoreboard.summarise(results).score == 1.0
