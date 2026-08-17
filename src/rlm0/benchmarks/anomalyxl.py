"""A local, hash-locked adapter for precise time-series anomaly questions.

This is a clean-room adapter informed by the task shapes described by
AnomalyXL/TimeRLM.  It does not import, vendor, download, or claim to run that
project's data.  A caller supplies a small local dataset plus a manifest that
names the revision and hashes the exact JSONL bytes being scored.

The local format is intentionally narrow::

    manifest.json
      {"format": "rlm0-anomalyxl-local/v1", "revision": "...",
       "split": "test", "data_file": "data.jsonl", "sha256": "...",
       "n_rows": 12}

    data.jsonl
      {"id": "...", "split": "test", "context": "...",
       "question": "...", "category": "localize", "label": {...}}

The model sees only context and question.  Labels stay in the scoreboard.
Predictions must be one JSON object with no surrounding prose.  This keeps a
formatting failure visible as a parsing failure instead of silently recovering
an answer from a model's chain of thought.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm0.benchmarks.context import DEFAULT_CHUNK_CHARS, chunk_context
from rlm0.benchmarks.dataset import (
    BenchmarkDataError,
    DatasetRequirement,
    DatasetUnavailableError,
    load_files,
    resolve_root,
)
from rlm0.benchmarks.scoring import Fidelity, OfficialItem, OfficialResult, Scoreboard
from rlm0.benchmarks.suite import BenchmarkManifest, BenchmarkSuite, corpus_spec_for
from rlm0.harness.corpus import Corpus, Sample, TaskFamily

__all__ = [
    "ANOMALYXL_LOCAL_FORMAT",
    "AnomalyMetrics",
    "AnomalyXL",
    "parse_prediction",
    "score_prediction",
]

ANOMALYXL_LOCAL_FORMAT = "rlm0-anomalyxl-local/v1"
_MANIFEST_NAME = "manifest.json"
_DATA_NAME = "data.jsonl"
_CATEGORIES = frozenset(
    {
        "localize",
        "classify_with_evidence",
        "measure_magnitude",
        "localize_all_channels",
        "lead_lag_with_magnitude",
    }
)
_NONE_KIND = "No anomaly"


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _index(value: object, field: str) -> int:
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or int(value) != float(value)
    ):
        raise BenchmarkDataError(f"{field} must be a finite integer")
    return int(value)


def _interval_iou(left: int, right: int, other_left: int, other_right: int) -> float:
    """Intersection over union for half-open, non-empty index intervals."""
    if left >= right or other_left >= other_right:
        return 0.0
    overlap = max(0, min(right, other_right) - max(left, other_left))
    union = (right - left) + (other_right - other_left) - overlap
    return overlap / union if union else 0.0


def _window(
    label: Mapping[str, Any], *, required: bool = True
) -> tuple[int, int] | None:
    if "start" not in label or "end" not in label:
        if required:
            raise BenchmarkDataError("a present anomaly needs integer start and end")
        return None
    start = _index(label["start"], "start")
    end = _index(label["end"], "end")
    if start >= end:
        raise BenchmarkDataError("an anomaly interval must satisfy start < end")
    return start, end


def _validate_events(events: object, *, where: str) -> list[dict[str, Any]]:
    if not isinstance(events, list):
        raise BenchmarkDataError(f"{where}.anomalies must be a list")
    checked: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        if not isinstance(event, dict) or not isinstance(event.get("channel"), str):
            raise BenchmarkDataError(
                f"{where}.anomalies[{index}] needs a string channel"
            )
        interval = _window(event)
        assert interval is not None
        start, end = interval
        checked.append({"channel": event["channel"], "start": start, "end": end})
    return checked


def _positive_length(label: Mapping[str, Any]) -> int:
    """Read the series length needed to normalize a lead or lag error."""

    if "length" not in label:
        raise BenchmarkDataError("lead-lag label needs a positive integer length")
    length = _index(label["length"], "length")
    if length < 1:
        raise BenchmarkDataError("lead-lag label length must be at least one")
    return length


def _validate_label(
    category: str, label: object, *, require_length: bool = False
) -> dict[str, Any]:
    if not isinstance(label, dict):
        raise BenchmarkDataError(f"{category} label must be an object")
    result = dict(label)
    if category == "localize":
        if not isinstance(result.get("present"), bool):
            raise BenchmarkDataError("localize label needs boolean present")
        if result["present"]:
            _window(result)
        return result
    if category == "classify_with_evidence":
        kind = result.get("kind")
        if not isinstance(kind, str) or not kind:
            raise BenchmarkDataError(
                "classify_with_evidence label needs a non-empty kind"
            )
        if kind != _NONE_KIND:
            _window(result)
        return result
    if category == "measure_magnitude":
        value = result.get("magnitude_sigma")
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise BenchmarkDataError("magnitude_sigma must be a positive finite number")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise BenchmarkDataError("magnitude_sigma must be a positive finite number")
        return result
    if category == "localize_all_channels":
        result["anomalies"] = _validate_events(result.get("anomalies"), where=category)
        return result
    if category == "lead_lag_with_magnitude":
        direction = result.get("direction")
        if direction not in {"lead", "lag", "independent"}:
            raise BenchmarkDataError(
                "lead-lag direction must be lead, lag, or independent"
            )
        if direction != "independent":
            lag = _index(result.get("lag_samples"), "lag_samples")
            if lag < 0:
                raise BenchmarkDataError("lag_samples must not be negative")
            if require_length:
                length = _positive_length(result)
                if lag > length:
                    raise BenchmarkDataError(
                        "lag_samples must not exceed the series length"
                    )
        return result
    raise BenchmarkDataError(f"unknown AnomalyXL category {category!r}")


def parse_prediction(text: str | None, category: str) -> dict[str, Any] | None:
    """Parse exactly one category-valid JSON object, or return ``None``.

    No brace extraction and no prose fallback: strict output is part of the
    benchmark contract.  Invalid model output scores zero, while invalid gold
    input is rejected by :func:`AnomalyXL.load`.
    """
    if text is None:
        return None
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    try:
        return _validate_label(category, value)
    except BenchmarkDataError:
        return None


@dataclass(frozen=True, slots=True)
class AnomalyMetrics:
    """Primary and diagnostic metrics for one strictly parsed answer."""

    primary: float
    values: Mapping[str, float]

    def to_dict(self) -> dict[str, float]:
        return {"primary": self.primary, **self.values}


def _zero(category: str) -> AnomalyMetrics:
    names = {
        "localize": (
            "presence_accuracy",
            "iou",
            "start_mae_fraction",
            "end_mae_fraction",
        ),
        "classify_with_evidence": ("kind_accuracy", "iou", "kind_x_iou"),
        "measure_magnitude": ("relative_error", "within_10pct", "within_25pct"),
        "localize_all_channels": ("precision", "recall", "f1", "tp", "fp", "fn"),
        "lead_lag_with_magnitude": (
            "direction_accuracy",
            "lag_mae_fraction",
            "within_1pct",
            "within_5pct",
        ),
    }[category]
    values = {name: 0.0 for name in names}
    if category in {"measure_magnitude", "lead_lag_with_magnitude"}:
        values[
            "relative_error" if category == "measure_magnitude" else "lag_mae_fraction"
        ] = 1.0
    return AnomalyMetrics(0.0, values)


def _series_length(label: Mapping[str, Any]) -> int:
    value = label.get("length", label.get("_L"))
    if value is None:
        return 0
    try:
        length = _index(value, "length")
    except BenchmarkDataError:
        return 0
    return max(0, length)


def _score_localize(
    pred: Mapping[str, Any] | None, gold: Mapping[str, Any]
) -> AnomalyMetrics:
    if pred is None:
        return _zero("localize")
    gold_present = bool(gold["present"])
    pred_present = bool(pred["present"])
    values = {
        "presence_accuracy": float(pred_present == gold_present),
        "iou": 0.0,
        "start_mae_fraction": 1.0,
        "end_mae_fraction": 1.0,
    }
    if not gold_present and not pred_present:
        values.update(iou=1.0, start_mae_fraction=0.0, end_mae_fraction=0.0)
        return AnomalyMetrics(1.0, values)
    if gold_present and pred_present:
        gold_start, gold_end = _window(gold) or (0, 0)
        pred_start, pred_end = _window(pred) or (0, 0)
        iou = _interval_iou(gold_start, gold_end, pred_start, pred_end)
        scale = _series_length(gold)
        values["iou"] = iou
        if scale:
            values["start_mae_fraction"] = abs(pred_start - gold_start) / scale
            values["end_mae_fraction"] = abs(pred_end - gold_end) / scale
        return AnomalyMetrics(iou, values)
    return AnomalyMetrics(0.0, values)


def _score_classify(
    pred: Mapping[str, Any] | None, gold: Mapping[str, Any]
) -> AnomalyMetrics:
    if pred is None:
        return _zero("classify_with_evidence")
    kind_ok = float(pred["kind"] == gold["kind"])
    values = {"kind_accuracy": kind_ok, "iou": 0.0, "kind_x_iou": 0.0}
    if gold["kind"] == _NONE_KIND or pred["kind"] == _NONE_KIND:
        if kind_ok:
            values.update(iou=1.0, kind_x_iou=1.0)
            return AnomalyMetrics(1.0, values)
        return AnomalyMetrics(0.0, values)
    gold_start, gold_end = _window(gold) or (0, 0)
    pred_start, pred_end = _window(pred) or (0, 0)
    iou = _interval_iou(gold_start, gold_end, pred_start, pred_end)
    values.update(iou=iou, kind_x_iou=kind_ok * iou)
    return AnomalyMetrics(values["kind_x_iou"], values)


def _score_magnitude(
    pred: Mapping[str, Any] | None, gold: Mapping[str, Any]
) -> AnomalyMetrics:
    if pred is None:
        return _zero("measure_magnitude")
    error = abs(
        float(pred["magnitude_sigma"]) - float(gold["magnitude_sigma"])
    ) / float(gold["magnitude_sigma"])
    values = {
        "relative_error": error,
        "within_10pct": float(error <= 0.1),
        "within_25pct": float(error <= 0.25),
    }
    return AnomalyMetrics(max(0.0, 1.0 - error / 0.5), values)


def _score_multichannel(
    pred: Mapping[str, Any] | None, gold: Mapping[str, Any]
) -> AnomalyMetrics:
    if pred is None:
        return _zero("localize_all_channels")
    gold_events = _validate_events(gold["anomalies"], where="gold")
    pred_events = _validate_events(pred["anomalies"], where="prediction")
    candidates: list[tuple[float, int, int]] = []
    for gi, expected in enumerate(gold_events):
        for pi, actual in enumerate(pred_events):
            if expected["channel"] == actual["channel"]:
                iou = _interval_iou(
                    expected["start"], expected["end"], actual["start"], actual["end"]
                )
                if iou >= 0.3:
                    candidates.append((iou, gi, pi))
    used_gold: set[int] = set()
    used_pred: set[int] = set()
    for _, gi, pi in sorted(candidates, reverse=True):
        if gi not in used_gold and pi not in used_pred:
            used_gold.add(gi)
            used_pred.add(pi)
    tp = len(used_gold)
    fp, fn = len(pred_events) - tp, len(gold_events) - tp
    if not gold_events and not pred_events:
        precision = recall = f1 = 1.0
    else:
        precision = tp / len(pred_events) if pred_events else 0.0
        recall = tp / len(gold_events) if gold_events else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    values = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }
    return AnomalyMetrics(f1, values)


def _score_leadlag(
    pred: Mapping[str, Any] | None, gold: Mapping[str, Any]
) -> AnomalyMetrics:
    if pred is None:
        return _zero("lead_lag_with_magnitude")
    correct = float(pred["direction"] == gold["direction"])
    values = {
        "direction_accuracy": correct,
        "lag_mae_fraction": 1.0,
        "within_1pct": 0.0,
        "within_5pct": 0.0,
    }
    if gold["direction"] == "independent":
        if correct:
            values.update(lag_mae_fraction=0.0, within_1pct=1.0, within_5pct=1.0)
        return AnomalyMetrics(correct, values)
    if not correct:
        return AnomalyMetrics(0.0, values)
    length = _series_length(gold)
    if not length:
        return AnomalyMetrics(0.0, values)
    error = abs(int(pred["lag_samples"]) - int(gold["lag_samples"])) / length
    values.update(
        lag_mae_fraction=error,
        within_1pct=float(error <= 0.01),
        within_5pct=float(error <= 0.05),
    )
    return AnomalyMetrics(max(0.0, 1.0 - min(1.0, error / 0.05)), values)


def score_prediction(
    category: str, gold: Mapping[str, Any], output: str | None
) -> AnomalyMetrics:
    """Score one strict prediction using local, task-specific labels."""
    checked_gold = _validate_label(category, gold, require_length=True)
    prediction = parse_prediction(output, category)
    dispatch = {
        "localize": _score_localize,
        "classify_with_evidence": _score_classify,
        "measure_magnitude": _score_magnitude,
        "localize_all_channels": _score_multichannel,
        "lead_lag_with_magnitude": _score_leadlag,
    }
    return dispatch[category](prediction, checked_gold)


def _score_item(item: OfficialItem, output: str | None) -> OfficialResult:
    category = item.answer_type
    label = item.extra["label"]
    assert isinstance(label, Mapping)
    metrics = score_prediction(category, label, output)
    parsed = parse_prediction(output, category)
    return OfficialResult(
        sample_id=item.sample_id,
        score=metrics.primary,
        parsed=json.dumps(metrics.to_dict(), sort_keys=True),
        parse_confidence="high" if parsed is not None else "low",
        answered=output is not None,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_lock(root: Path, split: str) -> tuple[dict[str, Any], Path]:
    path = root / _MANIFEST_NAME
    if not path.is_file():
        raise DatasetUnavailableError(
            AnomalyXL().requirement(split=split), root, f"{_MANIFEST_NAME} is missing"
        )
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkDataError(f"{path} is not valid JSON") from exc
    if not isinstance(lock, dict) or lock.get("format") != ANOMALYXL_LOCAL_FORMAT:
        raise BenchmarkDataError(f"{path} must use {ANOMALYXL_LOCAL_FORMAT}")
    if (
        lock.get("split") != split
        or not isinstance(lock.get("revision"), str)
        or not lock["revision"]
    ):
        raise BenchmarkDataError(
            f"{path} must name this split and a non-empty revision"
        )
    data_file = lock.get("data_file")
    if not isinstance(data_file, str) or Path(data_file).name != data_file:
        raise BenchmarkDataError(f"{path} data_file must be a plain filename")
    data_path = root / data_file
    if not data_path.is_file():
        raise DatasetUnavailableError(
            AnomalyXL().requirement(split=split), root, f"{data_file} is missing"
        )
    expected = lock.get("sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or _sha256(data_path) != expected
    ):
        raise BenchmarkDataError(f"{path} sha256 does not match {data_file}")
    if not isinstance(lock.get("n_rows"), int) or lock["n_rows"] < 1:
        raise BenchmarkDataError(f"{path} n_rows must be a positive integer")
    return lock, data_path


@dataclass(frozen=True, slots=True)
class AnomalyXL:
    """Load and score a locally pinned AnomalyXL-compatible precise split."""

    chunk_chars: int = DEFAULT_CHUNK_CHARS

    @property
    def name(self) -> str:
        return "anomalyxl-local"

    def requirement(self, *, split: str) -> DatasetRequirement:
        if not split:
            raise ValueError("split must not be empty")
        return DatasetRequirement(
            benchmark=self.name,
            source="local, user-pinned AnomalyXL-compatible JSONL",
            revision="declared in manifest.json",
            config=ANOMALYXL_LOCAL_FORMAT,
            split=split,
            patterns=(_MANIFEST_NAME, _DATA_NAME),
            download=(
                "Place manifest.json and data.jsonl under the path above; "
                "this adapter never downloads benchmark data.",
            ),
            notes=("manifest.json must hash and count the exact local JSONL file",),
        )

    def answer_instruction(self) -> str:
        return (
            "Return exactly one JSON object and no prose. Its fields depend "
            "on the task category."
        )

    def load(
        self,
        *,
        split: str = "test",
        root: Path | None = None,
        limit: int | None = None,
        expected_hash: str | None = None,
    ) -> BenchmarkSuite:
        requirement = self.requirement(split=split)
        where = resolve_root(self.name, root)
        lock, data_path = _load_lock(where, split)
        local_requirement = DatasetRequirement(
            benchmark=requirement.benchmark,
            source=requirement.source,
            revision=str(lock["revision"]),
            config=requirement.config,
            split=split,
            patterns=(data_path.name,),
            download=requirement.download,
        )
        files = load_files(local_requirement, where, expected_hash=expected_hash)
        if len(files.rows) != lock["n_rows"]:
            raise BenchmarkDataError("manifest n_rows does not match the JSONL rows")

        samples: list[Sample] = []
        items: dict[str, OfficialItem] = {}
        seen: set[str] = set()
        for index, row in enumerate(files.rows):
            required = ("id", "split", "context", "question", "category", "label")
            missing = [key for key in required if key not in row]
            if missing:
                raise BenchmarkDataError(f"row {index} is missing {missing}")
            if row["split"] != split:
                raise BenchmarkDataError(
                    f"row {index} names split {row['split']!r}, not {split!r}"
                )
            sample_id = str(row["id"])
            if not sample_id or sample_id in seen:
                raise BenchmarkDataError(f"row {index} has a missing or duplicate id")
            seen.add(sample_id)
            category = str(row["category"])
            if category not in _CATEGORIES:
                raise BenchmarkDataError(
                    f"row {index} has unsupported category {category!r}"
                )
            if not isinstance(row["context"], str) or not isinstance(
                row["question"], str
            ):
                raise BenchmarkDataError(
                    f"row {index} context and question must be strings"
                )
            label = _validate_label(category, row["label"], require_length=True)
            docs = chunk_context(
                row["context"], f"anomalyxl-{sample_id}", target_chars=self.chunk_chars
            )
            scoped_id = f"anomalyxl-{sample_id}"
            samples.append(
                Sample(
                    sample_id=scoped_id,
                    family=TaskFamily.AGGREGATE_ARGMAX,
                    question=row["question"],
                    documents=docs,
                    answer=json.dumps(label, sort_keys=True, separators=(",", ":")),
                    required_doc_ids=frozenset(doc.doc_id for doc in docs),
                )
            )
            items[scoped_id] = OfficialItem(scoped_id, "", category, {"label": label})
            if limit is not None and len(samples) >= limit:
                break
        if not samples:
            raise BenchmarkDataError(
                "the requested AnomalyXL local slice has no samples"
            )
        scoreboard = Scoreboard(
            metric="mean category-primary score",
            fidelity=Fidelity.APPROXIMATES,
            fidelity_note=(
                "clean-room scoring over the local rlm0-anomalyxl-local/v1 schema; "
                "it is not a claim of parity with an external AnomalyXL release"
            ),
            items=items,
            scorer=_score_item,
        )
        return BenchmarkSuite(
            corpus=Corpus(
                spec=corpus_spec_for(files.content_hash), samples=tuple(samples)
            ),
            scoreboard=scoreboard,
            manifest=BenchmarkManifest(
                benchmark=self.name,
                source=requirement.source,
                revision=str(lock["revision"]),
                config=ANOMALYXL_LOCAL_FORMAT,
                split=split,
                dataset_hash=files.content_hash,
                files=files.relative_paths,
                n_samples=len(samples),
                official_metric=scoreboard.metric,
                fidelity=scoreboard.fidelity,
                fidelity_note=scoreboard.fidelity_note,
                deviations=(
                    "local data is required and checked against manifest.json; "
                    "no data is downloaded",
                    "contexts are chunked into identified documents for the "
                    "rlm0 evidence grader",
                    "the reported metric is a local clean-room mean, not an "
                    "external leaderboard claim",
                ),
            ),
        )
