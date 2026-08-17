"""Immutable records for optional research strategies.

The stable runtime produces a :class:`rlm0.run.Run`.  Research strategies can
produce several such runs, branch, retry, or hand work to a fresh root.  They
must not turn that flexibility into an excuse to lose the paired depth-zero
control.  These records keep the stable control alongside every experimental
trial and make configuration identity explicit.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from rlm0.run import Run

__all__ = [
    "RESEARCH_SCHEMA_VERSION",
    "ResearchRun",
    "ResearchStage",
    "ResearchTrial",
    "canonical_json",
    "fingerprint",
    "research_run_from_dict",
    "run_from_dict",
    "run_to_dict",
]

RESEARCH_SCHEMA_VERSION = 1


def canonical_json(value: object) -> str:
    """Encode JSON data in the one spelling used for fingerprints."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(value: object) -> str:
    """Return a content fingerprint for canonical JSON-compatible data."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_fingerprint(value: str, field: str) -> None:
    valid = len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
    if not valid:
        raise ValueError(f"{field} must be a lowercase SHA-256 fingerprint")


def run_to_dict(run: Run) -> dict[str, Any]:
    """Serialise a Run without adding research-only summary fields."""
    from rlm0.harness.runner import run_to_dict as _run_to_dict

    return _run_to_dict(run)


def run_from_dict(payload: Mapping[str, Any]) -> Run:
    """Rebuild an immutable Run for replay without invoking any provider."""
    from rlm0.run import Attempt, BaselineWaiver, CallRecord, Outcome, Role, TokenUsage

    attempts: list[Attempt] = []
    raw_attempts = payload.get("attempts")
    if not isinstance(raw_attempts, list):
        raise ValueError("run record needs an attempts list")
    for raw_attempt in raw_attempts:
        if not isinstance(raw_attempt, Mapping):
            raise ValueError("run attempt must be an object")
        raw_calls = raw_attempt.get("calls")
        if not isinstance(raw_calls, list):
            raise ValueError("run attempt needs a calls list")
        calls: list[CallRecord] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, Mapping):
                raise ValueError("call record must be an object")
            raw_usage = raw_call.get("usage")
            if not isinstance(raw_usage, Mapping):
                raise ValueError("call record needs usage")
            calls.append(
                CallRecord(
                    role=Role(str(raw_call["role"])),
                    depth=int(raw_call["depth"]),
                    model=str(raw_call["model"]),
                    usage=TokenUsage(
                        input_tokens=int(raw_usage["input_tokens"]),
                        output_tokens=int(raw_usage["output_tokens"]),
                        cache_read_tokens=int(raw_usage["cache_read_tokens"]),
                        cache_write_tokens=int(raw_usage["cache_write_tokens"]),
                    ),
                    wall_clock_s=float(raw_call["wall_clock_s"]),
                    cost_usd=(
                        None
                        if raw_call.get("cost_usd") is None
                        else float(raw_call["cost_usd"])
                    ),
                    cached_prefix=bool(raw_call.get("cached_prefix", False)),
                )
            )
        attempts.append(
            Attempt(
                max_depth=int(raw_attempt["max_depth"]),
                outcome=Outcome(str(raw_attempt["outcome"])),
                calls=tuple(calls),
                wall_clock_s=float(raw_attempt["wall_clock_s"]),
                answer=(
                    None
                    if raw_attempt.get("answer") is None
                    else str(raw_attempt["answer"])
                ),
                detail=str(raw_attempt.get("detail", "")),
                completion_source=(
                    None
                    if raw_attempt.get("completion_source") is None
                    else str(raw_attempt["completion_source"])
                ),
            )
        )
    raw_waiver = payload.get("waiver")
    waiver = None
    if raw_waiver is not None:
        if not isinstance(raw_waiver, Mapping):
            raise ValueError("waiver must be an object or null")
        waiver = BaselineWaiver(
            reason=str(raw_waiver["reason"]),
            approved_by=str(raw_waiver["approved_by"]),
        )
    raw_labels = payload.get("labels", {})
    if not isinstance(raw_labels, Mapping):
        raise ValueError("labels must be an object")
    return Run(
        task=str(payload["task"]),
        attempts=tuple(attempts),
        budget_summary=str(payload["budget_summary"]),
        waiver=waiver,
        labels={str(key): str(value) for key, value in raw_labels.items()},
    )


@dataclass(frozen=True, slots=True)
class ResearchStage:
    """One named, ordered part of an experimental strategy."""

    ordinal: int
    name: str
    config_json: str
    config_fingerprint: str
    metadata_json: str = "{}"
    metadata_fingerprint: str = field(default_factory=lambda: fingerprint({}))

    @classmethod
    def create(
        cls,
        ordinal: int,
        name: str,
        config: Mapping[str, Any],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> ResearchStage:
        encoded = canonical_json(dict(config))
        metadata_data = dict(metadata or {})
        return cls(
            ordinal,
            name,
            encoded,
            fingerprint(dict(config)),
            canonical_json(metadata_data),
            fingerprint(metadata_data),
        )

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError("stage ordinal cannot be negative")
        if not self.name.strip():
            raise ValueError("stage needs a name")
        _require_fingerprint(self.config_fingerprint, "config_fingerprint")
        try:
            parsed = json.loads(self.config_json)
        except json.JSONDecodeError as error:
            raise ValueError("stage config_json must be JSON") from error
        if (
            not isinstance(parsed, dict)
            or fingerprint(parsed) != self.config_fingerprint
        ):
            raise ValueError("stage configuration does not match its fingerprint")
        _require_fingerprint(self.metadata_fingerprint, "metadata_fingerprint")
        metadata = json.loads(self.metadata_json)
        if (
            not isinstance(metadata, dict)
            or fingerprint(metadata) != self.metadata_fingerprint
        ):
            raise ValueError("stage metadata does not match its fingerprint")

    @property
    def config(self) -> dict[str, Any]:
        parsed = json.loads(self.config_json)
        assert isinstance(parsed, dict)
        return parsed

    @property
    def metadata(self) -> dict[str, Any]:
        """Artifact references and immutable handoff provenance for this stage."""
        parsed = json.loads(self.metadata_json)
        assert isinstance(parsed, dict)
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "name": self.name,
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "metadata": self.metadata,
            "metadata_fingerprint": self.metadata_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ResearchTrial:
    """One strategy execution, always retaining its own depth-zero control."""

    trial_id: str
    strategy: str
    run: Run
    stages: tuple[ResearchStage, ...]
    config_json: str
    config_fingerprint: str
    budget_fingerprint: str

    @classmethod
    def create(
        cls,
        trial_id: str,
        strategy: str,
        run: Run,
        *,
        stages: tuple[ResearchStage, ...] = (),
        config: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
    ) -> ResearchTrial:
        config_data = dict(config or {})
        budget_data = dict(budget or {"summary": run.budget_summary})
        return cls(
            trial_id=trial_id,
            strategy=strategy,
            run=run,
            stages=stages,
            config_json=canonical_json(config_data),
            config_fingerprint=fingerprint(config_data),
            budget_fingerprint=fingerprint(budget_data),
        )

    def __post_init__(self) -> None:
        if not self.trial_id.strip() or not self.strategy.strip():
            raise ValueError("trial needs an id and strategy")
        if self.run.baseline is None:
            raise ValueError("a research trial needs a depth-zero control Run")
        _require_fingerprint(self.config_fingerprint, "config_fingerprint")
        _require_fingerprint(self.budget_fingerprint, "budget_fingerprint")
        parsed = json.loads(self.config_json)
        if (
            not isinstance(parsed, dict)
            or fingerprint(parsed) != self.config_fingerprint
        ):
            raise ValueError("trial configuration does not match its fingerprint")
        if tuple(stage.ordinal for stage in self.stages) != tuple(
            range(len(self.stages))
        ):
            raise ValueError("trial stages must have contiguous ordinals from zero")

    @property
    def config(self) -> dict[str, Any]:
        parsed = json.loads(self.config_json)
        assert isinstance(parsed, dict)
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "strategy": self.strategy,
            "run": run_to_dict(self.run),
            "stages": [stage.to_dict() for stage in self.stages],
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "budget_fingerprint": self.budget_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ResearchRun:
    """A reproducible optional-strategy session and its stable control."""

    research_id: str
    control: Run
    trials: tuple[ResearchTrial, ...]
    config_json: str
    config_fingerprint: str
    budget_fingerprint: str
    schema_version: int = RESEARCH_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        research_id: str,
        control: Run,
        trials: tuple[ResearchTrial, ...],
        *,
        config: Mapping[str, Any] | None = None,
        budget: Mapping[str, Any] | None = None,
    ) -> ResearchRun:
        config_data = dict(config or {})
        budget_data = dict(budget or {"summary": control.budget_summary})
        return cls(
            research_id=research_id,
            control=control,
            trials=trials,
            config_json=canonical_json(config_data),
            config_fingerprint=fingerprint(config_data),
            budget_fingerprint=fingerprint(budget_data),
        )

    def __post_init__(self) -> None:
        if self.schema_version != RESEARCH_SCHEMA_VERSION:
            raise ValueError(f"unsupported research schema {self.schema_version}")
        if not self.research_id.strip():
            raise ValueError("research run needs an id")
        if self.control.baseline is None or len(self.control.attempts) != 1:
            raise ValueError("research control must be one real depth-zero Run")
        _require_fingerprint(self.config_fingerprint, "config_fingerprint")
        _require_fingerprint(self.budget_fingerprint, "budget_fingerprint")
        parsed = json.loads(self.config_json)
        if (
            not isinstance(parsed, dict)
            or fingerprint(parsed) != self.config_fingerprint
        ):
            raise ValueError("research configuration does not match its fingerprint")
        trial_ids = [trial.trial_id for trial in self.trials]
        if len(trial_ids) != len(set(trial_ids)):
            raise ValueError("research trial ids must be unique")

    @property
    def config(self) -> dict[str, Any]:
        parsed = json.loads(self.config_json)
        assert isinstance(parsed, dict)
        return parsed

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "research_id": self.research_id,
            "control": run_to_dict(self.control),
            "trials": [trial.to_dict() for trial in self.trials],
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "budget_fingerprint": self.budget_fingerprint,
        }


def research_run_from_dict(payload: Mapping[str, Any]) -> ResearchRun:
    """Rebuild a research record and validate every embedded fingerprint."""
    raw_trials = payload.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("research record needs trials")
    trials: list[ResearchTrial] = []
    for raw_trial in raw_trials:
        if not isinstance(raw_trial, Mapping):
            raise ValueError("trial must be an object")
        raw_stages = raw_trial.get("stages", [])
        if not isinstance(raw_stages, list):
            raise ValueError("trial stages must be a list")
        stages: list[ResearchStage] = []
        for stage in raw_stages:
            if not isinstance(stage, Mapping):
                raise ValueError("stage must be an object")
            created = ResearchStage.create(
                int(stage["ordinal"]),
                str(stage["name"]),
                dict(stage["config"]),
                metadata=dict(stage.get("metadata", {})),
            )
            if created.config_fingerprint != stage.get(
                "config_fingerprint"
            ) or created.metadata_fingerprint != stage.get("metadata_fingerprint"):
                raise ValueError("stage configuration fingerprint mismatch")
            stages.append(created)
        raw_config = raw_trial.get("config", {})
        if not isinstance(raw_config, Mapping):
            raise ValueError("trial configuration must be an object")
        trial = ResearchTrial(
            trial_id=str(raw_trial["trial_id"]),
            strategy=str(raw_trial["strategy"]),
            run=run_from_dict(raw_trial["run"]),
            stages=tuple(stages),
            config_json=canonical_json(dict(raw_config)),
            config_fingerprint=str(raw_trial["config_fingerprint"]),
            budget_fingerprint=str(raw_trial["budget_fingerprint"]),
        )
        trials.append(trial)
    raw_config = payload.get("config", {})
    if not isinstance(raw_config, Mapping):
        raise ValueError("research configuration must be an object")
    return ResearchRun(
        research_id=str(payload["research_id"]),
        control=run_from_dict(payload["control"]),
        trials=tuple(trials),
        config_json=canonical_json(dict(raw_config)),
        config_fingerprint=str(payload["config_fingerprint"]),
        budget_fingerprint=str(payload["budget_fingerprint"]),
        schema_version=int(payload.get("schema_version", RESEARCH_SCHEMA_VERSION)),
    )
