"""Where a public benchmark's data lives, and what to say when it is missing.

Nothing here downloads anything, and nothing here is vendored. Both choices
are deliberate and they point the same way: a test suite that reaches the
network is a test suite that fails for reasons unrelated to the code, and a
benchmark checked into a repository is a benchmark that quietly drifts from
the version the leaderboard is computed on.

So an adapter states what it needs as data rather than as prose: which
repository, which revision, which split, which files. When the files are not
on disk the failure is a single message naming the exact command that would
fix it. That message is the entire user interface of this layer, and it is
the part most benchmark integrations get wrong: several implementations
surveyed for this project shipped benchmark names with no data behind them at
all, so the only symptom of a missing dataset was a score of zero.

Every file that is actually read is hashed, and the digest of those hashes is
carried into the manifest. The revision cannot be verified offline, so it is
recorded rather than asserted; the file digest is what actually ties a
published number to the bytes that produced it, and a caller who knows what
the digest should be can demand it.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_ROOT",
    "ENV_ROOT",
    "BenchmarkDataError",
    "DatasetRequirement",
    "DatasetUnavailableError",
    "LoadedFiles",
    "load_files",
    "resolve_root",
]

ENV_ROOT = "RLM0_BENCHMARK_DATA"
"""Environment variable naming the directory that holds downloaded corpora."""

DEFAULT_ROOT = Path.home() / ".cache" / "rlm0" / "benchmarks"


class BenchmarkDataError(RuntimeError):
    """The files are present but do not support the questions asked of them.

    Separate from absence because the two need different reactions. Absence is
    a download away. This one means a row carried no answer, or a needle that
    the benchmark says is in the context is not in the context, and continuing
    past it would produce a grader that scores nothing and reports a number
    anyway.
    """


@dataclass(frozen=True, slots=True)
class DatasetRequirement:
    """Exactly what a benchmark needs on disk, and how to put it there.

    `download` is a list of shell commands and not a description of them. A
    user who has just been told a dataset is missing wants a line to paste,
    and an instruction they have to translate is one they will translate
    wrongly.
    """

    benchmark: str
    source: str
    """Where the data comes from, as an addressable identifier."""

    revision: str
    """A pinned commit. Recorded in the manifest so a number is traceable."""

    config: str
    split: str
    patterns: tuple[str, ...]
    """Glob patterns, relative to the root, of the files that must exist."""

    download: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def message(self, root: Path, detail: str) -> str:
        """The whole error text, written for somebody who has nothing yet."""
        lines = [
            f"{self.benchmark}: {detail}",
            f"  expected under: {root}",
            f"  source:         {self.source}",
            f"  revision:       {self.revision}",
            f"  config/split:   {self.config}/{self.split}",
            f"  files:          {', '.join(self.patterns)}",
            "",
            "  to obtain it, run:",
        ]
        lines.extend(f"    {command}" for command in self.download)
        if self.notes:
            lines.append("")
            lines.extend(f"  note: {note}" for note in self.notes)
        lines.append("")
        lines.append(
            f"  or set {ENV_ROOT} to a directory holding a "
            f"{self.benchmark}/ subdirectory with those files."
        )
        return "\n".join(lines)


class DatasetUnavailableError(RuntimeError):
    """The dataset is not on disk, and the message says how to get it.

    Raised at load time rather than at score time. An adapter that discovers
    its own emptiness only once a suite is running has already let somebody
    start paying for a run that cannot produce a result.
    """

    def __init__(
        self, requirement: DatasetRequirement, root: Path, detail: str
    ) -> None:
        super().__init__(requirement.message(root, detail))
        self.requirement = requirement
        self.root = root


def resolve_root(benchmark: str, override: Path | None = None) -> Path:
    """Decide where to look, preferring what the caller said explicitly."""
    if override is not None:
        return override
    env = os.environ.get(ENV_ROOT)
    base = Path(env) if env else DEFAULT_ROOT
    return base / benchmark


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise BenchmarkDataError(
                    f"{path} line {number} is not an object, so it carries no "
                    "named fields to read"
                )
            rows.append(payload)
    return rows


def _read_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("data", payload.get("rows", []))
    if not isinstance(payload, list):
        raise BenchmarkDataError(f"{path} does not hold a list of records")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise BenchmarkDataError(f"{path} holds a non-object record")
        rows.append(item)
    return rows


def _read_parquet(path: Path, requirement: DatasetRequirement) -> list[dict[str, Any]]:
    """Read parquet through pyarrow, which is not a dependency of this project.

    Imported by name rather than at module scope so that the absence of an
    optional dependency is reported as part of the same actionable message as
    the absence of the data, instead of as an ImportError from an unrelated
    line at import time.
    """
    try:
        parquet = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise DatasetUnavailableError(
            requirement,
            path.parent,
            f"{path.name} is parquet and pyarrow is not installed",
        ) from exc
    table = parquet.read_table(str(path))
    rows: list[dict[str, Any]] = []
    for row in table.to_pylist():
        if not isinstance(row, dict):  # pragma: no cover - arrow always maps
            raise BenchmarkDataError(f"{path} produced a non-object row")
        rows.append(row)
    return rows


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class LoadedFiles:
    """The rows that were read, and the identity of the bytes they came from.

    `content_hash` covers the relative paths as well as their contents, so
    reading the same records out of a differently split set of shards is not
    mistaken for reading the same file.
    """

    root: Path
    paths: tuple[Path, ...]
    rows: tuple[dict[str, Any], ...]
    content_hash: str

    @property
    def relative_paths(self) -> tuple[str, ...]:
        return tuple(p.relative_to(self.root).as_posix() for p in self.paths)


def load_files(
    requirement: DatasetRequirement,
    root: Path,
    *,
    expected_hash: str | None = None,
    limit: int | None = None,
) -> LoadedFiles:
    """Find, read and hash the files a requirement names.

    `expected_hash` is optional because this project cannot know the digest of
    a dataset it refuses to vendor. When a caller does know it, supplying it
    turns a silent data swap into a refusal, which is the only way a published
    figure stays attached to its inputs.
    """
    if not root.exists():
        raise DatasetUnavailableError(requirement, root, "directory does not exist")

    found: list[Path] = []
    for pattern in requirement.patterns:
        found.extend(sorted(root.glob(pattern)))
    paths = tuple(dict.fromkeys(found))
    if not paths:
        raise DatasetUnavailableError(
            requirement, root, "no file matched the expected patterns"
        )

    rows: list[dict[str, Any]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            rows.extend(_read_jsonl(path))
        elif suffix == ".json":
            rows.extend(_read_json(path))
        elif suffix == ".parquet":
            rows.extend(_read_parquet(path, requirement))
        else:
            raise BenchmarkDataError(
                f"{path} has an unsupported extension; this loader reads "
                "jsonl, json and parquet"
            )
    if not rows:
        raise BenchmarkDataError(
            f"{requirement.benchmark}: the files under {root} hold no records, "
            "so any score computed over them would be an average over nothing"
        )

    outer = hashlib.sha256()
    for path in paths:
        outer.update(path.relative_to(root).as_posix().encode("utf-8"))
        outer.update(b"\0")
        outer.update(_digest(path).encode("ascii"))
        outer.update(b"\n")
    content_hash = outer.hexdigest()
    if expected_hash is not None and content_hash != expected_hash:
        raise BenchmarkDataError(
            f"{requirement.benchmark}: the data under {root} hashes to "
            f"{content_hash[:16]} but {expected_hash[:16]} was required; these "
            "are different bytes and a result from one does not transfer"
        )

    kept: Sequence[dict[str, Any]] = rows if limit is None else rows[:limit]
    return LoadedFiles(
        root=root,
        paths=paths,
        rows=tuple(kept),
        content_hash=content_hash,
    )
