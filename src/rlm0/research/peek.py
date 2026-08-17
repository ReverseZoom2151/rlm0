"""Bounded, content-addressed maps for reusable long-context research.

A map is an index over a specific immutable context, not a substitute for the
context itself.  Its identity includes every input which can change its
contents, so changing the builder, model, prompt, schema, or limits produces a
cache miss.  Maps are JSON only, persisted with an atomic rename, and never
loaded through executable serialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

__all__ = [
    "ContextMap",
    "MapBuildError",
    "MapBuilder",
    "MapIdentity",
    "MapSection",
    "MapStore",
    "MapStoreError",
    "build_context_map",
]

_FORMAT_VERSION: Final = 1


class MapBuildError(ValueError):
    """A map cannot be built under the requested identity or limits."""


class MapStoreError(RuntimeError):
    """The on-disk map store cannot safely satisfy an operation."""


@dataclass(frozen=True, slots=True)
class MapIdentity:
    """Everything which makes a generated content map reproducible."""

    context_sha256: str
    context_chars: int
    builder_id: str
    model: str
    prompt_version: str
    schema_version: str
    max_entries: int
    summary_char_limit: int

    def __post_init__(self) -> None:
        if len(self.context_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.context_sha256
        ):
            raise ValueError("context_sha256 must be a lowercase SHA-256 digest")
        if self.context_chars < 0:
            raise ValueError("context_chars cannot be negative")
        for name in ("builder_id", "model", "prompt_version", "schema_version"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.max_entries < 1:
            raise ValueError("max_entries must be at least one")
        if self.summary_char_limit < 1:
            raise ValueError("summary_char_limit must be at least one")

    @classmethod
    def for_context(
        cls,
        context: str,
        *,
        builder_id: str,
        model: str,
        prompt_version: str,
        schema_version: str,
        max_entries: int,
        summary_char_limit: int,
    ) -> MapIdentity:
        """Make an identity bound to exactly this Unicode string."""

        if not isinstance(context, str):
            raise TypeError("context must be a string")
        return cls(
            context_sha256=_context_hash(context),
            context_chars=len(context),
            builder_id=builder_id,
            model=model,
            prompt_version=prompt_version,
            schema_version=schema_version,
            max_entries=max_entries,
            summary_char_limit=summary_char_limit,
        )

    @property
    def key(self) -> str:
        """A filename-safe hash of all cache-relevant identity fields."""

        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class MapSection:
    """One source span and its bounded map text."""

    start: int
    end: int
    summary: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("a map section must cover a nonempty forward span")
        if not self.summary:
            raise ValueError("a map section summary must not be empty")


@dataclass(frozen=True, slots=True)
class ContextMap:
    """A map with entries ordered by their source span."""

    identity: MapIdentity
    sections: tuple[MapSection, ...]
    format_version: int = _FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.format_version != _FORMAT_VERSION:
            raise ValueError(f"unsupported map format {self.format_version}")
        if not self.sections:
            raise ValueError("a context map needs at least one section")
        if len(self.sections) > self.identity.max_entries:
            raise ValueError("map has more sections than its identity permits")
        previous_end = 0
        for section in self.sections:
            if section.start != previous_end:
                raise ValueError("map sections must be contiguous and ordered")
            if section.end > self.identity.context_chars:
                raise ValueError("map section exceeds the source context")
            if len(section.summary) > self.identity.summary_char_limit:
                raise ValueError("map section summary exceeds its configured limit")
            previous_end = section.end
        if previous_end != self.identity.context_chars:
            raise ValueError("map sections must cover the full source context")


MapBuilder = Callable[[str, int, int], str]


def build_context_map(
    context: str,
    identity: MapIdentity,
    builder: MapBuilder,
) -> ContextMap:
    """Build a bounded, complete map with a caller-supplied deterministic builder.

    The builder receives a source span, zero-based section index, and total
    section count.  Its output is trimmed, never coerced, so broken builders
    cannot quietly turn an arbitrary object into a map.
    """

    if not isinstance(context, str):
        raise TypeError("context must be a string")
    if (
        _context_hash(context) != identity.context_sha256
        or len(context) != identity.context_chars
    ):
        raise MapBuildError("context does not match the map identity")
    if not context:
        raise MapBuildError("cannot build a content map for empty context")

    count = min(identity.max_entries, len(context))
    bounds = _balanced_bounds(len(context), count)
    sections: list[MapSection] = []
    for index, (start, end) in enumerate(bounds):
        try:
            result = builder(context[start:end], index, count)
        except Exception as exc:
            raise MapBuildError(f"map builder failed for section {index}") from exc
        if not isinstance(result, str):
            raise MapBuildError(f"map builder returned non-text for section {index}")
        summary = result.strip()
        if not summary:
            raise MapBuildError(f"map builder returned empty text for section {index}")
        sections.append(
            MapSection(
                start=start,
                end=end,
                summary=summary[: identity.summary_char_limit],
            )
        )
    return ContextMap(identity=identity, sections=tuple(sections))


class MapStore:
    """A private, JSON-only, content-addressed store for context maps.

    Cache files are named only from a SHA-256 identity, avoiding caller-derived
    paths.  Writes first create a mode-0600 temporary file in the target
    directory, fsync it, then atomically replace the destination.
    """

    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not root.is_dir():
            raise MapStoreError(f"map-store root is not a directory: {root}")
        self._root = root.resolve()
        self._set_private_permissions(self._root)

    @property
    def root(self) -> Path:
        """The resolved, private store directory."""

        return self._root

    def load(self, identity: MapIdentity) -> ContextMap | None:
        """Load a map if it exists; identity changes naturally invalidate it."""

        path = self._path(identity)
        if not path.exists():
            return None
        self._assert_regular_private_file(path)
        try:
            raw = path.read_text(encoding="utf-8")
            decoded = json.loads(raw)
            context_map = _map_from_json(decoded)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise MapStoreError(
                f"cannot safely load cached map {identity.key}"
            ) from exc
        if context_map.identity != identity:
            raise MapStoreError(
                "cached map identity does not match its content-addressed key"
            )
        return context_map

    def save(self, context_map: ContextMap) -> Path:
        """Atomically persist a map and return its content-addressed path."""

        path = self._path(context_map.identity)
        self._set_private_permissions(self._root)
        payload = json.dumps(
            _map_to_json(context_map),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{context_map.identity.key}.", suffix=".tmp", dir=self._root
            )
            temporary = Path(temporary_name)
            try:
                # Windows does not expose ``fchmod``. Permissions there are
                # governed by the caller's ACL and this store never treats an
                # ACL as proof of privacy; the POSIX mode hardening remains
                # useful where it is available.
                fchmod = getattr(os, "fchmod", None)
                if os.name == "posix" and callable(fchmod):
                    fchmod(descriptor, 0o600)
                with os.fdopen(descriptor, "wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                self._set_private_permissions(path)
                _fsync_directory(self._root)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise MapStoreError(
                f"cannot persist cached map {context_map.identity.key}"
            ) from exc
        return path

    def get_or_build(
        self, context: str, identity: MapIdentity, builder: MapBuilder
    ) -> ContextMap:
        """Return a matching cached map or build and atomically store one."""

        cached = self.load(identity)
        if cached is not None:
            return cached
        context_map = build_context_map(context, identity, builder)
        self.save(context_map)
        return context_map

    def invalidate(self, identity: MapIdentity) -> bool:
        """Remove only this exact identity's cache file, if present."""

        path = self._path(identity)
        if not path.exists():
            return False
        self._assert_regular_private_file(path)
        try:
            path.unlink()
            _fsync_directory(self._root)
        except OSError as exc:
            raise MapStoreError(f"cannot invalidate cached map {identity.key}") from exc
        return True

    def _path(self, identity: MapIdentity) -> Path:
        return self._root / f"{identity.key}.json"

    @staticmethod
    def _set_private_permissions(path: Path) -> None:
        if os.name != "posix":
            return
        try:
            path.chmod(0o700 if path.is_dir() else 0o600)
        except OSError as exc:
            raise MapStoreError(f"cannot secure map-store path {path}") from exc

    @staticmethod
    def _assert_regular_private_file(path: Path) -> None:
        try:
            file_stat = path.lstat()
        except OSError as exc:
            raise MapStoreError(f"cannot inspect cached map {path.name}") from exc
        if not stat.S_ISREG(file_stat.st_mode) or stat.S_ISLNK(file_stat.st_mode):
            raise MapStoreError("cached map path is not a regular file")
        if os.name == "posix" and file_stat.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise MapStoreError("cached map is accessible outside its owner")


def _balanced_bounds(length: int, count: int) -> tuple[tuple[int, int], ...]:
    """Partition a source into contiguous, nearly equal, nonempty spans."""

    return tuple(
        ((length * index) // count, (length * (index + 1)) // count)
        for index in range(count)
    )


def _context_hash(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def _map_to_json(context_map: ContextMap) -> dict[str, object]:
    return {
        "format_version": context_map.format_version,
        "identity": asdict(context_map.identity),
        "sections": [asdict(section) for section in context_map.sections],
    }


def _map_from_json(value: object) -> ContextMap:
    if not isinstance(value, dict):
        raise ValueError("map file must contain an object")
    if set(value) != {"format_version", "identity", "sections"}:
        raise ValueError("map file has an unsupported schema")
    raw_identity = value["identity"]
    raw_sections = value["sections"]
    raw_version = value["format_version"]
    if not isinstance(raw_identity, dict) or not isinstance(raw_sections, list):
        raise ValueError("map identity or sections have an invalid shape")
    if not isinstance(raw_version, int):
        raise ValueError("map format version must be an integer")
    identity = MapIdentity(**raw_identity)
    sections = tuple(MapSection(**raw_section) for raw_section in raw_sections)
    return ContextMap(identity=identity, sections=sections, format_version=raw_version)


def _fsync_directory(directory: Path) -> None:
    """Persist a rename on POSIX; some Windows filesystems lack directory FDs."""

    if os.name != "posix":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError as exc:
        raise MapStoreError(f"cannot fsync map-store directory {directory}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise MapStoreError(f"cannot fsync map-store directory {directory}") from exc
    finally:
        os.close(descriptor)
