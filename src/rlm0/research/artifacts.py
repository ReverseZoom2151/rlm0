"""Bounded, content-addressed artifacts for fresh-root research handoffs."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

__all__ = ["ArtifactLimitError", "ArtifactRef", "ArtifactStore"]


class ArtifactLimitError(ValueError):
    """An artifact cannot be stored without exceeding its declared bounds."""


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    """A content address, not a host path handed to an untrusted guest."""

    digest: str
    size_bytes: int
    media_type: str = "application/octet-stream"

    def __post_init__(self) -> None:
        valid = len(self.digest) == 64 and all(
            char in "0123456789abcdef" for char in self.digest
        )
        if not valid:
            raise ValueError("artifact digest must be lowercase SHA-256")
        if self.size_bytes < 0 or not self.media_type.strip():
            raise ValueError("artifact needs a nonnegative size and media type")


class ArtifactStore:
    """Atomic local object store with explicit per-object and total bounds."""

    def __init__(
        self,
        root: Path,
        *,
        max_total_bytes: int,
        max_artifact_bytes: int,
    ) -> None:
        if max_total_bytes < 0 or max_artifact_bytes < 0:
            raise ValueError("artifact limits cannot be negative")
        if max_artifact_bytes > max_total_bytes:
            raise ValueError("one artifact cannot exceed the whole store")
        self.root = root
        self.objects = root / "objects"
        self.max_total_bytes = max_total_bytes
        self.max_artifact_bytes = max_artifact_bytes
        self._lock = RLock()
        self.objects.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        return self.objects / digest[:2] / digest[2:]

    def _total_bytes(self) -> int:
        return sum(
            path.stat().st_size for path in self.objects.rglob("*") if path.is_file()
        )

    def put_bytes(
        self, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> ArtifactRef:
        """Store data once. Existing identical data does not spend capacity twice."""
        if len(data) > self.max_artifact_bytes:
            raise ArtifactLimitError("artifact exceeds max_artifact_bytes")
        digest = hashlib.sha256(data).hexdigest()
        reference = ArtifactRef(digest, len(data), media_type)
        destination = self._path(digest)
        with self._lock:
            if destination.exists():
                if self.read_bytes(reference) != data:
                    raise RuntimeError("artifact content address collision")
                return reference
            if self._total_bytes() + len(data) > self.max_total_bytes:
                raise ArtifactLimitError("artifact store exceeds max_total_bytes")
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".artifact-", dir=destination.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return reference

    def put_text(
        self, text: str, *, media_type: str = "text/plain; charset=utf-8"
    ) -> ArtifactRef:
        return self.put_bytes(text.encode("utf-8"), media_type=media_type)

    def read_bytes(self, reference: ArtifactRef) -> bytes:
        """Load and hash-check a referenced object before handing it onward."""
        path = self._path(reference.digest)
        try:
            data = path.read_bytes()
        except FileNotFoundError as error:
            raise KeyError(f"artifact {reference.digest} does not exist") from error
        if len(data) != reference.size_bytes:
            raise ValueError("artifact size does not match reference")
        if hashlib.sha256(data).hexdigest() != reference.digest:
            raise ValueError("artifact content does not match reference")
        return data

    def contains(self, reference: ArtifactRef) -> bool:
        try:
            self.read_bytes(reference)
        except (KeyError, ValueError):
            return False
        return True
