from __future__ import annotations

from pathlib import Path

import pytest

from rlm0.research.artifacts import ArtifactLimitError, ArtifactRef, ArtifactStore


def test_store_is_content_addressed_and_deduplicated(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_total_bytes=10, max_artifact_bytes=10)
    first = store.put_text("hello")
    second = store.put_text("hello")

    assert first == second
    assert store.read_bytes(first) == b"hello"
    total = sum(
        path.stat().st_size
        for path in (tmp_path / "objects").rglob("*")
        if path.is_file()
    )
    assert total == 5


def test_store_enforces_per_artifact_and_total_limits(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_total_bytes=5, max_artifact_bytes=4)
    with pytest.raises(ArtifactLimitError, match="max_artifact"):
        store.put_bytes(b"12345")
    store.put_bytes(b"1234")
    with pytest.raises(ArtifactLimitError, match="max_total"):
        store.put_bytes(b"xy")


def test_store_detects_tampered_object(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_total_bytes=10, max_artifact_bytes=10)
    reference = store.put_bytes(b"safe")
    object_path = tmp_path / "objects" / reference.digest[:2] / reference.digest[2:]
    object_path.write_bytes(b"evil")

    with pytest.raises(ValueError, match=r"size|content"):
        store.read_bytes(reference)
    assert not store.contains(reference)


def test_unknown_reference_is_not_present(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path, max_total_bytes=10, max_artifact_bytes=10)
    reference = ArtifactRef("0" * 64, 0)
    assert not store.contains(reference)
