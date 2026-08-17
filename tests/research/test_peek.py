from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import cast

import pytest

from rlm0.research.peek import (
    MapBuilder,
    MapBuildError,
    MapIdentity,
    MapStore,
    MapStoreError,
    build_context_map,
)


def make_identity(context: str, **changes: object) -> MapIdentity:
    options: dict[str, object] = {
        "builder_id": "peek-v1",
        "model": "fake-map-builder",
        "prompt_version": "map-prompt-v1",
        "schema_version": "peek-schema-v1",
        "max_entries": 3,
        "summary_char_limit": 12,
    }
    options.update(changes)
    return MapIdentity.for_context(context, **options)  # type: ignore[arg-type]


def labelled_builder(span: str, index: int, total: int) -> str:
    return f"{index + 1}/{total}:{span}"


def test_identity_changes_for_every_cache_relevant_input() -> None:
    context = "abcdefgh"
    base = make_identity(context)
    variants = [
        make_identity(context + "!"),
        make_identity(context, builder_id="peek-v2"),
        make_identity(context, model="other-model"),
        make_identity(context, prompt_version="map-prompt-v2"),
        make_identity(context, schema_version="peek-schema-v2"),
        make_identity(context, max_entries=2),
        make_identity(context, summary_char_limit=11),
    ]

    assert len({base.key, *(variant.key for variant in variants)}) == 8


def test_build_partitions_the_full_context_and_bounds_each_summary() -> None:
    context = "abcdefghij"
    result = build_context_map(context, make_identity(context), labelled_builder)

    assert [(section.start, section.end) for section in result.sections] == [
        (0, 3),
        (3, 6),
        (6, 10),
    ]
    assert [section.summary for section in result.sections] == [
        "1/3:abc",
        "2/3:def",
        "3/3:ghij",
    ]
    assert all(
        len(section.summary) <= result.identity.summary_char_limit
        for section in result.sections
    )


def test_build_rejects_mismatched_or_empty_context_and_bad_builder_output() -> None:
    identity = make_identity("corpus")
    with pytest.raises(MapBuildError, match="does not match"):
        build_context_map("other", identity, labelled_builder)
    with pytest.raises(MapBuildError, match="empty"):
        build_context_map("", make_identity(""), labelled_builder)
    with pytest.raises(MapBuildError, match="non-text"):
        build_context_map("corpus", identity, cast(MapBuilder, lambda *_: 42))
    with pytest.raises(MapBuildError, match="empty text"):
        build_context_map("corpus", identity, lambda *_: "  ")


def test_store_round_trip_is_content_addressed_and_get_or_build_reuses(
    tmp_path: Path,
) -> None:
    context = "a source long enough to split"
    identity = make_identity(context, max_entries=2)
    store = MapStore(tmp_path / "maps")
    calls = 0

    def counted_builder(span: str, index: int, total: int) -> str:
        nonlocal calls
        calls += 1
        return labelled_builder(span, index, total)

    first = store.get_or_build(context, identity, counted_builder)
    second = store.get_or_build(context, identity, counted_builder)
    path = store.save(first)

    assert first == second
    assert calls == len(first.sections)
    assert path.name == f"{identity.key}.json"
    assert store.load(identity) == first
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700


def test_identity_change_is_a_cache_miss_and_invalidation_is_exact(
    tmp_path: Path,
) -> None:
    context = "same source"
    original = make_identity(context, prompt_version="one")
    changed = make_identity(context, prompt_version="two")
    store = MapStore(tmp_path / "maps")
    first = build_context_map(context, original, labelled_builder)
    second = build_context_map(context, changed, labelled_builder)
    store.save(first)
    store.save(second)

    assert store.invalidate(original)
    assert store.load(original) is None
    assert store.load(changed) == second
    assert not store.invalidate(original)


def test_corrupt_or_symlinked_cache_is_not_read_as_data(tmp_path: Path) -> None:
    context = "corpus"
    identity = make_identity(context)
    store = MapStore(tmp_path / "maps")
    path = store.root / f"{identity.key}.json"
    path.write_text("not-json", encoding="utf-8")
    if os.name == "posix":
        path.chmod(0o600)

    with pytest.raises(MapStoreError, match="cannot safely load"):
        store.load(identity)

    if hasattr(os, "symlink"):
        path.unlink()
        target = tmp_path / "outside.json"
        target.write_text("{}", encoding="utf-8")
        path.symlink_to(target)
        with pytest.raises(MapStoreError, match="regular file"):
            store.load(identity)


def test_store_root_must_be_a_directory(tmp_path: Path) -> None:
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("x", encoding="utf-8")

    with pytest.raises((FileExistsError, MapStoreError)):
        MapStore(file_root)
