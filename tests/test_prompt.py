"""Tests for the prompt, most of which are about the control.

The assertion this file exists for is that the depth-zero prompt and the
recursive prompt differ in exactly one place. Everything the run layer claims
to measure depends on it, and it is the kind of property that decays the first
time somebody improves one variant's wording and not the other's.
"""

from __future__ import annotations

import pytest

from rlm0.prompt import (
    MIN_SUB_CALL_CHARS,
    SUB_CALL_CHAR_BUDGET,
    SUB_CALL_SECTIONS,
    ContextShape,
    build_system_prompt,
    build_turn_prompt,
    system_prompt_sections,
)

SHAPE = ContextShape.of(["a" * 100, "b" * 250, "c" * 3])


def test_shape_of_a_string() -> None:
    shape = ContextShape.of("x" * 1234)
    assert shape.kind == "str"
    assert shape.total_chars == 1234
    assert shape.n_chunks == 0
    assert "1234" in shape.describe()


def test_shape_of_a_chunk_list() -> None:
    assert SHAPE.kind == "list[str]"
    assert SHAPE.total_chars == 353
    assert SHAPE.chunk_lengths == (100, 250, 3)
    described = SHAPE.describe()
    assert "[100, 250, 3]" in described
    assert "353" in described


def test_long_chunk_listings_are_abbreviated() -> None:
    shape = ContextShape.of(["z" * 10] * 500)
    described = shape.describe()
    assert "and 400 more" in described
    assert len(described) < 1500


def test_shape_refuses_lengths_that_do_not_add_up() -> None:
    with pytest.raises(ValueError, match="sum to total_chars"):
        ContextShape(total_chars=10, chunk_lengths=(3, 3))


def test_shape_never_carries_content() -> None:
    # The dataclass has no field that could hold a preview, which is the
    # guarantee, so this asserts the field set rather than a value.
    fields = set(ContextShape.__dataclass_fields__)
    assert fields == {"total_chars", "chunk_lengths", "kind"}


def test_the_two_variants_are_identical_outside_the_sub_call_sections() -> None:
    deep = system_prompt_sections(SHAPE, sub_calls=True)
    zero = system_prompt_sections(SHAPE, sub_calls=False)
    assert [s.name for s in deep] == [s.name for s in zero]
    differing = set()
    for a, b in zip(deep, zero, strict=True):
        if a.body != b.body:
            differing.add(a.name)
    assert differing == SUB_CALL_SECTIONS


def test_the_shared_text_is_byte_identical() -> None:
    def shared(sub_calls: bool) -> str:
        return "\n\n".join(
            s.body
            for s in system_prompt_sections(SHAPE, sub_calls=sub_calls)
            if s.name not in SUB_CALL_SECTIONS
        )

    assert shared(True) == shared(False)
    # The sections that carry tone and strategy framing are among the shared
    # ones, so an identical remainder is not a vacuous claim.
    shared_names = {
        s.name
        for s in system_prompt_sections(SHAPE, sub_calls=True)
        if s.name not in SUB_CALL_SECTIONS
    }
    assert {"role", "protocol", "truncation", "final", "closing"} <= shared_names


def test_depth_zero_never_mentions_the_sub_call_machinery() -> None:
    zero = build_system_prompt(SHAPE, sub_calls=False)
    assert "llm_query" not in zero
    deep = build_system_prompt(SHAPE, sub_calls=True)
    assert "llm_query" in deep


def test_both_variants_carry_the_same_three_strategies() -> None:
    for sub_calls in (True, False):
        text = build_system_prompt(SHAPE, sub_calls=sub_calls)
        assert "Iterative buffering" in text
        assert "Split and aggregate" in text
        assert "Split on structure" in text


def test_sizing_guidance_is_stated_in_characters() -> None:
    deep = build_system_prompt(SHAPE, sub_calls=True)
    assert str(SUB_CALL_CHAR_BUDGET) in deep
    assert str(MIN_SUB_CALL_CHARS) in deep


def test_a_snippet_sized_sub_call_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="snippet classifier"):
        build_system_prompt(SHAPE, sub_calls=True, sub_call_chars=500)


def test_prompt_states_the_shape_and_nothing_of_the_content() -> None:
    shape = ContextShape.of(["secret payload".ljust(1237, "x"), "another secret"])
    text = build_system_prompt(shape, sub_calls=True)
    assert "secret" not in text
    assert str(shape.total_chars) in text
    assert "1237" in text


def test_final_answer_mechanisms_are_both_documented() -> None:
    text = build_system_prompt(SHAPE, sub_calls=False)
    assert "FINAL(" in text
    assert "FINAL_VAR(" in text
    assert "output limit" in text


def test_first_turn_carries_the_safeguard() -> None:
    first = build_turn_prompt("who won", iteration=0)
    assert "have not run any code yet" in first
    assert "who won" in first
    later = build_turn_prompt("who won", iteration=3)
    assert "have not run any code yet" not in later
    assert "who won" in later


def test_wrap_up_turn_asks_for_the_answer() -> None:
    text = build_turn_prompt("who won", iteration=5, wrap_up=True)
    assert "last turn" in text
    assert "FINAL" in text


def test_turn_prompt_does_not_depend_on_depth() -> None:
    # The turn prompt takes no sub_calls argument at all, which is the
    # enforcement; this pins the wording that would otherwise reintroduce the
    # difference once per turn.
    for iteration in (0, 1, 7):
        text = build_turn_prompt("q", iteration=iteration)
        assert "llm_query" not in text
        assert "sub-model" not in text


def test_negative_iteration_is_refused() -> None:
    with pytest.raises(ValueError, match="iteration"):
        build_turn_prompt("q", iteration=-1)


def test_no_em_or_en_dashes_anywhere_in_the_prompt() -> None:
    texts = [
        build_system_prompt(SHAPE, sub_calls=True),
        build_system_prompt(SHAPE, sub_calls=False),
        build_turn_prompt('q', iteration=0),
        build_turn_prompt('q', iteration=1),
        build_turn_prompt('q', iteration=1, wrap_up=True),
    ]
    banned = (chr(0x2014), chr(0x2013), chr(0x2212))
    for text in texts:
        for dash in banned:
            assert dash not in text
