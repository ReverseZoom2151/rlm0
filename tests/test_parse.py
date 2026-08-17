"""Tests for the parser, mostly adversarial.

The unit tests below are a list of things that have gone wrong in real
implementations. The property tests are there because the failure mode is
always an answer that the parser silently reshaped, and the only reliable way
to find those is to generate answers nobody would have thought to write.
"""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from rlm0.parse import (
    CompletionSource,
    FinalKind,
    Rejection,
    extract_code_blocks,
    find_final_answer,
    parse_turn,
)


def _balanced(text: str) -> bool:
    depth = 0
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


# --------------------------------------------------------------------------
# code blocks


def test_extracts_a_repl_block() -> None:
    text = "Here goes.\n\n```repl\nprint(len(context))\n```\n\nThat should do it."
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].code == "print(len(context))\n"
    assert blocks[0].language == "repl"
    assert blocks[0].executable


def test_extracts_several_blocks_in_order() -> None:
    text = "```repl\na = 1\n```\ntalk\n```python\nb = 2\n```"
    blocks = extract_code_blocks(text)
    assert [b.code.strip() for b in blocks] == ["a = 1", "b = 2"]


def test_non_code_fences_are_seen_but_not_run() -> None:
    text = "```json\n{}\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert not blocks[0].executable
    assert parse_turn(text).executable_blocks == ()


def test_nested_fences_belong_to_the_outer_block() -> None:
    text = "````repl\nprint('''```repl\nnot code\n```''')\n````"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert "not code" in blocks[0].code


def test_unterminated_fence_is_kept_and_flagged() -> None:
    text = "```repl\nfor chunk in context:\n    print(chunk[:10])"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 1
    assert not blocks[0].terminated
    assert blocks[0].code.startswith("for chunk")


def test_indented_fence_still_opens_a_block() -> None:
    text = "  ```repl\n  x = 1\n  ```"
    assert len(extract_code_blocks(text)) == 1


def test_prose_without_any_fence() -> None:
    parsed = parse_turn("I think the answer is probably in chunk 6.")
    assert parsed.code_blocks == ()
    assert parsed.final is None
    assert parsed.is_empty


# --------------------------------------------------------------------------
# final answers


def test_plain_final_answer() -> None:
    final, rejection = find_final_answer("All done. FINAL(the answer is 42)")
    assert rejection is None
    assert final is not None
    assert final.kind is FinalKind.LITERAL
    assert final.value == "the answer is 42"
    assert final.source is CompletionSource.RECOVERED_FINAL


def test_versioned_completion_protocol_carries_evidence() -> None:
    text = (
        'RLM0_FINAL_V1({"protocol_version": 1, "status": "answered", '
        '"answer": "42", "evidence": ["sum at row 7"], '
        '"answer_artifact": null})'
    )
    final, rejection = find_final_answer(text)
    assert rejection is None
    assert final is not None
    assert final.source is CompletionSource.V1
    assert final.protocol_version == 1
    assert final.evidence == ("sum at row 7",)


def test_versioned_completion_rejects_missing_or_extra_fields() -> None:
    final, rejection = find_final_answer(
        'RLM0_FINAL_V1({"protocol_version": 1, "status": "answered", '
        '"answer": "42", "evidence": []})'
    )
    assert final is None
    assert rejection is Rejection.MALFORMED_PROTOCOL


def test_answer_containing_parentheses_survives() -> None:
    text = "FINAL(Agoo (La Union) held the 13th festival (in 2017))"
    final, _ = find_final_answer(text)
    assert final is not None
    assert final.value == "Agoo (La Union) held the 13th festival (in 2017)"


def test_answer_with_an_unbalanced_close_takes_the_longer_reading() -> None:
    # A non-greedy or a naive balanced match both stop at the smiley.
    text = "FINAL(the winner was Maria :) and the runner up was Ana)"
    final, _ = find_final_answer(text)
    assert final is not None
    assert final.value == "the winner was Maria :) and the runner up was Ana"


def test_unterminated_directive_takes_the_tail_and_is_flagged() -> None:
    final, _ = find_final_answer("FINAL(the answer is 42 and it was cut off")
    assert final is not None
    assert not final.terminated
    assert final.value == "the answer is 42 and it was cut off"


def test_directive_inside_a_code_block_does_not_fire() -> None:
    text = "```repl\n# remember to write FINAL(answer) when done\nx = 1\n```"
    final, rejection = find_final_answer(text)
    assert final is None
    assert rejection is None


def test_directive_inside_a_string_literal_does_not_fire() -> None:
    text = (
        "```repl\n"
        'note = "when finished, emit FINAL(the result) to stop"\n'
        "print(note)\n"
        "```"
    )
    assert find_final_answer(text)[0] is None


def test_directive_in_an_inline_code_span_does_not_fire() -> None:
    text = "I will use `FINAL(x)` once the buffers are built."
    assert find_final_answer(text)[0] is None


def test_a_word_ending_in_final_does_not_fire() -> None:
    for text in ("MY_FINAL(x)", "semifinal(x)", "the_FINAL(x)"):
        assert find_final_answer(text)[0] is None
    assert find_final_answer("FINALIZE(x)")[0] is None


def test_the_last_directive_wins() -> None:
    text = "I could say FINAL(a guess) but instead FINAL(the real answer)"
    final, _ = find_final_answer(text)
    assert final is not None
    assert final.value == "the real answer"


def test_final_var_returns_a_name() -> None:
    final, _ = find_final_answer("Done. FINAL_VAR(summary_buffer)")
    assert final is not None
    assert final.kind is FinalKind.VARIABLE
    assert final.value == "summary_buffer"


def test_final_var_with_something_that_is_not_a_name_is_rejected() -> None:
    for bad in ("FINAL_VAR(context[0])", "FINAL_VAR(a b)", "FINAL_VAR(class)"):
        final, rejection = find_final_answer(bad)
        assert final is None
        assert rejection is Rejection.MALFORMED_VARIABLE


def test_no_sentinel_variable_is_ever_harvested() -> None:
    # Nothing here can turn a conventionally named variable into an answer.
    parsed = parse_turn("```repl\nfinal_answer = 'nope'\nresult = 'nope'\n```")
    assert parsed.final is None
    assert parsed.rejection is None


def test_empty_directive_is_rejected() -> None:
    final, rejection = find_final_answer("FINAL()")
    assert final is None
    assert rejection is Rejection.EMPTY_ANSWER


def test_answer_before_any_code_has_run_is_refused() -> None:
    parsed = parse_turn("FINAL(Paris, obviously)", code_has_run=False)
    assert parsed.final is None
    assert parsed.rejection is Rejection.NO_CODE_HAS_RUN
    accepted = parse_turn("FINAL(Paris, obviously)", code_has_run=True)
    assert accepted.final is not None


def test_code_and_an_answer_in_one_turn_runs_the_code() -> None:
    text = "```repl\nsummary = build()\n```\nFINAL_VAR(summary)"
    parsed = parse_turn(text)
    assert len(parsed.executable_blocks) == 1
    assert parsed.final is None
    assert parsed.rejection is Rejection.CODE_IN_SAME_TURN


def test_answer_alongside_a_non_executable_fence_is_accepted() -> None:
    text = "```json\n{\"note\": 1}\n```\nFINAL(the answer)"
    parsed = parse_turn(text)
    assert parsed.final is not None
    assert parsed.final.value == "the answer"


def test_unicode_and_long_answers_pass_through_unchanged() -> None:
    payload = chr(0x4E2D) + chr(0x6587) + " " + chr(0x00E9) + "e " + "x" * 50_000
    final, _ = find_final_answer(f"FINAL({payload})")
    assert final is not None
    assert final.value == payload


def test_answer_containing_backticks_is_kept_verbatim() -> None:
    final, _ = find_final_answer("FINAL(call `llm_query` on each chunk)")
    assert final is not None
    assert final.value == "call `llm_query` on each chunk"


# --------------------------------------------------------------------------
# properties


ANSWER_CHARS = st.characters(codec="utf-8", exclude_characters="()`~")

# Balanced by construction rather than by filtering, so the generator spends
# its time on deep nesting instead of on rejected samples.
BALANCED_ANSWERS = st.lists(
    st.recursive(
        st.text(alphabet=ANSWER_CHARS, max_size=30),
        lambda inner: st.lists(inner, min_size=1, max_size=3).map(
            lambda parts: "(" + "".join(parts) + ")"
        ),
        max_leaves=6,
    ),
    min_size=1,
    max_size=4,
).map("".join)


@settings(max_examples=300, deadline=None)
@given(st.text(alphabet=ANSWER_CHARS, min_size=1, max_size=400).map(str.strip))
def test_any_paren_free_answer_round_trips(answer: str) -> None:
    assume(answer)
    assume("FINAL" not in answer)
    final, _ = find_final_answer(f"FINAL({answer})")
    assert final is not None
    assert final.value == answer


@settings(max_examples=300, deadline=None)
@given(BALANCED_ANSWERS.map(str.strip))
def test_balanced_answers_round_trip_whatever_the_nesting(answer: str) -> None:
    assume(answer)
    assume("FINAL" not in answer)
    assert _balanced(answer)
    final, _ = find_final_answer(f"FINAL({answer})")
    assert final is not None
    assert final.value == answer


@settings(max_examples=200, deadline=None)
@given(st.text(alphabet=ANSWER_CHARS, min_size=1, max_size=200))
def test_a_directive_mentioned_in_code_never_fires(answer: str) -> None:
    assume("`" not in answer and "\n" not in answer)
    text = f"```repl\n# FINAL({answer})\nprint(1)\n```\nStill working."
    parsed = parse_turn(text)
    assert parsed.final is None
    assert parsed.rejection is None
    assert len(parsed.executable_blocks) == 1


@settings(max_examples=200, deadline=None)
@given(st.lists(st.text(alphabet=ANSWER_CHARS, max_size=60), min_size=1, max_size=5))
def test_code_blocks_round_trip(bodies: list[str]) -> None:
    for body in bodies:
        assume("```" not in body and "~~~" not in body)
        assume(not any(line.strip().startswith("`") for line in body.splitlines()))
    text = "\n".join(f"```repl\n{body}\n```" for body in bodies)
    blocks = extract_code_blocks(text)
    assert len(blocks) == len(bodies)
    assert [b.code for b in blocks] == [f"{body}\n" for body in bodies]


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=300))
def test_parsing_never_raises(text: str) -> None:
    for flag in (True, False):
        parsed = parse_turn(text, code_has_run=flag)
        assert isinstance(parsed.code_blocks, tuple)
