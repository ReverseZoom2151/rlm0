"""The corpus has to be reproducible, adversarial, and honest about itself.

Determinism is tested at the byte level rather than by comparing answers,
because a corpus that produces the same labels from different text is not
reproducible in the way a published number needs it to be.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from rlm0.harness.corpus import (
    Corpus,
    CorpusSpec,
    GroundTruthError,
    Regime,
    TaskFamily,
    generate_corpus,
    verify_sample,
)

from .conftest import SMALL


class TestDeterminism:
    def test_the_same_seed_yields_byte_identical_corpora(self) -> None:
        first = generate_corpus(SMALL).to_json()
        second = generate_corpus(SMALL).to_json()
        assert first == second

    def test_a_different_seed_yields_a_different_corpus(self) -> None:
        other = dataclasses.replace(SMALL, seed=SMALL.seed + 1)
        assert (
            generate_corpus(SMALL).content_hash
            != generate_corpus(other).content_hash
        )

    def test_the_hash_covers_the_spec_not_just_the_seed(self) -> None:
        # A knob change with the seed held fixed must produce a different
        # corpus identity, or a result carried forward across a settings
        # change silently attaches to the wrong text.
        louder = dataclasses.replace(SMALL, distractor_similarity=0.1)
        assert generate_corpus(SMALL).content_hash != generate_corpus(
            louder
        ).content_hash


class TestSelfVerification:
    def test_a_corrupted_answer_is_caught(self, corpus: Corpus) -> None:
        sample = corpus.samples[0]
        corrupted = dataclasses.replace(sample, answer="1234567")
        with pytest.raises(GroundTruthError, match="text supports answer"):
            verify_sample(corrupted)

    def test_a_corrupted_evidence_set_is_caught(self, corpus: Corpus) -> None:
        sample = next(s for s in corpus.samples if len(s.required_doc_ids) > 1)
        trimmed = dataclasses.replace(
            sample,
            required_doc_ids=frozenset(sorted(sample.required_doc_ids)[:1]),
        )
        with pytest.raises(GroundTruthError, match="evidence set disagrees"):
            verify_sample(trimmed)

    def test_an_answer_listed_as_its_own_distractor_is_caught(
        self, corpus: Corpus
    ) -> None:
        sample = corpus.samples[0]
        confused = dataclasses.replace(
            sample, distractor_answers=(*sample.distractor_answers, sample.answer)
        )
        with pytest.raises(GroundTruthError, match="also listed as a distractor"):
            verify_sample(confused)

    def test_every_generated_sample_verifies(self, corpus: Corpus) -> None:
        for sample in corpus.samples:
            verify_sample(sample)


class TestTaskFamilies:
    def test_every_family_is_generated(self, corpus: Corpus) -> None:
        assert {s.family for s in corpus.samples} == set(TaskFamily)

    def test_the_regimes_are_separable(self, corpus: Corpus) -> None:
        retrieval = [s for s in corpus.samples if s.regime is Regime.RETRIEVAL]
        aggregation = [s for s in corpus.samples if s.regime is Regime.AGGREGATION]
        assert retrieval and aggregation
        # The dimension that decides whether recursion can pay: aggregation
        # items cannot be answered from a handful of documents.
        assert max(len(s.required_doc_ids) for s in retrieval) < min(
            len(s.required_doc_ids) for s in aggregation
        )

    def test_multi_hop_cannot_be_answered_from_one_document(
        self, corpus: Corpus
    ) -> None:
        sample = next(s for s in corpus.samples if s.family is TaskFamily.MULTI_HOP)
        start = next(
            token.strip(".,?")
            for token in sample.question.split()
            if token.strip(".,?").startswith("NODE-")
        )
        for document in sample.documents:
            text = document.render()
            assert not (start in text and sample.answer in text)

    def test_supersession_does_not_put_the_current_record_last(
        self, corpus: Corpus
    ) -> None:
        sample = next(s for s in corpus.samples if s.family is TaskFamily.SUPERSESSION)
        # A recency shortcut reads the end of the context. The record that is
        # currently in force is never the last thing there.
        assert sample.answer not in sample.documents[-1].render()

    def test_aggregate_argmax_punishes_reading_the_biggest_number(
        self, corpus: Corpus
    ) -> None:
        sample = next(
            s for s in corpus.samples if s.family is TaskFamily.AGGREGATE_ARGMAX
        )
        # The spike subject holds the single largest record and is not the
        # answer, so a system that reports the largest value's owner is wrong.
        assert sample.distractor_answers[0] != sample.answer


class TestShortcutsAreClosed:
    def test_identifiers_are_opaque(self, corpus: Corpus) -> None:
        # Nothing about a document is guessable from its name: no word of the
        # question appears in it, and the names are drawn from a fixed
        # alphabet that carries no meaning at all.
        pattern = re.compile(r"^DOC-[A-HJ-NP-Z2-9]{8}$")
        for sample in corpus.samples:
            words = {word.strip(".,?").lower() for word in sample.question.split()}
            for document in sample.documents:
                assert pattern.match(document.doc_id)
                assert document.doc_id.lower() not in words

    def test_required_evidence_is_a_small_share_of_the_context(
        self, corpus: Corpus
    ) -> None:
        # Otherwise citing everything would be indistinguishable from
        # citing the right thing.
        for sample in corpus.samples:
            assert len(sample.required_doc_ids) < len(sample.documents)

    def test_distractor_similarity_changes_the_text(self) -> None:
        near = generate_corpus(dataclasses.replace(SMALL, distractor_similarity=1.0))
        far = generate_corpus(dataclasses.replace(SMALL, distractor_similarity=0.0))
        assert near.samples[0].context() != far.samples[0].context()

    def test_a_solver_task_carries_no_answer(self, corpus: Corpus) -> None:
        task = corpus.samples[0].as_task()
        assert not hasattr(task, "answer")
        assert not hasattr(task, "required_doc_ids")


class TestSpecValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("distractor_similarity", 1.5),
            ("samples_per_family", 0),
            ("n_distractors", 1),
            ("chain_length", 1),
            ("cohort_size", 3),
            ("hops", 4),
            ("filler_lines", 0),
        ],
    )
    def test_a_degenerate_spec_is_refused(self, field: str, value: object) -> None:
        with pytest.raises(ValueError):
            CorpusSpec(seed=1, **{field: value})  # type: ignore[arg-type]
