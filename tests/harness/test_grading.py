"""Grading has to separate the answer from the reason for it."""

from __future__ import annotations

from rlm0.harness.corpus import Corpus, Sample
from rlm0.harness.grading import GradingPolicy, grade, normalise, summarise


def _sample(corpus: Corpus) -> Sample:
    return next(s for s in corpus.samples if len(s.required_doc_ids) >= 2)


def _wrong_docs(sample: Sample, n: int) -> tuple[str, ...]:
    others = [
        doc.doc_id
        for doc in sample.documents
        if doc.doc_id not in sample.required_doc_ids
    ]
    return tuple(others[:n])


class TestEvidenceSeparatesLuckFromSolving:
    def test_right_answer_with_wrong_evidence_scores_below_right_evidence(
        self, corpus: Corpus
    ) -> None:
        sample = _sample(corpus)
        good = grade(sample, sample.answer, sorted(sample.required_doc_ids))
        lucky = grade(sample, sample.answer, _wrong_docs(sample, 2))
        assert good.score > lucky.score
        assert good.supported and not lucky.supported
        # Both are right on the axis everyone else publishes, which is the
        # entire reason that axis is not enough.
        assert good.answer_correct and lucky.answer_correct

    def test_a_right_answer_citing_nothing_is_not_supported(
        self, corpus: Corpus
    ) -> None:
        sample = _sample(corpus)
        score = grade(sample, sample.answer, [])
        assert score.answer_correct
        assert not score.supported
        assert score.score == 0.0

    def test_citing_everything_does_not_buy_support(self, corpus: Corpus) -> None:
        sample = _sample(corpus)
        score = grade(sample, sample.answer, [d.doc_id for d in sample.documents])
        assert score.evidence_recall == 1.0
        assert not score.supported, "recall bought by spam must not count"

    def test_partial_evidence_can_be_allowed_by_policy(self, corpus: Corpus) -> None:
        sample = _sample(corpus)
        partial = sorted(sample.required_doc_ids)[:1]
        strict = grade(sample, sample.answer, partial)
        lenient = grade(
            sample,
            sample.answer,
            partial,
            policy=GradingPolicy(require_complete_evidence=False),
        )
        assert not strict.supported
        assert lenient.supported


class TestPlausibleIsNotPartlyRight:
    def test_a_distractor_answer_scores_zero_and_is_flagged(
        self, corpus: Corpus
    ) -> None:
        sample = next(s for s in corpus.samples if s.distractor_answers)
        score = grade(
            sample, sample.distractor_answers[0], sorted(sample.required_doc_ids)
        )
        assert score.score == 0.0
        assert not score.answer_correct
        assert score.matched_distractor

    def test_no_answer_at_all_scores_zero(self, corpus: Corpus) -> None:
        sample = corpus.samples[0]
        score = grade(sample, None, sorted(sample.required_doc_ids))
        assert not score.answered
        assert score.score == 0.0


class TestNormalisation:
    def test_formatting_is_forgiven_but_content_is_not(self, corpus: Corpus) -> None:
        sample = next(s for s in corpus.samples if s.answer.isdigit())
        assert grade(sample, f" {sample.answer}. ", []).answer_correct
        assert not grade(sample, sample.answer + "1", []).answer_correct

    def test_normalise_folds_only_formatting(self) -> None:
        assert normalise(" 'SUBJ-9K2'. ") == "subj-9k2"
        assert normalise("1,024") == "1 024"


class TestSummary:
    def test_the_luck_gap_is_the_headline_minus_the_support(
        self, corpus: Corpus
    ) -> None:
        sample = _sample(corpus)
        scores = [
            grade(sample, sample.answer, sorted(sample.required_doc_ids)),
            grade(sample, sample.answer, []),
        ]
        summary = summarise(scores)
        assert summary.answer_accuracy == 1.0
        assert summary.supported_accuracy == 0.5
        assert summary.luck_gap == 0.5
