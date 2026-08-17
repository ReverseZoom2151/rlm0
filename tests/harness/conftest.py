"""Shared fixtures. Small corpora, because every test here is offline."""

from __future__ import annotations

import pytest

from rlm0.harness.corpus import Corpus, CorpusSpec, generate_corpus

SMALL = CorpusSpec(
    seed=11,
    samples_per_family=1,
    n_distractors=2,
    distractor_similarity=0.8,
    filler_docs=2,
    filler_lines=1,
    chain_length=2,
    cohort_size=6,
    hops=2,
)


@pytest.fixture(scope="session")
def spec() -> CorpusSpec:
    return SMALL


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return generate_corpus(SMALL)
