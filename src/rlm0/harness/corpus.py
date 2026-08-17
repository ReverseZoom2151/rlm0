"""A deterministic adversarial corpus that checks its own ground truth.

Long-context benchmarks are easy to pass by accident. A filename that names
its own contents, a needle written in vocabulary nothing else in the haystack
uses, an answer that happens to be the most recent record because the most
recent record was appended last: each of these lets a system score well
without doing the work the benchmark claims to measure. Every shortcut here is
closed deliberately.

Identifiers are opaque tokens, so nothing is guessable from a name and the
only way to tell two surface-identical sentences apart is to read the
identifier. Distractors are paraphrases whose closeness to the answer is a
single tunable number, so the difficulty of the corpus is a parameter and not
a property of whatever the author happened to type. Records supersede one
another through explicit chains that are shuffled before emission, so recency
has to be resolved from the text rather than assumed from position. Multi-hop
items require joining two or three documents, so retrieving one is not enough.

The families split along the axis that decides whether this whole technique
is worth anything. Retrieval-shaped work is where the published evidence says
depth zero already wins, and dense aggregation is where recursion is supposed
to earn its cost. A benchmark that mixes the two into one accuracy number
cannot say which regime a system is in, which is the only interesting question
here.

Finally, the generator re-derives every answer from the text it just emitted,
the way a solver would, and refuses to hand back a corpus whose ground truth
disagrees. Most synthetic benchmarks trust the variable they built the text
from, which means a rendering bug becomes a silently wrong label.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from random import Random
from typing import Any

__all__ = [
    "Corpus",
    "CorpusSpec",
    "Document",
    "GroundTruthError",
    "Regime",
    "Sample",
    "SolverTask",
    "TaskFamily",
    "generate_corpus",
    "verify_sample",
]


class GroundTruthError(AssertionError):
    """The emitted text does not support the answer that was recorded for it.

    An AssertionError subclass because it always means a generator bug rather
    than bad input. A corpus that raises this must never be scored against.
    """


class Regime(StrEnum):
    """The axis along which recursion is or is not worth paying for.

    Kept as an explicit dimension rather than left implicit in the family name
    because every headline accuracy figure in this project is reported broken
    down by it. The source paper's own finding is that the REPL handles length
    while recursion helps on information density, so a system can be excellent
    in one regime and wasteful in the other while a pooled average says
    nothing about either.
    """

    RETRIEVAL = "retrieval"
    AGGREGATION = "aggregation"


class TaskFamily(StrEnum):
    """What shape of work a sample demands.

    Three retrieval shapes and two aggregation shapes. The retrieval shapes
    are ordered by how much resolution they need beyond locating a string:
    none, then a supersession chain, then a join across documents. The
    aggregation shapes cannot be answered from any single document at all,
    which is what makes them the case for fanning out.
    """

    NEEDLE = "needle"
    SUPERSESSION = "supersession"
    MULTI_HOP = "multi_hop"
    AGGREGATE_COUNT = "aggregate_count"
    AGGREGATE_ARGMAX = "aggregate_argmax"

    @property
    def regime(self) -> Regime:
        if self in {TaskFamily.AGGREGATE_COUNT, TaskFamily.AGGREGATE_ARGMAX}:
            return Regime.AGGREGATION
        return Regime.RETRIEVAL


@dataclass(frozen=True, slots=True)
class Document:
    """One document in a sample's context.

    The identifier carries no information about the contents. That is the
    point: a citation is only evidence that the document was read if the
    identifier could not have been guessed from the question.
    """

    doc_id: str
    lines: tuple[str, ...]

    def render(self) -> str:
        return "\n".join((f"DOCUMENT {self.doc_id}", *self.lines))


@dataclass(frozen=True, slots=True)
class SolverTask:
    """Everything a solver is allowed to see.

    Deliberately not the Sample. Handing a solver the object that also holds
    the answer is how an evaluation harness ends up unable to distinguish a
    system from a lookup, and at least one surveyed implementation passed the
    full record through to the thing being measured.
    """

    sample_id: str
    family: TaskFamily
    question: str
    documents: tuple[Document, ...]

    def context(self) -> str:
        return "\n\n".join(doc.render() for doc in self.documents)

    @property
    def doc_ids(self) -> tuple[str, ...]:
        return tuple(doc.doc_id for doc in self.documents)


@dataclass(frozen=True, slots=True)
class Sample:
    """One question, its context, its answer and the evidence it requires.

    `required_doc_ids` is the set a correct solve must have read. It is not
    the same as the set that contains the answer string: on a supersession
    chain you have to read the stale records to know they are stale, and on an
    aggregation you have to read every member of the cohort to know the count.
    Grading the evidence against this set is what separates a solve from a
    lucky guess.

    `distractor_answers` are the values a system lands on when it matches on
    surface similarity instead of resolving the identifier. They are recorded
    so the report can say how often a wrong answer was the plausible wrong
    answer, which is a far more useful failure signal than a bare error rate.
    """

    sample_id: str
    family: TaskFamily
    question: str
    documents: tuple[Document, ...]
    answer: str
    required_doc_ids: frozenset[str]
    distractor_answers: tuple[str, ...] = ()

    @property
    def regime(self) -> Regime:
        return self.family.regime

    def context(self) -> str:
        return "\n\n".join(doc.render() for doc in self.documents)

    def as_task(self) -> SolverTask:
        return SolverTask(
            sample_id=self.sample_id,
            family=self.family,
            question=self.question,
            documents=self.documents,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "family": self.family.value,
            "question": self.question,
            "documents": [
                {"doc_id": doc.doc_id, "lines": list(doc.lines)}
                for doc in self.documents
            ],
            "answer": self.answer,
            "required_doc_ids": sorted(self.required_doc_ids),
            "distractor_answers": list(self.distractor_answers),
        }


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """Every knob that changes what gets generated, and nothing else.

    The spec is hashed into the corpus identity, so a number can never be
    attributed to a corpus that was generated under different settings. The
    similarity knob is the one that matters most: it is the difference between
    a benchmark that measures identifier resolution and one that measures
    keyword overlap.
    """

    seed: int
    samples_per_family: int = 2
    n_distractors: int = 6
    distractor_similarity: float = 0.8
    filler_docs: int = 8
    filler_lines: int = 4
    chain_length: int = 4
    cohort_size: int = 12
    hops: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.distractor_similarity <= 1.0:
            raise ValueError("distractor_similarity must lie in [0, 1]")
        if self.samples_per_family < 1:
            raise ValueError("samples_per_family must be at least 1")
        if self.n_distractors < 2:
            raise ValueError(
                "n_distractors below 2 makes a corpus that rewards guessing"
            )
        if self.chain_length < 2:
            raise ValueError("a supersession chain needs at least two records")
        if self.cohort_size < 6:
            raise ValueError(
                "a cohort below 6 members can be aggregated by reading it all "
                "at once, which defeats the point of the aggregation family"
            )
        if self.hops not in {2, 3}:
            raise ValueError("hops must be 2 or 3")
        if self.filler_lines < 1:
            raise ValueError("filler_lines must be at least 1")


@dataclass(frozen=True, slots=True)
class Corpus:
    """A generated suite, identified by the hash of its own bytes.

    The hash is over the canonical serialisation rather than over the seed,
    because a change to the generator that keeps the seed produces a different
    corpus and a result carried forward against it would be silently wrong.
    """

    spec: CorpusSpec
    samples: tuple[Sample, ...]

    def to_json(self) -> str:
        payload = {
            "spec": asdict(self.spec),
            "samples": [sample.to_dict() for sample in self.samples],
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=True, indent=2)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256(self.to_json().encode("utf-8"))
        return digest.hexdigest()

    def by_id(self, sample_id: str) -> Sample:
        for sample in self.samples:
            if sample.sample_id == sample_id:
                return sample
        raise KeyError(sample_id)


# Surface forms. Index 0 of each is canonical and is the only form the answer
# is ever written in, which is what lets the self-check re-derive answers with
# a regex instead of trusting the variable the text was built from.
_CANONICAL_ATTRIBUTE = "retention margin"
_ATTRIBUTES = (
    _CANONICAL_ATTRIBUTE,
    "retention window",
    "retention envelope",
    "clearance margin",
    "settlement band",
)
_TEMPLATES = (
    "Record {rid} states that the {attr} for {subj} is {val}.",
    "Record {rid} states that the {attr} for {subj} was set to {val}.",
    "Record {rid} reports the {attr} of {subj} as {val}.",
    "Per record {rid}, {subj} carries a {attr} of {val}.",
    "{rid} logs {val} against the {attr} associated with {subj}.",
)
_NOISE = (
    "Note: transcription pass {n} completed without exception.",
    "Cross reference index {ref} was rebuilt during the same cycle.",
    "Filed under routing bucket {n} pending review.",
    "No adjustment was applied to this entry in cycle {n}.",
)

_FACT_RE = re.compile(
    r"^Record (\S+) states that the retention margin for (\S+) is (\d+)\.$"
)
_SUPERSEDES_RE = re.compile(r"^Record (\S+) supersedes record (\S+)\.$")
_ROUTE_RE = re.compile(r"^Node (\S+) routes to node (\S+)\.$")
_COHORT_RE = re.compile(r"^cohort: (\S+)$")

_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


class _Ids:
    """Opaque identifiers, unique within one sample.

    Uniqueness is enforced rather than hoped for because the near-miss
    generator deliberately produces identifiers one character apart, and a
    collision there would turn a distractor into a second copy of the answer.
    """

    def __init__(self, rng: Random) -> None:
        self._rng = rng
        self._used: set[str] = set()

    def make(self, prefix: str, length: int = 8) -> str:
        while True:
            token = "".join(self._rng.choice(_ALPHABET) for _ in range(length))
            candidate = f"{prefix}-{token}"
            if candidate not in self._used:
                self._used.add(candidate)
                return candidate

    def near_miss(self, existing: str) -> str:
        """An identifier one character from an existing one.

        The single most effective distractor available, because it survives
        every fuzzy match and fails only on an exact comparison.
        """
        prefix, _, token = existing.partition("-")
        for _ in range(200):
            position = self._rng.randrange(len(token))
            replacement = self._rng.choice(_ALPHABET)
            tail = token[position + 1 :]
            candidate = f"{prefix}-{token[:position]}{replacement}{tail}"
            if candidate != existing and candidate not in self._used:
                self._used.add(candidate)
                return candidate
        return self.make(prefix, len(token))


def _seed_for(spec: CorpusSpec, family: TaskFamily, index: int) -> int:
    """A stable per-sample seed.

    Derived through a digest rather than by arithmetic on the seed so that
    neighbouring seeds do not produce correlated samples, and so the same
    sample is reproducible without generating the ones before it.
    """
    key = f"{spec.seed}:{family.value}:{index}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def _canonical_fact(rid: str, subj: str, val: int) -> str:
    return _TEMPLATES[0].format(rid=rid, attr=_CANONICAL_ATTRIBUTE, subj=subj, val=val)


def _by_similarity(similarity: float, options: int, floor: int = 0) -> int:
    """Pick an index where similarity 1.0 is the closest available option.

    Closest, not identical: the canonical form is reserved for statements that
    really are about the subject in question, so a paraphrase at similarity
    1.0 is the nearest thing that is still not the answer.
    """
    span = options - 1 - floor
    return floor + round((1.0 - similarity) * span)


def _paraphrase(rng: Random, ids: _Ids, similarity: float, val: int) -> str:
    """A distractor sentence whose distance from the answer form is tunable."""
    attr = _ATTRIBUTES[_by_similarity(similarity, len(_ATTRIBUTES), floor=1)]
    template = _TEMPLATES[_by_similarity(similarity, len(_TEMPLATES))]
    return template.format(
        rid=ids.make("REC"), attr=attr, subj=ids.make("SUBJ", 6), val=val
    )


def _near_miss_value(rng: Random, value: int) -> int:
    digits = list(str(value))
    position = rng.randrange(len(digits))
    for _ in range(20):
        replacement = str(rng.randrange(10))
        leading_zero = position == 0 and replacement == "0"
        if replacement != digits[position] and not leading_zero:
            digits[position] = replacement
            return int("".join(digits))
    return value + 1


def _distractor_value(rng: Random, similarity: float, value: int) -> int:
    if similarity >= 0.5:
        return _near_miss_value(rng, value)
    return rng.randrange(1000, 9999)


def _noise_lines(rng: Random, ids: _Ids, count: int) -> list[str]:
    lines: list[str] = []
    for _ in range(count):
        template = _NOISE[rng.randrange(len(_NOISE))]
        lines.append(
            template.format(n=rng.randrange(10, 99), ref=ids.make("IDX", 6))
        )
    return lines


def _make_doc(
    ids: _Ids,
    rng: Random,
    spec: CorpusSpec,
    body: list[str],
    *,
    cohort: str | None = None,
) -> Document:
    """Wrap body lines in noise, with the body scattered rather than first.

    Position is information. A benchmark whose answers always sit at the top
    of the document rewards reading the top of every document.
    """
    lines = _noise_lines(rng, ids, spec.filler_lines)
    for line in body:
        lines.insert(rng.randrange(len(lines) + 1), line)
    if cohort is not None:
        lines.insert(0, f"cohort: {cohort}")
    return Document(doc_id=ids.make("DOC"), lines=tuple(lines))


def _filler_docs(ids: _Ids, rng: Random, spec: CorpusSpec) -> list[Document]:
    """Bulk that contains no canonical fact and belongs to no cohort.

    Filler exists to make the context long. It must never be able to change an
    answer, which is why it is built only from paraphrase forms.
    """
    docs: list[Document] = []
    for _ in range(spec.filler_docs):
        body = [
            _paraphrase(rng, ids, spec.distractor_similarity, rng.randrange(1000, 9999))
            for _ in range(2)
        ]
        docs.append(_make_doc(ids, rng, spec, body))
    return docs


def _shuffled(rng: Random, docs: list[Document]) -> tuple[Document, ...]:
    ordered = list(docs)
    rng.shuffle(ordered)
    return tuple(ordered)


def _build_needle(spec: CorpusSpec, index: int) -> Sample:
    rng = Random(_seed_for(spec, TaskFamily.NEEDLE, index))
    ids = _Ids(rng)
    subject = ids.make("SUBJ", 6)
    value = rng.randrange(1000, 9999)
    answer_doc = _make_doc(
        ids, rng, spec, [_canonical_fact(ids.make("REC"), subject, value)]
    )

    docs = [answer_doc]
    distractor_answers: list[str] = []
    for i in range(spec.n_distractors):
        wrong_value = _distractor_value(rng, spec.distractor_similarity, value)
        distractor_answers.append(str(wrong_value))
        if i % 2 == 0:
            # Same sentence, same attribute, an identifier one character off.
            # Nothing but the identifier separates this from the answer.
            other = ids.near_miss(subject)
            body = [_canonical_fact(ids.make("REC"), other, wrong_value)]
        else:
            # The right subject discussed under a paraphrased attribute, which
            # is what an attribute-blind matcher falls for.
            attr = _ATTRIBUTES[
                _by_similarity(spec.distractor_similarity, len(_ATTRIBUTES), floor=1)
            ]
            template = _TEMPLATES[
                _by_similarity(spec.distractor_similarity, len(_TEMPLATES))
            ]
            body = [
                template.format(
                    rid=ids.make("REC"), attr=attr, subj=subject, val=wrong_value
                )
            ]
        docs.append(_make_doc(ids, rng, spec, body))
    docs.extend(_filler_docs(ids, rng, spec))

    return Sample(
        sample_id=f"needle-{index:03d}-{ids.make('SMP', 6)}",
        family=TaskFamily.NEEDLE,
        question=f"What retention margin is recorded for {subject}?",
        documents=_shuffled(rng, docs),
        answer=str(value),
        required_doc_ids=frozenset({answer_doc.doc_id}),
        distractor_answers=tuple(distractor_answers),
    )


def _build_supersession(spec: CorpusSpec, index: int) -> Sample:
    rng = Random(_seed_for(spec, TaskFamily.SUPERSESSION, index))
    ids = _Ids(rng)
    subject = ids.make("SUBJ", 6)

    values = rng.sample(range(1000, 9999), spec.chain_length)
    records = [ids.make("REC") for _ in range(spec.chain_length)]
    chain_docs: list[Document] = []
    for position, (rid, value) in enumerate(zip(records, values, strict=True)):
        body = [_canonical_fact(rid, subject, value)]
        if position > 0:
            body.append(f"Record {rid} supersedes record {records[position - 1]}.")
        chain_docs.append(_make_doc(ids, rng, spec, body))

    docs = list(chain_docs)
    distractor_answers = [str(v) for v in values[:-1]]
    for _ in range(spec.n_distractors):
        # A parallel chain about a near-miss subject, so following the wrong
        # chain to its end still produces a confidently wrong answer.
        other = ids.near_miss(subject)
        other_records = [ids.make("REC") for _ in range(2)]
        other_value = _distractor_value(rng, spec.distractor_similarity, values[-1])
        distractor_answers.append(str(other_value))
        docs.append(
            _make_doc(
                ids,
                rng,
                spec,
                [
                    _canonical_fact(other_records[0], other, other_value),
                    f"Record {other_records[1]} supersedes record {other_records[0]}.",
                ],
            )
        )
    docs.extend(_filler_docs(ids, rng, spec))

    ordered = list(_shuffled(rng, docs))
    terminal = chain_docs[-1]
    if ordered[-1].doc_id == terminal.doc_id:
        # Never let the current record be the last thing in the context. A
        # benchmark where recency of position tracks recency of record is one
        # that rewards reading the end.
        ordered[0], ordered[-1] = ordered[-1], ordered[0]

    return Sample(
        sample_id=f"supersession-{index:03d}-{ids.make('SMP', 6)}",
        family=TaskFamily.SUPERSESSION,
        question=(
            f"After applying every supersession, what retention margin is "
            f"currently in force for {subject}?"
        ),
        documents=tuple(ordered),
        answer=str(values[-1]),
        required_doc_ids=frozenset(doc.doc_id for doc in chain_docs),
        distractor_answers=tuple(distractor_answers),
    )


def _build_multi_hop(spec: CorpusSpec, index: int) -> Sample:
    rng = Random(_seed_for(spec, TaskFamily.MULTI_HOP, index))
    ids = _Ids(rng)
    nodes = [ids.make("NODE", 6) for _ in range(spec.hops + 1)]
    value = rng.randrange(1000, 9999)

    hop_docs: list[Document] = []
    for position in range(spec.hops):
        hop_docs.append(
            _make_doc(
                ids,
                rng,
                spec,
                [f"Node {nodes[position]} routes to node {nodes[position + 1]}."],
            )
        )
    terminal_doc = _make_doc(
        ids, rng, spec, [_canonical_fact(ids.make("REC"), nodes[-1], value)]
    )
    required = [*hop_docs, terminal_doc]

    docs = list(required)
    distractor_answers: list[str] = []
    for i in range(spec.n_distractors):
        # Decoy chains that begin at a near-miss of a node on the real chain,
        # so a solver that resolves one identifier loosely lands on a complete
        # and self-consistent wrong chain rather than on nothing.
        start = ids.near_miss(nodes[i % len(nodes)])
        middle = ids.make("NODE", 6)
        end = ids.make("NODE", 6)
        wrong_value = _distractor_value(rng, spec.distractor_similarity, value)
        distractor_answers.append(str(wrong_value))
        for line in (
            f"Node {start} routes to node {middle}.",
            f"Node {middle} routes to node {end}.",
            _canonical_fact(ids.make("REC"), end, wrong_value),
        ):
            docs.append(_make_doc(ids, rng, spec, [line]))
    docs.extend(_filler_docs(ids, rng, spec))

    return Sample(
        sample_id=f"multi_hop-{index:03d}-{ids.make('SMP', 6)}",
        family=TaskFamily.MULTI_HOP,
        question=(
            f"Follow the routing chain that begins at node {nodes[0]}. What "
            f"retention margin is recorded for the node at the end of it?"
        ),
        documents=_shuffled(rng, docs),
        answer=str(value),
        required_doc_ids=frozenset(doc.doc_id for doc in required),
        distractor_answers=tuple(distractor_answers),
    )


def _decoy_cohorts(
    ids: _Ids, rng: Random, spec: CorpusSpec, cohort: str
) -> list[Document]:
    """Cohorts that look like the target one, including a one-character twin.

    An aggregation is only a test of aggregation if the set being aggregated
    has to be determined exactly. Otherwise a system that sums everything that
    looks roughly relevant scores the same as one that reads the membership.
    """
    docs: list[Document] = []
    twin = ids.near_miss(cohort)
    for other in (twin, ids.make("CHT", 6)):
        for _ in range(spec.cohort_size):
            docs.append(
                _make_doc(
                    ids,
                    rng,
                    spec,
                    [
                        _canonical_fact(
                            ids.make("REC"),
                            ids.make("SUBJ", 6),
                            rng.randrange(1000, 9999),
                        )
                    ],
                    cohort=other,
                )
            )
    return docs


def _build_aggregate_count(spec: CorpusSpec, index: int) -> Sample:
    rng = Random(_seed_for(spec, TaskFamily.AGGREGATE_COUNT, index))
    ids = _Ids(rng)
    cohort = ids.make("CHT", 6)
    values = rng.sample(range(1000, 9999), spec.cohort_size)
    threshold = sorted(values)[spec.cohort_size // 2]
    count = sum(1 for value in values if value > threshold)

    member_docs = [
        _make_doc(
            ids,
            rng,
            spec,
            [_canonical_fact(ids.make("REC"), ids.make("SUBJ", 6), value)],
            cohort=cohort,
        )
        for value in values
    ]
    docs = [*member_docs, *_decoy_cohorts(ids, rng, spec, cohort)]
    docs.extend(_filler_docs(ids, rng, spec))

    # The plausible wrong answers are the off-by-one from reading the
    # comparison as inclusive, and the count over everything in sight.
    inclusive = sum(1 for value in values if value >= threshold)
    return Sample(
        sample_id=f"aggregate_count-{index:03d}-{ids.make('SMP', 6)}",
        family=TaskFamily.AGGREGATE_COUNT,
        question=(
            f"Across every record in cohort {cohort}, how many report a "
            f"retention margin strictly greater than {threshold}?"
        ),
        documents=_shuffled(rng, docs),
        answer=str(count),
        required_doc_ids=frozenset(doc.doc_id for doc in member_docs),
        distractor_answers=(str(inclusive), str(spec.cohort_size), "0"),
    )


def _build_aggregate_argmax(spec: CorpusSpec, index: int) -> Sample:
    rng = Random(_seed_for(spec, TaskFamily.AGGREGATE_ARGMAX, index))
    ids = _Ids(rng)
    cohort = ids.make("CHT", 6)

    n_subjects = max(3, spec.cohort_size // 3)
    subjects = [ids.make("SUBJ", 6) for _ in range(n_subjects)]
    winner, spike = subjects[0], subjects[1]

    # Built so the subject holding the single largest record is not the
    # subject with the largest total. Any system that answers by finding the
    # biggest number rather than by summing lands on the spike, which is the
    # whole reason this family exists.
    holdings: dict[str, list[int]] = {}
    holdings[winner] = [rng.randrange(2000, 2400) for _ in range(4)]
    winner_total = sum(holdings[winner])
    spike_value = winner_total - rng.randrange(200, 400)
    holdings[spike] = [spike_value, rng.randrange(100, 200)]
    for subject in subjects[2:]:
        holdings[subject] = [rng.randrange(300, 900) for _ in range(3)]

    totals = {subject: sum(values) for subject, values in holdings.items()}
    best = max(totals.values())
    tied = sum(1 for total in totals.values() if total == best) > 1
    if tied or totals[winner] != best:
        # Construction should already guarantee this; the adjustment keeps the
        # invariant true rather than leaving it to the random draw.
        holdings[winner] = [*holdings[winner], best - totals[winner] + 1]

    member_docs: list[Document] = []
    for subject, values in holdings.items():
        for value in values:
            member_docs.append(
                _make_doc(
                    ids,
                    rng,
                    spec,
                    [_canonical_fact(ids.make("REC"), subject, value)],
                    cohort=cohort,
                )
            )
    docs = [*member_docs, *_decoy_cohorts(ids, rng, spec, cohort)]
    docs.extend(_filler_docs(ids, rng, spec))

    return Sample(
        sample_id=f"aggregate_argmax-{index:03d}-{ids.make('SMP', 6)}",
        family=TaskFamily.AGGREGATE_ARGMAX,
        question=(
            f"Summing every record in cohort {cohort} by subject, which "
            f"subject has the greatest total retention margin?"
        ),
        documents=_shuffled(rng, docs),
        answer=winner,
        required_doc_ids=frozenset(doc.doc_id for doc in member_docs),
        distractor_answers=(spike, *subjects[2:3]),
    )


_BUILDERS = {
    TaskFamily.NEEDLE: _build_needle,
    TaskFamily.SUPERSESSION: _build_supersession,
    TaskFamily.MULTI_HOP: _build_multi_hop,
    TaskFamily.AGGREGATE_COUNT: _build_aggregate_count,
    TaskFamily.AGGREGATE_ARGMAX: _build_aggregate_argmax,
}


@dataclass(frozen=True, slots=True)
class _Reading:
    """What a solver could learn from the emitted text and nothing else."""

    facts: list[tuple[str, str, str, int]] = field(default_factory=list)
    """(doc_id, record_id, subject, value) for canonical statements only."""

    supersedes: list[tuple[str, str, str]] = field(default_factory=list)
    """(doc_id, superseding_record, superseded_record)."""

    routes: list[tuple[str, str, str]] = field(default_factory=list)
    """(doc_id, from_node, to_node)."""

    cohorts: dict[str, str] = field(default_factory=dict)
    """doc_id to the cohort it declares membership of."""


def _read(sample: Sample) -> _Reading:
    """Parse the context the way a solver would, with no generator state.

    This is the part that makes the self-check worth anything. Re-deriving the
    answer from the objects the text was rendered from would only prove the
    renderer was called; re-deriving it from the rendered text proves the text
    says what the label claims.
    """
    reading = _Reading()
    for doc in sample.documents:
        for line in doc.lines:
            cohort = _COHORT_RE.match(line)
            if cohort is not None:
                reading.cohorts[doc.doc_id] = cohort.group(1)
                continue
            fact = _FACT_RE.match(line)
            if fact is not None:
                reading.facts.append(
                    (doc.doc_id, fact.group(1), fact.group(2), int(fact.group(3)))
                )
                continue
            edge = _SUPERSEDES_RE.match(line)
            if edge is not None:
                reading.supersedes.append((doc.doc_id, edge.group(1), edge.group(2)))
                continue
            route = _ROUTE_RE.match(line)
            if route is not None:
                reading.routes.append((doc.doc_id, route.group(1), route.group(2)))
    return reading


def _subject_of_question(sample: Sample, prefix: str) -> str:
    tokens = [
        token.strip(".,?")
        for token in sample.question.split()
        if token.strip(".,?").startswith(f"{prefix}-")
    ]
    if not tokens:
        raise GroundTruthError(f"{sample.sample_id}: question names no {prefix}")
    return tokens[0]


def _rederive_needle(sample: Sample) -> tuple[str, set[str]]:
    subject = _subject_of_question(sample, "SUBJ")
    hits = [f for f in _read(sample).facts if f[2] == subject]
    if len(hits) != 1:
        raise GroundTruthError(
            f"{sample.sample_id}: {len(hits)} canonical statements about "
            f"{subject}, expected exactly one"
        )
    doc_id, _, _, value = hits[0]
    return str(value), {doc_id}


def _rederive_supersession(sample: Sample) -> tuple[str, set[str]]:
    subject = _subject_of_question(sample, "SUBJ")
    reading = _read(sample)
    mine = {rid: (doc_id, value) for doc_id, rid, subj, value in reading.facts
            if subj == subject}
    superseded = {
        target for _, source, target in reading.supersedes
        if source in mine or target in mine
    }
    current = [rid for rid in mine if rid not in superseded]
    if len(current) != 1:
        raise GroundTruthError(
            f"{sample.sample_id}: {len(current)} records for {subject} are "
            "unsuperseded, expected exactly one"
        )
    return str(mine[current[0]][1]), {doc_id for doc_id, _ in mine.values()}


def _rederive_multi_hop(sample: Sample) -> tuple[str, set[str]]:
    start = _subject_of_question(sample, "NODE")
    reading = _read(sample)
    outgoing: dict[str, tuple[str, str]] = {}
    for doc_id, source, target in reading.routes:
        if source in outgoing:
            raise GroundTruthError(
                f"{sample.sample_id}: node {source} has more than one route"
            )
        outgoing[source] = (doc_id, target)

    evidence: set[str] = set()
    node = start
    for _ in range(len(outgoing) + 1):
        if node not in outgoing:
            break
        doc_id, node = outgoing[node]
        evidence.add(doc_id)
    else:  # pragma: no cover - only reachable on a cyclic generator bug
        raise GroundTruthError(f"{sample.sample_id}: routing chain does not terminate")

    hits = [f for f in reading.facts if f[2] == node]
    if len(hits) != 1:
        raise GroundTruthError(
            f"{sample.sample_id}: chain terminus {node} has {len(hits)} "
            "canonical statements, expected exactly one"
        )
    doc_id, _, _, value = hits[0]
    evidence.add(doc_id)
    return str(value), evidence


def _cohort_of_question(sample: Sample) -> str:
    return _subject_of_question(sample, "CHT")


def _cohort_facts(sample: Sample) -> tuple[list[tuple[str, str, int]], set[str]]:
    cohort = _cohort_of_question(sample)
    reading = _read(sample)
    members = {
        doc_id for doc_id, declared in reading.cohorts.items() if declared == cohort
    }
    facts = [
        (doc_id, subj, value)
        for doc_id, _, subj, value in reading.facts
        if doc_id in members
    ]
    if not facts:
        raise GroundTruthError(f"{sample.sample_id}: cohort {cohort} has no records")
    return facts, members


def _rederive_aggregate_count(sample: Sample) -> tuple[str, set[str]]:
    facts, members = _cohort_facts(sample)
    threshold = int(sample.question.rstrip("?").split()[-1])
    count = sum(1 for _, _, value in facts if value > threshold)
    return str(count), members


def _rederive_aggregate_argmax(sample: Sample) -> tuple[str, set[str]]:
    facts, members = _cohort_facts(sample)
    totals: dict[str, int] = {}
    for _, subject, value in facts:
        totals[subject] = totals.get(subject, 0) + value
    best = max(totals.values())
    winners = sorted(subject for subject, total in totals.items() if total == best)
    if len(winners) != 1:
        raise GroundTruthError(
            f"{sample.sample_id}: {len(winners)} subjects tie for the greatest "
            "total, so the question has no single answer"
        )
    return winners[0], members


_REDERIVERS = {
    TaskFamily.NEEDLE: _rederive_needle,
    TaskFamily.SUPERSESSION: _rederive_supersession,
    TaskFamily.MULTI_HOP: _rederive_multi_hop,
    TaskFamily.AGGREGATE_COUNT: _rederive_aggregate_count,
    TaskFamily.AGGREGATE_ARGMAX: _rederive_aggregate_argmax,
}


def verify_sample(sample: Sample) -> None:
    """Re-derive the answer and the evidence set from the emitted text.

    Raises GroundTruthError on any disagreement. Called on every sample as it
    is generated, so a corpus that reaches a caller has been solved once
    already by something that only had what the solver will have.
    """
    answer, evidence = _REDERIVERS[sample.family](sample)
    if answer != sample.answer:
        raise GroundTruthError(
            f"{sample.sample_id}: text supports answer {answer!r} but the "
            f"corpus records {sample.answer!r}"
        )
    if evidence != set(sample.required_doc_ids):
        missing = sorted(evidence - set(sample.required_doc_ids))
        extra = sorted(set(sample.required_doc_ids) - evidence)
        raise GroundTruthError(
            f"{sample.sample_id}: evidence set disagrees with the text; "
            f"unrecorded {missing}, unsupported {extra}"
        )
    if sample.answer in sample.distractor_answers:
        raise GroundTruthError(
            f"{sample.sample_id}: the answer is also listed as a distractor, "
            "which would make a correct answer count as a plausible mistake"
        )
    known = {doc.doc_id for doc in sample.documents}
    if not set(sample.required_doc_ids) <= known:
        raise GroundTruthError(
            f"{sample.sample_id}: required evidence names documents that are "
            "not in the context"
        )


def generate_corpus(spec: CorpusSpec) -> Corpus:
    """Generate and self-check a corpus.

    Deterministic in the spec alone: the same spec produces byte-identical
    output on any machine and any Python build, because every draw comes from
    a digest-derived seed rather than from anything the interpreter salts.
    """
    samples: list[Sample] = []
    for family in TaskFamily:
        for index in range(spec.samples_per_family):
            sample = _BUILDERS[family](spec, index)
            verify_sample(sample)
            samples.append(sample)
    return Corpus(spec=spec, samples=tuple(samples))
