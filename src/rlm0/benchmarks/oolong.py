"""OOLONG, which is where the aggregation half of this project's claim is tested.

rlm0 claims recursion pays on dense aggregation and loses on retrieval. That
claim is currently testable only on a corpus this project generates itself,
which is necessary and not sufficient: a benchmark whose author also wrote the
system is evidence about the system's fit to the benchmark. OOLONG
(arXiv:2511.02817) is the closest public benchmark to the aggregation half.
Its questions require classifying every chunk of a window and then answering a
distributional question over those classifications, so no amount of retrieval
skill substitutes for reading all of it, and no frontier model exceeds 50
percent at 128K.

## The official metric, and what this module does to it

OOLONG-synth. Answers are parsed by splitting on the last colon, stripping
asterisks and brackets, and assigning a parse confidence. Exact string match
scores 1. Numeric answers get partial credit `0.75 ** abs(gold - guess)`. Date
answers are parsed and compared as dates. Comparison answers ("more common",
"less common", "same frequency") are matched as substrings of the gold.

`score_synth` below reproduces that function, including two behaviours that
look like bugs and are kept because reproducing a metric means reproducing it:
the comparison-phrase normalisation sits in an `elif` after the
short-answer confidence check, so it only fires for candidates of twenty
characters or more, and an unparseable numeric answer downgrades the recorded
parse confidence rather than the score.

The one departure is date parsing. The official harness uses
`dateutil.parser.parse`, which is not a dependency of this project. If
`dateutil` is importable it is used and the metric is reproduced exactly;
otherwise a stdlib parser over a fixed list of formats stands in, and the
manifest says so. Fidelity is therefore REPRODUCES when dateutil is present
and APPROXIMATES when it is not, decided at load time rather than claimed.

OOLONG-real. Answers are extracted from a LaTeX `\\boxed{}` wrapper, then
scored by type: exponential partial credit for integers, case-insensitive
exact match for strings, and set overlap over the gold for lists.
`score_real` reproduces that exactly. Note the consequence, which is sharp: a
solver that does not wrap its answer in `\\boxed{}` scores zero on every real
sample for a formatting reason. `answer_instruction` exists so that the
solver's prompt can be made to comply, and `n_low_confidence_parse` in the
official summary is how a failure to comply shows up as a parse problem rather
than as a capability result.

## What this module changes about the input

The context is chunked into identified documents so the harness can grade
evidence. That is a real deviation from the official prompt, it is recorded in
every manifest, and it is the price of being able to tell a solve from a lucky
guess. Because an OOLONG question is distributional over the whole window,
the required evidence set is every chunk.

## Getting the data

Neither split is vendored and neither is downloaded by any test. Both are on
the Hugging Face Hub under `oolongbench`, pinned here to the revisions this
adapter was written against. `oolong-real` ships JSONL and needs nothing
extra; `oolong-synth` ships parquet, so reading it needs pyarrow, and the
absence of pyarrow is reported through the same actionable message as the
absence of the data.
"""

from __future__ import annotations

import ast
import importlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from rlm0.benchmarks.context import DEFAULT_CHUNK_CHARS, chunk_context
from rlm0.benchmarks.dataset import (
    BenchmarkDataError,
    DatasetRequirement,
    load_files,
    resolve_root,
)
from rlm0.benchmarks.scoring import (
    Fidelity,
    OfficialItem,
    OfficialResult,
    Scoreboard,
)
from rlm0.benchmarks.suite import BenchmarkManifest, BenchmarkSuite, corpus_spec_for
from rlm0.harness.corpus import Corpus, Sample, TaskFamily

__all__ = [
    "OOLONG_REAL_REVISION",
    "OOLONG_SYNTH_REVISION",
    "OolongReal",
    "OolongSynth",
    "score_real",
    "score_synth",
]

OOLONG_SYNTH_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
"""Pinned revision of oolongbench/oolong-synth, read from the Hub API."""

OOLONG_REAL_REVISION = "6bc9ef04866fcf005c9749b70649be69dd37fffb"
"""Pinned revision of oolongbench/oolong-real, read from the Hub API."""

_COMPARISONS = ("more common", "less common", "same frequency")

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%B %d %Y",
    "%m/%d/%Y",
)

_BOXED_TEXT = re.compile(r"\\boxed\{\\text\{([^}]*)\}\}")
_BOXED = re.compile(r"\\boxed[{]+([^}]*)[}]+")


def _dateutil_available() -> bool:
    try:
        importlib.import_module("dateutil.parser")
    except ImportError:
        return False
    return True


def _parse_date(text: str) -> datetime:
    """Parse a date the way the official harness would, or as close as possible.

    Prefers dateutil so that the metric is reproduced rather than
    approximated. The stdlib fallback covers the formats a model actually
    emits; anything else raises, which the caller turns into a low-confidence
    parse exactly as the official code does.
    """
    try:
        parser = importlib.import_module("dateutil.parser")
    except ImportError:
        pass
    else:
        parsed = parser.parse(text)
        if not isinstance(parsed, datetime):  # pragma: no cover - dateutil contract
            raise ValueError(f"unparseable date: {text!r}")
        return parsed
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"unparseable date: {text!r}")


def synth_parse_answer(answer: str) -> tuple[str, str]:
    """The official OOLONG-synth answer parser, reproduced.

    Kept faithful down to the ordering of the confidence checks. The
    comparison-phrase normalisation is unreachable for candidates shorter than
    twenty characters because it sits after the `vhigh` branch in the same
    `elif` chain; that is how the published scorer behaves, so it is how this
    one behaves.
    """
    parse_confidence = "low"
    if ":" not in answer:
        if len(answer) < 20:
            return answer, parse_confidence
        tokens = answer.split()
        # The official code indexes the last token unconditionally, which
        # raises on whitespace. Returning the empty string keeps the score
        # identical for every input the official code survives.
        return (tokens[-1] if tokens else ""), parse_confidence

    candidate = answer.split(":")[-1].strip()
    candidate = candidate.replace("*", "")
    candidate = candidate.replace("[", "")
    candidate = candidate.replace("]", "")
    parse_confidence = "med"
    if (
        "User:" in answer
        or "Answer:" in answer
        or "Date:" in answer
        or "Label" in answer
    ):
        parse_confidence = "high"
    if len(candidate) < 20:
        parse_confidence = "vhigh"
    elif "more common" in candidate:
        candidate = "more common"
    elif "less common" in candidate:
        candidate = "less common"
    elif "same frequency" in candidate:
        candidate = "same frequency"
    return candidate, parse_confidence


def synth_gold(raw_answer: str) -> str | datetime:
    """Recover the gold value from the dataset's stringified list field.

    The dataset stores answers as the repr of a one-element list, which is
    either a literal or a `datetime.date` call that no literal evaluator will
    accept. The official code branches on the substring "datetime" and so does
    this.
    """
    if "datetime" in raw_answer:
        return datetime.strptime(raw_answer, "[datetime.date(%Y, %m, %d)]")
    value = ast.literal_eval(raw_answer)
    if not isinstance(value, list | tuple) or not value:
        raise BenchmarkDataError(
            f"answer field {raw_answer!r} is not a non-empty list, so this row "
            "has no gold answer and grading it would score nothing"
        )
    return str(value[0])


def score_synth(item: OfficialItem, output: str | None) -> OfficialResult:
    """The official OOLONG-synth scoring function, reproduced."""
    text = output if output is not None else ""
    gold = synth_gold(item.raw_answer)
    trimmed, confidence = synth_parse_answer(text)
    parsed: str = trimmed
    score = 0.0

    if str(trimmed) == str(gold):
        score = 1.0
    elif str(trimmed) in _COMPARISONS:
        if str(trimmed) in str(gold):
            score = 1.0
    elif item.answer_type == "ANSWER_TYPE.NUMERIC":
        try:
            guess = int(trimmed)
            target = int(str(gold))
        except ValueError:
            confidence = "low"
        else:
            parsed = str(guess)
            score = 0.75 ** abs(target - guess)
    elif item.answer_type == "ANSWER_TYPE.DATE":
        try:
            when = _parse_date(trimmed)
        except ValueError:
            confidence = "low"
        else:
            parsed = str(when)
            score = float(when == gold)

    return OfficialResult(
        sample_id=item.sample_id,
        score=score,
        parsed=parsed,
        parse_confidence=confidence,
        answered=output is not None,
    )


def real_parse_gold(answer: str) -> int | str | list[str]:
    """The official OOLONG-real gold parser, reproduced.

    Type is inferred from the string: an integer if it parses as one, a list if
    it contains a comma, a bare string otherwise. The inferred type then
    selects the scoring rule, which is why it is reproduced rather than
    replaced by the dataset's own question_type field.
    """
    try:
        return int(answer)
    except ValueError:
        pass
    if "," in answer:
        return [item.strip() for item in answer.split(",") if item.strip()]
    return answer


def real_parse_response(answer: str) -> tuple[int | str | list[str], str]:
    """The official OOLONG-real response parser, reproduced.

    Returns the raw text with low confidence when no `\\boxed{}` wrapper is
    present, which is exactly what makes an unwrapped answer score zero
    against an integer gold. That is the published behaviour and it is left
    alone; the fix belongs in the prompt, not in the grader.
    """
    match = _BOXED_TEXT.search(answer) or _BOXED.search(answer)
    if not match:
        return answer, "low"
    return real_parse_gold(match.group(1)), "high"


def score_real(item: OfficialItem, output: str | None) -> OfficialResult:
    """The official OOLONG-real scoring function, reproduced."""
    text = output if output is not None else ""
    gold = real_parse_gold(item.raw_answer)
    trimmed, confidence = real_parse_response(text)
    score = 0.0
    if isinstance(gold, int) and isinstance(trimmed, int):
        score = 0.75 ** abs(gold - trimmed)
    elif isinstance(gold, str) and isinstance(trimmed, str):
        score = float(gold.strip().lower() == trimmed.strip().lower())
    elif isinstance(gold, list) and isinstance(trimmed, list):
        overlap = set(gold) & set(trimmed)
        score = len(overlap) / len(gold) if gold else 0.0
    return OfficialResult(
        sample_id=item.sample_id,
        score=score,
        parsed=str(trimmed),
        parse_confidence=confidence,
        answered=output is not None,
    )


def _require(row: Mapping[str, Any], keys: Sequence[str], where: str) -> None:
    missing = [key for key in keys if row.get(key) in (None, "")]
    if missing:
        raise BenchmarkDataError(
            f"{where}: row is missing {missing}. A benchmark row without an "
            "answer makes its grader dead code, so it is refused rather than "
            "scored as zero."
        )


def _harness_answer(gold: str | datetime) -> str:
    """What the harness exact-match grader compares against.

    Dates are written ISO rather than as a datetime repr, because the harness
    grades the literal string a solver produced and no solver emits
    "2020-01-02 00:00:00". The official scorer still compares parsed dates, so
    the two graders can disagree on a date row; that disagreement is visible
    side by side rather than hidden by making one of them lenient.
    """
    if isinstance(gold, datetime):
        return gold.date().isoformat()
    return gold


_SYNTH_DEVIATIONS = (
    "the context is split into identified documents so evidence can be graded; "
    "the official prompt presents one undivided string",
    "the required evidence set is every chunk of the window, because an OOLONG "
    "question is distributional over all of it",
)

_REAL_DEVIATIONS = (
    "the context is split into identified documents so evidence can be graded; "
    "the official prompt presents one undivided string",
    "the required evidence set is every chunk of the episode transcript",
    "the official parser needs a \\boxed{} wrapper; a solver that omits it "
    "scores zero on the official metric regardless of correctness",
)


@dataclass(frozen=True, slots=True)
class OolongSynth:
    """Adapter for OOLONG-synth, the ablatable half of the benchmark."""

    chunk_chars: int = DEFAULT_CHUNK_CHARS
    context_len: int | None = None
    """Keep only rows at this context length, so a figure names its regime."""

    task_groups: tuple[str, ...] = ()
    """Restrict to counting, user or timeline. Empty means all of them."""

    @property
    def name(self) -> str:
        return "oolong-synth"

    def requirement(self, *, split: str) -> DatasetRequirement:
        if split not in {"validation", "test"}:
            raise ValueError(f"oolong-synth has validation and test, not {split!r}")
        root = resolve_root(self.name)
        return DatasetRequirement(
            benchmark=self.name,
            source="hf:oolongbench/oolong-synth",
            revision=OOLONG_SYNTH_REVISION,
            config="default",
            split=split,
            patterns=(f"{split}-*.parquet", f"{split}.jsonl"),
            download=(
                "pip install pyarrow  # oolong-synth ships parquet",
                (
                    "hf download oolongbench/oolong-synth --repo-type dataset"
                    f" --revision {OOLONG_SYNTH_REVISION}"
                    f" --local-dir {root}"
                ),
            ),
            notes=(
                "the full test split is roughly 5,200 rows of long context; "
                "pass limit= or context_len= to run a subset",
            ),
        )

    def answer_instruction(self) -> str:
        return (
            "End your reply with a line of the form 'Answer: <value>'. The "
            "official OOLONG-synth parser reads the text after the final "
            "colon, so nothing may follow that line."
        )

    def load(
        self,
        *,
        split: str = "validation",
        root: Path | None = None,
        limit: int | None = None,
        expected_hash: str | None = None,
    ) -> BenchmarkSuite:
        requirement = self.requirement(split=split)
        where = resolve_root(self.name, root)
        files = load_files(requirement, where, expected_hash=expected_hash)

        samples: list[Sample] = []
        items: dict[str, OfficialItem] = {}
        for index, row in enumerate(files.rows):
            if self.context_len is not None and row.get("context_len") != (
                self.context_len
            ):
                continue
            if self.task_groups and str(row.get("task_group")) not in self.task_groups:
                continue
            _require(
                row,
                ("id", "context_window_text", "question", "answer", "answer_type"),
                f"{self.name} row {index}",
            )
            sample_id = f"oolong-synth-{row['id']}"
            gold = synth_gold(str(row["answer"]))
            documents = chunk_context(
                str(row["context_window_text"]),
                sample_id,
                target_chars=self.chunk_chars,
            )
            answer_type = str(row["answer_type"])
            samples.append(
                Sample(
                    sample_id=sample_id,
                    family=(
                        TaskFamily.AGGREGATE_COUNT
                        if answer_type == "ANSWER_TYPE.NUMERIC"
                        else TaskFamily.AGGREGATE_ARGMAX
                    ),
                    question=str(row["question"]),
                    documents=documents,
                    answer=_harness_answer(gold),
                    required_doc_ids=frozenset(doc.doc_id for doc in documents),
                )
            )
            items[sample_id] = OfficialItem(
                sample_id=sample_id,
                raw_answer=str(row["answer"]),
                answer_type=answer_type,
                extra={
                    "task": str(row.get("task", "")),
                    "task_group": str(row.get("task_group", "")),
                    "dataset": str(row.get("dataset", "")),
                    "context_len": row.get("context_len"),
                },
            )
            if limit is not None and len(samples) >= limit:
                break

        if not samples:
            raise BenchmarkDataError(
                f"{self.name}: no row survived the filters "
                f"(context_len={self.context_len}, task_groups={self.task_groups}). "
                "An empty suite would report an accuracy over nothing."
            )

        exact = _dateutil_available()
        scoreboard = Scoreboard(
            metric="oolong-synth mean score",
            fidelity=Fidelity.REPRODUCES if exact else Fidelity.APPROXIMATES,
            fidelity_note=(
                "reproduces the official scorer, dateutil included"
                if exact
                else "reproduces the official scorer except that dateutil is "
                "absent, so ANSWER_TYPE.DATE answers are parsed by a stdlib "
                "parser over a fixed format list and unusual date spellings "
                "will score zero where the official harness scores them"
            ),
            items=items,
            scorer=score_synth,
        )
        return BenchmarkSuite(
            corpus=Corpus(
                spec=corpus_spec_for(files.content_hash), samples=tuple(samples)
            ),
            scoreboard=scoreboard,
            manifest=BenchmarkManifest(
                benchmark=self.name,
                source=requirement.source,
                revision=requirement.revision,
                config=requirement.config,
                split=split,
                dataset_hash=files.content_hash,
                files=files.relative_paths,
                n_samples=len(samples),
                official_metric=scoreboard.metric,
                fidelity=scoreboard.fidelity,
                fidelity_note=scoreboard.fidelity_note,
                deviations=_SYNTH_DEVIATIONS,
            ),
        )


@dataclass(frozen=True, slots=True)
class OolongReal:
    """Adapter for OOLONG-real, the downstream half over D&D transcripts."""

    chunk_chars: int = DEFAULT_CHUNK_CHARS
    config: str = "dnd"
    question_types: tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return "oolong-real"

    def requirement(self, *, split: str) -> DatasetRequirement:
        if split not in {"validation", "test"}:
            raise ValueError(f"oolong-real has validation and test, not {split!r}")
        if self.config not in {"dnd", "toy_dnd"}:
            raise ValueError(f"oolong-real has dnd and toy_dnd, not {self.config!r}")
        root = resolve_root(self.name)
        return DatasetRequirement(
            benchmark=self.name,
            source="hf:oolongbench/oolong-real",
            revision=OOLONG_REAL_REVISION,
            config=self.config,
            split=split,
            patterns=(f"{self.config}/{split}.jsonl",),
            download=(
                (
                    "hf download oolongbench/oolong-real --repo-type dataset"
                    f" --revision {OOLONG_REAL_REVISION}"
                    f" --local-dir {root}"
                ),
            ),
            notes=("the toy_dnd config is the small one, useful for a smoke run",),
        )

    def answer_instruction(self) -> str:
        return (
            "Give your final answer wrapped as \\boxed{<value>}. The official "
            "OOLONG-real parser reads only that wrapper, and an answer without "
            "it scores zero however correct it is."
        )

    def load(
        self,
        *,
        split: str = "validation",
        root: Path | None = None,
        limit: int | None = None,
        expected_hash: str | None = None,
    ) -> BenchmarkSuite:
        requirement = self.requirement(split=split)
        where = resolve_root(self.name, root)
        files = load_files(requirement, where, expected_hash=expected_hash)

        samples: list[Sample] = []
        items: dict[str, OfficialItem] = {}
        for index, row in enumerate(files.rows):
            if self.question_types and str(row.get("question_type")) not in (
                self.question_types
            ):
                continue
            _require(
                row,
                ("id", "context_window_text", "question", "answer"),
                f"{self.name} row {index}",
            )
            sample_id = f"oolong-real-{row['id']}"
            raw_answer = str(row["answer"])
            gold = real_parse_gold(raw_answer)
            documents = chunk_context(
                str(row["context_window_text"]),
                sample_id,
                target_chars=self.chunk_chars,
            )
            samples.append(
                Sample(
                    sample_id=sample_id,
                    family=(
                        TaskFamily.AGGREGATE_COUNT
                        if isinstance(gold, int)
                        else TaskFamily.AGGREGATE_ARGMAX
                    ),
                    question=str(row["question"]),
                    documents=documents,
                    answer=raw_answer,
                    required_doc_ids=frozenset(doc.doc_id for doc in documents),
                )
            )
            items[sample_id] = OfficialItem(
                sample_id=sample_id,
                raw_answer=raw_answer,
                answer_type=type(gold).__name__,
                extra={
                    "question_type": str(row.get("question_type", "")),
                    "campaign": str(row.get("campaign", "")),
                },
            )
            if limit is not None and len(samples) >= limit:
                break

        if not samples:
            raise BenchmarkDataError(
                f"{self.name}: no row survived the question_type filter "
                f"{self.question_types}; an empty suite reports nothing"
            )

        scoreboard = Scoreboard(
            metric="oolong-real mean score",
            fidelity=Fidelity.REPRODUCES,
            fidelity_note=(
                "reproduces the official scorer: boxed-answer extraction, "
                "0.75**|difference| for integers, case-insensitive exact match "
                "for strings, and overlap over the gold for lists"
            ),
            items=items,
            scorer=score_real,
        )
        return BenchmarkSuite(
            corpus=Corpus(
                spec=corpus_spec_for(files.content_hash), samples=tuple(samples)
            ),
            scoreboard=scoreboard,
            manifest=BenchmarkManifest(
                benchmark=self.name,
                source=requirement.source,
                revision=requirement.revision,
                config=requirement.config,
                split=split,
                dataset_hash=files.content_hash,
                files=files.relative_paths,
                n_samples=len(samples),
                official_metric=scoreboard.metric,
                fidelity=scoreboard.fidelity,
                fidelity_note=scoreboard.fidelity_note,
                deviations=_REAL_DEVIATIONS,
            ),
        )
