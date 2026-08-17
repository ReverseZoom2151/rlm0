"""S-NIAH, which is where this project has to lose gracefully.

The aggregation claim needs OOLONG. This is the other half, and it matters
more than a win would. The independent reproduction of the RLM paper measured
the base model at 100 percent and depth two at 70 percent on single needle in
a haystack, so a retrieval benchmark is where recursion is known to be
actively harmful. rlm0 runs depth zero first and escalates only on failure, so
on S-NIAH the expected result is not that rlm0 is clever: it is that depth
zero answers, the escalation never fires, and the harness verdict column fills
with NOT_ATTEMPTED. A depth policy that cannot decline to recurse is a policy
that will pay for the 30 point drop, and this adapter is how that gets
measured instead of asserted.

## The official metric

RULER scores NIAH tasks with `string_match_all`: for each sample, the
fraction of the gold strings that appear as a case-insensitive substring of
the prediction, averaged over samples and multiplied by 100. Predictions are
first passed through RULER's `postprocess_pred`, which strips whitespace and
replaces control characters with newlines. Both are reproduced here exactly,
including the final `round(score, 2)`.

Note what `string_match_all` forgives and what it does not. It is substring
containment, so a prediction that repeats the whole context scores 100. That
is a property of the official metric and it is left alone, but it is also
precisely why this adapter reports the evidence-aware harness score beside it:
the harness requires the answer to match after normalisation and requires the
document holding the needle to have been cited, and a context-echoing solver
fails both.

## Getting the data

RULER generates its own data rather than distributing it, which makes a
pinned, hashable artifact awkward. Two sources are accepted.

The first is the `simonjegou/ruler` mirror on the Hugging Face Hub, pinned to
the revision below. It carries context, question and answers as separate
columns, which is the cleanest fit, and it is what the download command names.
It is a third-party mirror of RULER's generator output rather than an
NVIDIA-published artifact, and the manifest says so.

The second is whatever `RULER/scripts/data/prepare.py` wrote, which is a JSONL
of `{index, input, outputs, length}`. There the prompt arrives as one string,
so the question is taken to be the final blank-line-separated block of it.
That heuristic is checked rather than trusted: a row whose question comes out
empty, or whose question contains the needle, is refused.

## Evidence

The required document set is derived from the text, not from a field: the
chunks that actually contain a gold string. A row where no chunk contains the
needle is refused loudly, because a needle benchmark whose needle is not in
its haystack has no gradeable evidence and would quietly turn the evidence
grader into decoration.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm0.benchmarks.context import DEFAULT_CHUNK_CHARS, chunk_context, locate
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
    "RULER_MIRROR_REVISION",
    "SINGLE_NEEDLE_TASKS",
    "RulerNiah",
    "postprocess_pred",
    "score_niah",
    "string_match_all",
]

RULER_MIRROR_REVISION = "24adceac8a0e6532936e8d721cd9e9084d2e4686"
"""Pinned revision of simonjegou/ruler, read from the Hub API."""

SINGLE_NEEDLE_TASKS = ("niah_single_1", "niah_single_2", "niah_single_3")
"""S-NIAH proper. The multi-needle and multi-value variants are a different
question and pooling them would blur the retrieval result this exists to get.
"""

_CONTROL = re.compile(r"[\x00-\x1f]")


def postprocess_pred(prediction: str) -> str:
    """RULER's prediction postprocessing, reproduced."""
    text = prediction.strip()
    return _CONTROL.sub("\n", text).strip()


def _match_fraction(prediction: str, refs: Sequence[str]) -> float:
    lowered = prediction.lower()
    if not refs:
        return 0.0
    return sum(1.0 for ref in refs if ref.lower() in lowered) / len(refs)


def string_match_all(
    predictions: Sequence[str], references: Sequence[Sequence[str]]
) -> float:
    """RULER's `string_match_all`, reproduced including the scaling and rounding.

    Exposed as a free function so it can be checked against the published
    implementation directly, rather than only through the adapter.
    """
    if not predictions:
        return 0.0
    total = sum(
        _match_fraction(pred, refs)
        for pred, refs in zip(predictions, references, strict=True)
    )
    return round(total / len(predictions) * 100, 2)


def score_niah(item: OfficialItem, output: str | None) -> OfficialResult:
    """One sample's contribution to `string_match_all`, before the scaling.

    Kept on the same 0 to 1 scale as every other per-sample score in this
    package; the aggregate multiplies by 100 and rounds, which is where the
    official presentation is restored.
    """
    refs = [str(ref) for ref in item.extra.get("refs", ())]
    text = postprocess_pred(output if output is not None else "")
    return OfficialResult(
        sample_id=item.sample_id,
        score=_match_fraction(text, refs),
        parsed=text[:200],
        parse_confidence="high" if output else "low",
        answered=output is not None,
    )


def _aggregate_niah(results: Sequence[OfficialResult]) -> float:
    """The official aggregate: mean, times 100, rounded to two places."""
    if not results:
        return 0.0
    return round(sum(r.score for r in results) / len(results) * 100, 2)


def _refs_of(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("answer", row.get("outputs"))
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list | tuple):
        return [str(item) for item in raw]
    return []


def _split_native(text: str) -> tuple[str, str]:
    """Separate a RULER prompt into haystack and question.

    The generator emits the task instruction, the haystack and then the
    question, separated by blank lines, so the final block is the question.
    The caller validates the result, because a heuristic that is wrong and
    unchecked would silently move part of the haystack into the question and
    make the task trivially easy.
    """
    blocks = [block for block in text.split("\n\n") if block.strip()]
    if len(blocks) < 2:
        return text, ""
    return "\n\n".join(blocks[:-1]), blocks[-1].strip()


@dataclass(frozen=True, slots=True)
class RulerNiah:
    """Adapter for RULER's single-needle retrieval tasks."""

    context_length: str = "4096"
    """Which of the mirror's length configs to read. Named, not measured."""

    tasks: tuple[str, ...] = SINGLE_NEEDLE_TASKS
    chunk_chars: int = DEFAULT_CHUNK_CHARS

    @property
    def name(self) -> str:
        return "ruler-s-niah"

    def requirement(self, *, split: str) -> DatasetRequirement:
        if split != "test":
            raise ValueError(f"the ruler mirror publishes only test, not {split!r}")
        root = resolve_root(self.name)
        return DatasetRequirement(
            benchmark=self.name,
            source="hf:simonjegou/ruler (mirror of NVIDIA/RULER generator output)",
            revision=RULER_MIRROR_REVISION,
            config=self.context_length,
            split=split,
            patterns=(
                f"{self.context_length}/test-*.parquet",
                f"{self.context_length}/*.jsonl",
                "*/validation.jsonl",
            ),
            download=(
                "pip install pyarrow  # the mirror ships parquet",
                (
                    "hf download simonjegou/ruler --repo-type dataset"
                    f" --revision {RULER_MIRROR_REVISION}"
                    f" --local-dir {root}"
                ),
            ),
            notes=(
                "this is a third-party mirror of RULER's generator output, not "
                "an NVIDIA-published artifact; to use RULER itself instead, run "
                "its scripts/data/prepare.py and put the resulting "
                "<task>/validation.jsonl files under the same directory",
                "context lengths published by the mirror are 4096, 8192 and 16384",
            ),
        )

    def answer_instruction(self) -> str:
        return (
            "Answer with the requested value alone. RULER scores by checking "
            "whether each gold string occurs in your reply, so extra commentary "
            "is harmless but a paraphrase of the value is not."
        )

    def load(
        self,
        *,
        split: str = "test",
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
            task = str(row.get("task", "")) or self.tasks[0]
            if self.tasks and task not in self.tasks:
                continue
            refs = _refs_of(row)
            if not refs:
                raise BenchmarkDataError(
                    f"{self.name} row {index} carries no gold string. A needle "
                    "row with no needle makes the grader dead code."
                )
            context, question = self._context_and_question(row, index)
            sample_id = f"ruler-{task}-{self.context_length}-{row.get('index', index)}"
            documents = chunk_context(
                context, sample_id, target_chars=self.chunk_chars
            )
            required = frozenset[str]().union(
                *(locate(documents, ref) for ref in refs)
            )
            if not required:
                raise BenchmarkDataError(
                    f"{self.name} row {index}: none of the gold strings "
                    f"{refs} occur in the context, so no evidence set can be "
                    "derived and the evidence grader would score nothing"
                )
            samples.append(
                Sample(
                    sample_id=sample_id,
                    family=TaskFamily.NEEDLE,
                    question=question,
                    documents=documents,
                    answer=refs[0],
                    required_doc_ids=required,
                )
            )
            items[sample_id] = OfficialItem(
                sample_id=sample_id,
                raw_answer=refs[0],
                answer_type="string",
                extra={"refs": tuple(refs), "task": task},
            )
            if limit is not None and len(samples) >= limit:
                break

        if not samples:
            raise BenchmarkDataError(
                f"{self.name}: no row matched tasks {self.tasks}; an empty suite "
                "would report a retrieval accuracy over nothing"
            )

        scoreboard = Scoreboard(
            metric="RULER string_match_all (percent)",
            fidelity=Fidelity.REPRODUCES,
            fidelity_note=(
                "reproduces RULER's postprocess_pred and string_match_all, "
                "including the multiplication by 100 and round to two places; "
                "the data is a pinned third-party mirror of RULER's generator "
                "output rather than an NVIDIA-published artifact"
            ),
            items=items,
            scorer=score_niah,
            aggregate=_aggregate_niah,
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
                deviations=(
                    "the haystack is split into identified documents so evidence "
                    "can be graded; RULER presents one undivided prompt",
                    "the required evidence set is derived from the text, as the "
                    "chunks that contain a gold string",
                    "rows in RULER's native format have their question taken to "
                    "be the final blank-line-separated block of the prompt",
                ),
            ),
        )

    def _context_and_question(
        self, row: Mapping[str, Any], index: int
    ) -> tuple[str, str]:
        """Recover haystack and question from either accepted row shape."""
        context = row.get("context")
        question = row.get("question")
        if isinstance(context, str) and isinstance(question, str):
            prefix = row.get("answer_prefix")
            if isinstance(prefix, str) and prefix.strip():
                question = f"{question.strip()}\n{prefix.strip()}"
            return context, question.strip()

        native = row.get("input")
        if not isinstance(native, str) or not native.strip():
            raise BenchmarkDataError(
                f"{self.name} row {index} has neither context/question columns "
                "nor a RULER-native input field"
            )
        context, question = _split_native(native)
        if not question:
            raise BenchmarkDataError(
                f"{self.name} row {index}: the prompt has no trailing question "
                "block, so haystack and question cannot be separated"
            )
        for ref in _refs_of(row):
            if ref.lower() in question.lower():
                raise BenchmarkDataError(
                    f"{self.name} row {index}: the derived question contains the "
                    f"gold string {ref!r}, so the split put part of the haystack "
                    "in the question and the task would be trivial"
                )
        return context, question
