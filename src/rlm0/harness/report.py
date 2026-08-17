"""A result table that refuses to render a headline it cannot support.

This is the piece the project actually exists for. `Run` will not construct
without its depth-zero control; the same rule has to hold one level up, or the
control gets run and then quietly left out of the table. So a table with no
depth-zero row does not render. It raises.

The other refusal is comparability. Two rows put beside each other are a
claim that they differ only in the thing being compared, and rows measured on
different corpora, different samples or different grading policies are not
that. Every surveyed cost comparison in this field puts numbers side by side
without recording what varied between them, and one of them reports zero
tokens for every configuration because it reads a usage key nothing sets.

Cost and wall clock sit next to every accuracy figure because the cost claim
for this technique is unfalsifiable otherwise. Unpriced is printed as
unpriced, never as zero. The recursion verdict distribution is printed because
it is the single most useful number this project can publish and nobody else
has it: how often recursion was not attempted, how often it helped, and how
often it was paid for and did not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rlm0.harness.corpus import Regime, TaskFamily
from rlm0.harness.grading import (
    GradingPolicy,
    SampleScore,
    ScoreSummary,
    score_from_dict,
    summarise,
)
from rlm0.run import Run, TokenUsage, Verdict

__all__ = [
    "HarnessVerdict",
    "IntegrityReport",
    "ReportRefusalError",
    "ReportRow",
    "ResultTable",
    "SampleRecord",
    "VerdictCounts",
    "build_report",
    "classify",
    "record_from_dict",
]


class ReportRefusalError(Exception):
    """The table will not render, and the reason is not a formatting problem.

    Raised rather than warned. A warning next to a printed number is read as a
    caveat on a result; the result is the thing that must not exist.
    """


class HarnessVerdict(StrEnum):
    """The run verdict, refined by knowing whether the answers were right.

    A `Run` can only see whether the deeper attempt produced an answer, which
    it says so explicitly. The harness holds the ground truth, so it can tell
    the difference between escalation that fixed the answer and escalation
    that produced a confident wrong one, and that difference is the entire
    argument about whether this technique pays.
    """

    NOT_ATTEMPTED = "not_attempted"
    """Depth zero answered and the run stopped. The cheap path worked."""

    HELPED = "helped"
    """Escalation turned a wrong or missing answer into a correct one."""

    WASTED = "wasted"
    """Escalation was paid for and the correctness of the answer did not move."""

    HARMED = "harmed"
    """Depth zero was right and escalation replaced it with something wrong."""

    UNTESTED = "untested"
    """No control was run, so nothing here can be attributed to recursion."""


def classify(
    run: Run,
    final_score: SampleScore,
    baseline_score: SampleScore | None,
) -> HarnessVerdict:
    """Decide what the escalation bought, using the ground truth.

    Correctness here means supported correctness. An escalation that arrives
    at the right string by reading the wrong documents has not been shown to
    help, and counting it as help is how a technique acquires a reputation it
    did not earn.
    """
    if run.baseline is None or baseline_score is None:
        return HarnessVerdict.UNTESTED
    if len(run.attempts) == 1:
        return HarnessVerdict.NOT_ATTEMPTED
    if final_score.supported and not baseline_score.supported:
        return HarnessVerdict.HELPED
    if baseline_score.supported and not final_score.supported:
        return HarnessVerdict.HARMED
    return HarnessVerdict.WASTED


def _usage_to_dict(usage: TokenUsage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def _usage_from_dict(payload: dict[str, int]) -> TokenUsage:
    return TokenUsage(
        input_tokens=int(payload["input_tokens"]),
        output_tokens=int(payload["output_tokens"]),
        cache_read_tokens=int(payload["cache_read_tokens"]),
        cache_write_tokens=int(payload["cache_write_tokens"]),
    )


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One sample, solved and scored, with the full run accounting kept.

    Persisted verbatim so that every aggregate in the report can be recomputed
    from the raw records, and so a number in a paper can be traced back to the
    trajectory that produced it. Aggregates alone are unauditable, which is
    the state most published evaluations of this technique are in.
    """

    sample_id: str
    family: TaskFamily
    corpus_hash: str
    final_score: SampleScore
    baseline_score: SampleScore | None
    harness_verdict: HarnessVerdict
    run_verdict: Verdict
    answer: str | None
    cited_doc_ids: tuple[str, ...]
    baseline_answer: str | None
    cost_usd: float | None
    wall_clock_s: float
    usage: TokenUsage
    n_calls: int
    n_sub_calls: int
    baseline_cost_usd: float | None
    baseline_wall_clock_s: float
    baseline_usage: TokenUsage
    baseline_n_calls: int
    run: dict[str, Any]
    """The serialised Run, kept whole rather than summarised."""

    @property
    def regime(self) -> Regime:
        return self.family.regime

    def to_dict(self) -> dict[str, Any]:
        baseline = self.baseline_score
        return {
            "sample_id": self.sample_id,
            "family": self.family.value,
            "corpus_hash": self.corpus_hash,
            "final_score": self.final_score.to_dict(),
            "baseline_score": None if baseline is None else baseline.to_dict(),
            "harness_verdict": self.harness_verdict.value,
            "run_verdict": self.run_verdict.value,
            "answer": self.answer,
            "cited_doc_ids": list(self.cited_doc_ids),
            "baseline_answer": self.baseline_answer,
            "cost_usd": self.cost_usd,
            "wall_clock_s": self.wall_clock_s,
            "usage": _usage_to_dict(self.usage),
            "n_calls": self.n_calls,
            "n_sub_calls": self.n_sub_calls,
            "baseline_cost_usd": self.baseline_cost_usd,
            "baseline_wall_clock_s": self.baseline_wall_clock_s,
            "baseline_usage": _usage_to_dict(self.baseline_usage),
            "baseline_n_calls": self.baseline_n_calls,
            "run": self.run,
        }


def record_from_dict(payload: dict[str, Any]) -> SampleRecord:
    baseline = payload["baseline_score"]
    return SampleRecord(
        sample_id=str(payload["sample_id"]),
        family=TaskFamily(payload["family"]),
        corpus_hash=str(payload["corpus_hash"]),
        final_score=score_from_dict(payload["final_score"]),
        baseline_score=None if baseline is None else score_from_dict(baseline),
        harness_verdict=HarnessVerdict(payload["harness_verdict"]),
        run_verdict=Verdict(payload["run_verdict"]),
        answer=payload["answer"],
        cited_doc_ids=tuple(payload["cited_doc_ids"]),
        baseline_answer=payload["baseline_answer"],
        cost_usd=payload["cost_usd"],
        wall_clock_s=float(payload["wall_clock_s"]),
        usage=_usage_from_dict(payload["usage"]),
        n_calls=int(payload["n_calls"]),
        n_sub_calls=int(payload["n_sub_calls"]),
        baseline_cost_usd=payload["baseline_cost_usd"],
        baseline_wall_clock_s=float(payload["baseline_wall_clock_s"]),
        baseline_usage=_usage_from_dict(payload["baseline_usage"]),
        baseline_n_calls=int(payload["baseline_n_calls"]),
        run=dict(payload["run"]),
    )


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One configuration measured on one set of samples.

    The row carries what it was measured on rather than pointing at it,
    because comparability has to be checkable from the table itself. A row
    that cannot say which corpus produced it cannot be placed beside another.
    """

    label: str
    is_depth_zero: bool
    corpus_hash: str
    policy: GradingPolicy
    scores: tuple[SampleScore, ...]
    cost_usd: float | None
    n_unpriced: int
    wall_clock_s: float
    usage: TokenUsage
    n_calls: int

    @property
    def sample_ids(self) -> frozenset[str]:
        return frozenset(score.sample_id for score in self.scores)

    @property
    def summary(self) -> ScoreSummary:
        return summarise(self.scores)

    def summary_for(self, regime: Regime) -> ScoreSummary:
        return summarise([s for s in self.scores if s.regime is regime])

    @property
    def cache_read_ratio(self) -> float | None:
        billed = self.usage.billed_input
        if billed == 0:
            return None
        return self.usage.cache_read_tokens / billed

    def cost_text(self) -> str:
        """Unpriced is a word, never a zero.

        Several surveyed implementations accumulate unpriced calls as zero,
        which is how a cost table comes to read as complete while omitting
        the calls it could not price.
        """
        if self.cost_usd is None:
            return f"unpriced ({self.n_unpriced} of {len(self.scores)})"
        return f"${self.cost_usd:.4f}"


@dataclass(frozen=True, slots=True)
class VerdictCounts:
    """How the run set divided across the recursion verdicts."""

    counts: dict[HarnessVerdict, int]
    raw_counts: dict[Verdict, int]

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def fraction(self, verdict: HarnessVerdict) -> float:
        return self.counts.get(verdict, 0) / self.total if self.total else 0.0

    @property
    def paid_and_did_not_help(self) -> int:
        return self.counts.get(HarnessVerdict.WASTED, 0) + self.counts.get(
            HarnessVerdict.HARMED, 0
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness": {v.value: n for v, n in sorted(self.counts.items())},
            "run": {v.value: n for v, n in sorted(self.raw_counts.items())},
        }


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Signals that a result should not be believed as it stands.

    These are cheap to compute and they catch the two solvers that make a
    benchmark meaningless: the one that answers without reading anything, and
    the one whose answers are right more often than its evidence is.
    """

    n_samples: int
    n_answers_without_calls: int
    n_answers_without_citations: int
    n_correct_but_unsupported: int

    def concerns(self) -> list[str]:
        found: list[str] = []
        if self.n_answers_without_calls:
            found.append(
                f"{self.n_answers_without_calls} answers were produced with no "
                "model call recorded at all; the run accounting says no work "
                "was done"
            )
        if self.n_answers_without_citations:
            found.append(
                f"{self.n_answers_without_citations} answers cited no evidence; "
                "an answer nothing supports is not a solve"
            )
        if self.n_correct_but_unsupported:
            found.append(
                f"{self.n_correct_but_unsupported} answers were correct without "
                "the evidence to support them, which is what luck looks like"
            )
        return found

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_answers_without_calls": self.n_answers_without_calls,
            "n_answers_without_citations": self.n_answers_without_citations,
            "n_correct_but_unsupported": self.n_correct_but_unsupported,
        }


_HEADER = (
    f"{'configuration':<26}{'n':>4}{'answer':>9}{'supported':>11}"
    f"{'ev-F1':>8}{'cost':>22}{'wall':>10}{'cache-read':>12}"
)


def _row_line(label: str, row: ReportRow, summary: ScoreSummary) -> str:
    ratio = row.cache_read_ratio
    ratio_text = "n/a" if ratio is None else f"{ratio:.2f}"
    return (
        f"{label:<26}{summary.n:>4}{summary.answer_accuracy:>9.3f}"
        f"{summary.supported_accuracy:>11.3f}{summary.evidence_f1:>8.3f}"
        f"{row.cost_text():>22}{row.wall_clock_s:>9.1f}s{ratio_text:>12}"
    )


@dataclass(frozen=True, slots=True)
class ResultTable:
    """Rows, verdicts and integrity flags, rendered only if they hold up."""

    title: str
    rows: tuple[ReportRow, ...]
    verdicts: VerdictCounts
    integrity: IntegrityReport

    def check(self) -> None:
        """Raise if this table would be a claim it cannot support."""
        if not self.rows:
            raise ReportRefusalError("a result table with no rows reports nothing")
        for row in self.rows:
            if not row.scores:
                raise ReportRefusalError(
                    f"row {row.label!r} has no scored samples, so its figures "
                    "are averages over nothing"
                )
        if not any(row.is_depth_zero for row in self.rows):
            raise ReportRefusalError(
                "no depth-zero row. A recursive result without the control "
                "that would have answered the same question without recursion "
                "is not a result. Run the control, or do not publish the "
                "table."
            )
        first = self.rows[0]
        for row in self.rows[1:]:
            if row.corpus_hash != first.corpus_hash:
                raise ReportRefusalError(
                    f"rows {first.label!r} and {row.label!r} were measured on "
                    f"different corpora ({first.corpus_hash[:12]} against "
                    f"{row.corpus_hash[:12]}), so putting them side by side "
                    "compares the corpora as much as the systems"
                )
            if row.policy != first.policy:
                raise ReportRefusalError(
                    f"rows {first.label!r} and {row.label!r} were graded under "
                    "different policies, so the difference between them is "
                    "partly the grader"
                )
            if row.sample_ids != first.sample_ids:
                raise ReportRefusalError(
                    f"rows {first.label!r} and {row.label!r} cover different "
                    "samples, so neither number is a measurement of the other's "
                    "task set"
                )

    def render(self) -> str:
        self.check()
        first = self.rows[0]
        lines = [
            self.title,
            f"corpus {first.corpus_hash[:16]}  |  {len(first.scores)} samples  "
            f"|  grading: {first.policy.describe()}",
            "",
            _HEADER,
            "-" * len(_HEADER),
        ]
        for row in self.rows:
            lines.append(_row_line(row.label, row, row.summary))

        for regime in Regime:
            sub = [
                (row, row.summary_for(regime))
                for row in self.rows
                if row.summary_for(regime).n
            ]
            if not sub:
                continue
            lines.append("")
            lines.append(f"{regime.value}:")
            for row, summary in sub:
                lines.append(_row_line(f"  {row.label}", row, summary))

        lines.append("")
        lines.append("recursion verdicts:")
        total = self.verdicts.total
        for verdict in HarnessVerdict:
            count = self.verdicts.counts.get(verdict, 0)
            share = count / total if total else 0.0
            lines.append(f"  {verdict.value:<16}{count:>5}  {share:>6.1%}")
        lines.append(
            f"  paid for and did not help: "
            f"{self.verdicts.paid_and_did_not_help} of {total}"
        )

        concerns = self.integrity.concerns()
        if concerns:
            lines.append("")
            lines.append("integrity:")
            lines.extend(f"  {concern}" for concern in concerns)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        self.check()
        return {
            "title": self.title,
            "corpus_hash": self.rows[0].corpus_hash,
            "policy": self.rows[0].policy.describe(),
            "rows": [
                {
                    "label": row.label,
                    "is_depth_zero": row.is_depth_zero,
                    "cost_usd": row.cost_usd,
                    "n_unpriced": row.n_unpriced,
                    "wall_clock_s": row.wall_clock_s,
                    "cache_read_ratio": row.cache_read_ratio,
                    "n_calls": row.n_calls,
                    "overall": row.summary.to_dict(),
                    "by_regime": {
                        regime.value: row.summary_for(regime).to_dict()
                        for regime in Regime
                    },
                }
                for row in self.rows
            ],
            "verdicts": self.verdicts.to_dict(),
            "integrity": self.integrity.to_dict(),
        }


def _sum_cost(values: Sequence[float | None]) -> tuple[float | None, int]:
    """Total, or None with a count when anything was unpriced."""
    unpriced = sum(1 for value in values if value is None)
    if unpriced:
        return None, unpriced
    return sum(value or 0.0 for value in values), 0


def _sum_usage(usages: Sequence[TokenUsage]) -> TokenUsage:
    total = TokenUsage()
    for usage in usages:
        total = total + usage
    return total


def build_report(
    records: Sequence[SampleRecord],
    *,
    title: str,
    policy: GradingPolicy,
    system_label: str = "rlm0 escalating",
) -> ResultTable:
    """Assemble the table, including the control row taken from the same runs.

    The depth-zero row is not a separate experiment. It is the first attempt
    of every run in the set, graded identically, which is the only way the two
    rows are guaranteed to have seen the same environment, prompt and parsing.
    When any run in the set lacks a control the row simply cannot be built,
    and the table then refuses to render, which is the intended outcome
    rather than an inconvenience to work around.
    """
    if not records:
        raise ReportRefusalError(
            "no records were produced, so there is nothing to report"
        )

    corpus_hashes = {record.corpus_hash for record in records}
    if len(corpus_hashes) != 1:
        raise ReportRefusalError(
            "records span more than one corpus; they cannot be pooled into a "
            "single result"
        )
    corpus_hash = corpus_hashes.pop()

    system_cost, system_unpriced = _sum_cost([r.cost_usd for r in records])
    rows = [
        ReportRow(
            label=system_label,
            is_depth_zero=False,
            corpus_hash=corpus_hash,
            policy=policy,
            scores=tuple(record.final_score for record in records),
            cost_usd=system_cost,
            n_unpriced=system_unpriced,
            wall_clock_s=sum(record.wall_clock_s for record in records),
            usage=_sum_usage([record.usage for record in records]),
            n_calls=sum(record.n_calls for record in records),
        )
    ]

    if all(record.baseline_score is not None for record in records):
        baseline_cost, baseline_unpriced = _sum_cost(
            [r.baseline_cost_usd for r in records]
        )
        rows.insert(
            0,
            ReportRow(
                label="depth 0 (control)",
                is_depth_zero=True,
                corpus_hash=corpus_hash,
                policy=policy,
                scores=tuple(
                    record.baseline_score
                    for record in records
                    if record.baseline_score is not None
                ),
                cost_usd=baseline_cost,
                n_unpriced=baseline_unpriced,
                wall_clock_s=sum(record.baseline_wall_clock_s for record in records),
                usage=_sum_usage([record.baseline_usage for record in records]),
                n_calls=sum(record.baseline_n_calls for record in records),
            ),
        )

    counts: dict[HarnessVerdict, int] = {}
    raw_counts: dict[Verdict, int] = {}
    for record in records:
        counts[record.harness_verdict] = counts.get(record.harness_verdict, 0) + 1
        raw_counts[record.run_verdict] = raw_counts.get(record.run_verdict, 0) + 1

    integrity = IntegrityReport(
        n_samples=len(records),
        n_answers_without_calls=sum(
            1 for r in records if r.answer is not None and r.n_calls == 0
        ),
        n_answers_without_citations=sum(
            1 for r in records if r.answer is not None and not r.cited_doc_ids
        ),
        n_correct_but_unsupported=sum(
            1
            for r in records
            if r.final_score.answer_correct and not r.final_score.supported
        ),
    )

    return ResultTable(
        title=title,
        rows=tuple(rows),
        verdicts=VerdictCounts(counts=counts, raw_counts=raw_counts),
        integrity=integrity,
    )
