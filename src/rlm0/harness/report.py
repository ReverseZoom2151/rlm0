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

Three more refusals were added after a literature sweep, and each of them
removes a way of reporting a result that is not there.

A difference in accuracy is not called meaningful unless it clears a measured
noise floor, because arXiv:2606.20695 found that many published coordination
gains sit inside run-to-run variation that nobody measured. The floor here is
the widest gap between replicates of one configuration against itself, which
is a delta that was produced by changing nothing.

A difference is not called a win unless the two rows spent comparable money,
because arXiv:2606.13003 found automatically designed multi-agent systems
losing to chain-of-thought with self-consistency at ten times the cost, in
tables that reported accuracy without spend. Every row is placed on an
accuracy-cost frontier and a dominated row is printed with the name of what
dominates it.

And accuracy on perturbed twins is reported beside accuracy on the originals
rather than averaged into it, because a gap between them is evidence that the
score came from recognising the instance (arXiv:2605.19999). Averaging is the
one presentation that makes that evidence disappear.
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
    "ContaminationReport",
    "CostBand",
    "Delta",
    "HarnessVerdict",
    "IntegrityReport",
    "NoiseFloor",
    "ParetoPoint",
    "ReportRefusalError",
    "ReportRow",
    "ResultTable",
    "SampleRecord",
    "VerdictCounts",
    "build_comparison",
    "build_report",
    "classify",
    "contamination_report",
    "noise_floor",
    "record_from_dict",
    "system_row",
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

    perturbed: bool = False
    """Whether this sample was the perturbed twin rather than the original."""

    origin_sample_id: str | None = None
    """The original this twin was rewritten from, for pairing at report time."""

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
            "perturbed": self.perturbed,
            "origin_sample_id": self.origin_sample_id,
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
        # Defaulted rather than required, so a records file written before
        # perturbation existed still loads. A resumed run whose earlier half
        # predates the pairing is reported as unperturbed, which is what it
        # was.
        perturbed=bool(payload.get("perturbed", False)),
        origin_sample_id=payload.get("origin_sample_id"),
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


@dataclass(frozen=True, slots=True)
class CostBand:
    """How close two rows have to be in spend before accuracy decides.

    The audit of automatically designed multi-agent systems found them beaten
    by chain-of-thought with self-consistency at under a tenth of the cost,
    and the reason that result was surprising to anyone is that the tables it
    overturned compared accuracy while leaving spend out of the row. So spend
    is in the row here, and a win is only a win inside a band.

    The tolerance is a fraction of the reference spend, not a factor. Twenty
    five percent is wide enough that two systems with genuinely similar cost
    profiles are compared on accuracy rather than on rounding, and narrow
    enough that a system spending double is never quietly called a winner.
    """

    tolerance: float = 0.25

    def __post_init__(self) -> None:
        if self.tolerance < 0.0:
            raise ValueError("tolerance cannot be negative")

    def matched(self, cost_usd: float | None, reference_usd: float | None) -> bool:
        """Whether two spends are close enough to compare on accuracy alone.

        Unpriced is never matched. A row that cannot say what it cost cannot
        be placed in a cost-matched comparison at all, and treating its
        missing cost as equal to anything is the exact failure the `Run` layer
        refuses to make.
        """
        if cost_usd is None or reference_usd is None:
            return False
        if reference_usd == 0.0:
            return cost_usd == 0.0
        return abs(cost_usd - reference_usd) <= self.tolerance * reference_usd

    def describe(self) -> str:
        return f"cost matched to within {self.tolerance:.0%}"


@dataclass(frozen=True, slots=True)
class ParetoPoint:
    """One row placed on the accuracy-cost plane.

    Supported accuracy rather than answer accuracy, because a right answer
    from the wrong documents is not a purchase, and a frontier drawn over
    luck ranks the luckiest system first.
    """

    label: str
    cost_usd: float | None
    accuracy: float
    dominated_by: tuple[str, ...]

    @property
    def priced(self) -> bool:
        return self.cost_usd is not None

    @property
    def on_frontier(self) -> bool:
        return self.priced and not self.dominated_by

    def describe(self) -> str:
        cost = "unpriced" if self.cost_usd is None else f"${self.cost_usd:.4f}"
        if not self.priced:
            return (
                f"{self.label}: {self.accuracy:.3f} at {cost}, so it cannot be "
                "placed on the frontier at all"
            )
        if self.dominated_by:
            return (
                f"{self.label}: {self.accuracy:.3f} at {cost}, dominated by "
                f"{', '.join(self.dominated_by)}"
            )
        return f"{self.label}: {self.accuracy:.3f} at {cost}, on the frontier"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cost_usd": self.cost_usd,
            "accuracy": self.accuracy,
            "dominated_by": list(self.dominated_by),
            "on_frontier": self.on_frontier,
        }


@dataclass(frozen=True, slots=True)
class NoiseFloor:
    """How much the same configuration moves when nothing is changed.

    arXiv:2606.20695 found that many reported coordination gains sit inside
    run-to-run noise, which is only detectable if the noise was measured. So
    the same configuration is run several times over the same samples and the
    spread of those runs becomes the floor that any claimed difference has to
    clear.

    The floor is the widest gap observed between two replicates of the one
    configuration, not a standard deviation and not a confidence interval.
    With three or five replicates a parametric interval is a decoration on a
    sample too small to support it, whereas the widest observed null delta is
    something that actually happened: a difference of that size was produced
    by changing nothing at all.

    `n_flipping_samples` is the paired half of the same measurement. Two
    replicates can land on the same accuracy while disagreeing on which
    samples they solved, and a system whose per-sample answers churn is not
    the same system as one whose answers are stable.
    """

    label: str
    replicate_accuracies: tuple[float, ...]
    n_samples: int
    n_flipping_samples: int

    def __post_init__(self) -> None:
        if len(self.replicate_accuracies) < 2:
            raise ReportRefusalError(
                "a noise floor needs at least two replicates of the same "
                "configuration; one run measures no spread, and a delta "
                "compared against a spread of nothing is an assertion"
            )

    @property
    def floor(self) -> float:
        return max(self.replicate_accuracies) - min(self.replicate_accuracies)

    @property
    def mean_accuracy(self) -> float:
        return sum(self.replicate_accuracies) / len(self.replicate_accuracies)

    def within(self, delta: float) -> bool:
        """Whether a difference of this size is indistinguishable from noise."""
        return abs(delta) <= self.floor

    def describe(self) -> str:
        runs = ", ".join(f"{value:.3f}" for value in self.replicate_accuracies)
        return (
            f"{len(self.replicate_accuracies)} replicates of {self.label} over "
            f"{self.n_samples} samples scored [{runs}], so the floor is "
            f"{self.floor:.3f}; {self.n_flipping_samples} samples changed "
            f"outcome between replicates"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "replicate_accuracies": list(self.replicate_accuracies),
            "floor": self.floor,
            "mean_accuracy": self.mean_accuracy,
            "n_samples": self.n_samples,
            "n_flipping_samples": self.n_flipping_samples,
        }


def noise_floor(
    label: str, replicates: Sequence[Sequence[SampleRecord]]
) -> NoiseFloor:
    """Measure the spread of one configuration over repeated runs.

    Refuses replicates that do not cover the same samples. Paired is the whole
    point: the spread of one configuration over one sample set is a null
    delta, whereas the spread of one configuration over different sample sets
    is a measurement of the sample sets.
    """
    if len(replicates) < 2:
        raise ReportRefusalError(
            "a noise floor needs at least two replicates of the same "
            "configuration"
        )
    id_sets: list[frozenset[str]] = []
    corpus_hashes: set[str] = set()
    for number, replicate in enumerate(replicates, 1):
        ids = [record.sample_id for record in replicate]
        if len(ids) != len(set(ids)):
            raise ReportRefusalError(
                f"replicate {number} contains a sample more than once, so its "
                "accuracy weights that sample more heavily than the others"
            )
        if not replicate:
            raise ReportRefusalError(
                f"replicate {number} has no records, so it measures no run-to-run "
                "variation"
            )
        id_sets.append(frozenset(ids))
        corpus_hashes.update(record.corpus_hash for record in replicate)
    if len(corpus_hashes) != 1:
        raise ReportRefusalError(
            "replicates were measured on different corpora, so their spread "
            "includes a corpus change rather than only run-to-run noise"
        )
    if len({*id_sets}) != 1:
        raise ReportRefusalError(
            "replicates cover different samples, so their spread measures the "
            "sample sets rather than the run-to-run noise of the configuration"
        )
    accuracies = tuple(
        summarise([record.final_score for record in rep]).supported_accuracy
        for rep in replicates
    )
    by_sample: dict[str, set[bool]] = {}
    for rep in replicates:
        for record in rep:
            by_sample.setdefault(record.sample_id, set()).add(
                record.final_score.supported
            )
    return NoiseFloor(
        label=label,
        replicate_accuracies=accuracies,
        n_samples=len(by_sample),
        n_flipping_samples=sum(1 for states in by_sample.values() if len(states) > 1),
    )


@dataclass(frozen=True, slots=True)
class ContaminationReport:
    """Accuracy on the original samples against accuracy on perturbed twins.

    A large gap is evidence that the score on the originals came partly from
    recognising them rather than from reading them. It is surfaced as its own
    section and as an integrity concern, because the one place it must not
    appear is folded into a single pooled accuracy figure, which is what
    reporting the mean of the two halves would do.
    """

    n_original: int
    n_perturbed: int
    n_paired: int
    original_accuracy: float
    perturbed_accuracy: float
    tolerated_gap: float = 0.1

    @property
    def measured(self) -> bool:
        return self.n_paired > 0

    @property
    def gap(self) -> float:
        return self.original_accuracy - self.perturbed_accuracy

    @property
    def memorisation_suspected(self) -> bool:
        return self.measured and self.gap > self.tolerated_gap

    def describe(self) -> str:
        if not self.measured:
            return (
                "no perturbed twins in this corpus, so contamination "
                "resistance was not measured here"
            )
        return (
            f"{self.n_paired} paired samples: {self.original_accuracy:.3f} on "
            f"the originals against {self.perturbed_accuracy:.3f} on the "
            f"perturbed twins, a gap of {self.gap:+.3f}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_original": self.n_original,
            "n_perturbed": self.n_perturbed,
            "n_paired": self.n_paired,
            "original_accuracy": self.original_accuracy,
            "perturbed_accuracy": self.perturbed_accuracy,
            "gap": self.gap,
            "memorisation_suspected": self.memorisation_suspected,
        }


def contamination_report(
    records: Sequence[SampleRecord], *, tolerated_gap: float = 0.1
) -> ContaminationReport:
    """Split the record set into originals and twins and score each half.

    Only paired samples count. A twin whose original was not run, or an
    original whose twin was not run, would put a task in one half that is not
    in the other, and the gap would then include the difference between two
    task sets.
    """
    originals = {r.sample_id: r for r in records if not r.perturbed}
    twins = {
        r.origin_sample_id: r
        for r in records
        if r.perturbed and r.origin_sample_id is not None
    }
    paired = sorted(set(originals) & set(twins))
    return ContaminationReport(
        n_original=len(originals),
        n_perturbed=len(twins),
        n_paired=len(paired),
        original_accuracy=summarise(
            [originals[sid].final_score for sid in paired]
        ).supported_accuracy,
        perturbed_accuracy=summarise(
            [twins[sid].final_score for sid in paired]
        ).supported_accuracy,
        tolerated_gap=tolerated_gap,
    )


@dataclass(frozen=True, slots=True)
class Delta:
    """One row's difference from another, with the floor it has to clear."""

    challenger: str
    incumbent: str
    delta: float
    floor: float | None
    cost_multiple: float | None
    cost_matched: bool

    @property
    def inside_noise(self) -> bool:
        return self.floor is None or abs(self.delta) <= self.floor

    @property
    def meaningful(self) -> bool:
        return not self.inside_noise and self.cost_matched

    def describe(self) -> str:
        head = (
            f"{self.challenger} against {self.incumbent}: "
            f"{self.delta:+.3f} supported accuracy"
        )
        if self.floor is None:
            return f"{head}, with no replicate spread measured, so not a claim"
        if abs(self.delta) <= self.floor:
            return (
                f"{head}, inside the {self.floor:.3f} noise floor, so not a "
                "difference"
            )
        if not self.cost_matched:
            multiple = (
                "unpriced"
                if self.cost_multiple is None
                else f"{self.cost_multiple:.2f}x the spend"
            )
            return f"{head}, clears the noise floor but at {multiple}"
        return f"{head}, clears the {self.floor:.3f} noise floor at matched cost"

    def to_dict(self) -> dict[str, Any]:
        return {
            "challenger": self.challenger,
            "incumbent": self.incumbent,
            "delta": self.delta,
            "floor": self.floor,
            "cost_multiple": self.cost_multiple,
            "cost_matched": self.cost_matched,
            "meaningful": self.meaningful,
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

    noise: NoiseFloor | None = None
    """The measured run-to-run spread, when the suite was replicated."""

    contamination: ContaminationReport | None = None
    """Original against perturbed, when the corpus carried twins."""

    band: CostBand = CostBand()
    """How close spends have to be before accuracy is allowed to decide."""

    reference_label: str | None = None
    """Which row the cost-matched comparison is held against.

    Defaults to the depth-zero control, which is the row every other row in
    this project exists to be compared with.
    """

    def check(self) -> None:
        """Raise if this table would be a claim it cannot support."""
        if not self.rows:
            raise ReportRefusalError("a result table with no rows reports nothing")
        labels = [row.label for row in self.rows]
        if len(labels) != len(set(labels)):
            raise ReportRefusalError(
                "two rows share a label, so a comparison cannot identify which "
                "configuration its delta refers to"
            )
        for row in self.rows:
            if not row.scores:
                raise ReportRefusalError(
                    f"row {row.label!r} has no scored samples, so its figures "
                    "are averages over nothing"
                )
            sample_ids = [score.sample_id for score in row.scores]
            if len(sample_ids) != len(set(sample_ids)):
                raise ReportRefusalError(
                    f"row {row.label!r} scores a sample more than once, so its "
                    "aggregate gives some tasks more weight than others"
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

    def row(self, label: str) -> ReportRow:
        for row in self.rows:
            if row.label == label:
                return row
        known = ", ".join(row.label for row in self.rows)
        raise ReportRefusalError(f"no row labelled {label!r}; the table has {known}")

    @property
    def reference(self) -> ReportRow:
        """The row every other row is priced against.

        The depth-zero control by default, because that is the row this
        project guarantees exists.
        """
        if self.reference_label is not None:
            return self.row(self.reference_label)
        for row in self.rows:
            if row.is_depth_zero:
                return row
        raise ReportRefusalError(
            "no depth-zero row to hold cost against, so there is no reference "
            "for a cost-matched comparison"
        )

    def pareto(self) -> tuple[ParetoPoint, ...]:
        """Every row placed on the accuracy-cost plane, with what dominates it.

        A row is dominated when another row is at least as accurate for no
        more money and strictly better on one of the two. That is the whole
        content of the claim that a system winning on accuracy while spending
        ten times as much is not winning: it will have a dominator, printed by
        name next to it.
        """
        points: list[tuple[str, float | None, float]] = [
            (row.label, row.cost_usd, row.summary.supported_accuracy)
            for row in self.rows
        ]
        out: list[ParetoPoint] = []
        for label, cost, accuracy in points:
            dominators: list[str] = []
            if cost is not None:
                for other_label, other_cost, other_accuracy in points:
                    if other_label == label or other_cost is None:
                        continue
                    no_worse = other_cost <= cost and other_accuracy >= accuracy
                    strictly = other_cost < cost or other_accuracy > accuracy
                    if no_worse and strictly:
                        dominators.append(other_label)
            out.append(
                ParetoPoint(
                    label=label,
                    cost_usd=cost,
                    accuracy=accuracy,
                    dominated_by=tuple(dominators),
                )
            )
        return tuple(out)

    def cost_multiple(self, label: str) -> float | None:
        """What this row spends per dollar the reference row spent."""
        reference = self.reference.cost_usd
        cost = self.row(label).cost_usd
        if reference is None or cost is None or reference == 0.0:
            return None
        return cost / reference

    def delta(self, challenger: str, incumbent: str | None = None) -> Delta:
        """One row's supported-accuracy difference from another.

        Carries the noise floor and the cost multiple with it, so that the
        difference cannot be quoted without the two things that decide whether
        it means anything.
        """
        incumbent_row = self.reference if incumbent is None else self.row(incumbent)
        challenger_row = self.row(challenger)
        return Delta(
            challenger=challenger_row.label,
            incumbent=incumbent_row.label,
            delta=(
                challenger_row.summary.supported_accuracy
                - incumbent_row.summary.supported_accuracy
            ),
            floor=None if self.noise is None else self.noise.floor,
            cost_multiple=self.cost_multiple(challenger_row.label),
            cost_matched=self.band.matched(
                challenger_row.cost_usd, incumbent_row.cost_usd
            ),
        )

    def deltas(self) -> tuple[Delta, ...]:
        reference = self.reference
        return tuple(
            self.delta(row.label)
            for row in self.rows
            if row.label != reference.label
        )

    def claim(self, challenger: str, incumbent: str | None = None) -> str:
        """State that one row beat another, or refuse to.

        The refusals are the point, and they are the same three the literature
        this project is arguing with skipped. A difference inside the measured
        run-to-run spread is not a difference (arXiv:2606.20695). A difference
        bought with several times the spend is not a win (arXiv:2606.13003).
        And a difference whose spread was never measured at all is an
        assertion, which is the state most of these results are published in.
        """
        found = self.delta(challenger, incumbent)
        if found.delta <= 0.0:
            raise ReportRefusalError(
                f"{found.challenger} does not lead {found.incumbent} on "
                f"supported accuracy ({found.delta:+.3f}), so there is no "
                "claim to make"
            )
        if found.floor is None:
            raise ReportRefusalError(
                f"{found.challenger} leads {found.incumbent} by "
                f"{found.delta:+.3f}, but this configuration was run once. "
                "Replicate it and compare the delta against the spread, or do "
                "not call it a result."
            )
        if found.inside_noise:
            raise ReportRefusalError(
                f"{found.challenger} leads {found.incumbent} by "
                f"{found.delta:+.3f}, which is inside the {found.floor:.3f} "
                "noise floor measured by rerunning one configuration against "
                "itself. A difference this size was produced by changing "
                "nothing."
            )
        if not found.cost_matched:
            multiple = (
                "an unpriced amount"
                if found.cost_multiple is None
                else f"{found.cost_multiple:.2f}x"
            )
            raise ReportRefusalError(
                f"{found.challenger} leads {found.incumbent} by "
                f"{found.delta:+.3f} while spending {multiple} of its cost, "
                f"which is outside the band ({self.band.describe()}). Accuracy "
                "bought with more money is not an advantage of the method."
            )
        point = {p.label: p for p in self.pareto()}[found.challenger]
        if point.dominated_by:
            raise ReportRefusalError(
                f"{found.challenger} leads {found.incumbent} but is dominated "
                f"on the accuracy-cost frontier by "
                f"{', '.join(point.dominated_by)}"
            )
        return (
            f"{found.challenger} beats {found.incumbent} by {found.delta:+.3f} "
            f"supported accuracy, outside a {found.floor:.3f} noise floor and "
            f"inside a {self.band.tolerance:.0%} cost band"
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

        lines.extend(self._cost_lines())
        lines.extend(self._noise_lines())
        lines.extend(self._contamination_lines())

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
        contamination = self.contamination
        if contamination is not None and contamination.memorisation_suspected:
            concerns.append(
                f"supported accuracy falls {contamination.gap:.3f} from the "
                "original samples to their perturbed twins, which is what "
                "memorisation of the originals looks like"
            )
        if concerns:
            lines.append("")
            lines.append("integrity:")
            lines.extend(f"  {concern}" for concern in concerns)
        return "\n".join(lines)

    def _cost_lines(self) -> list[str]:
        """The comparison that decides whether a lead was bought or earned."""
        try:
            reference = self.reference
        except ReportRefusalError:  # pragma: no cover - check() already refused
            return []
        points = {point.label: point for point in self.pareto()}
        lines = [
            "",
            f"accuracy at matched cost ({self.band.describe()}, held against "
            f"{reference.label}):",
        ]
        for row in self.rows:
            multiple = self.cost_multiple(row.label)
            spend = "unpriced" if multiple is None else f"{multiple:.2f}x"
            if row.label == reference.label:
                verdict = "reference"
            elif self.band.matched(row.cost_usd, reference.cost_usd):
                verdict = "matched"
            else:
                verdict = "not matched, so accuracy alone decides nothing"
            lines.append(
                f"  {row.label:<26}{row.summary.supported_accuracy:>9.3f}"
                f"{spend:>10}  {verdict}"
            )
        lines.append("")
        lines.append("accuracy-cost frontier:")
        lines.extend(f"  {points[row.label].describe()}" for row in self.rows)
        return lines

    def _noise_lines(self) -> list[str]:
        if self.noise is None:
            return [
                "",
                "noise floor: not measured. This configuration was run once, "
                "so no difference below is called meaningful.",
            ]
        lines = ["", f"noise floor: {self.noise.describe()}"]
        for found in self.deltas():
            lines.append(f"  {found.describe()}")
        return lines

    def _contamination_lines(self) -> list[str]:
        if self.contamination is None or not self.contamination.measured:
            return []
        report = self.contamination
        lines = ["", f"contamination check: {report.describe()}"]
        if report.memorisation_suspected:
            lines.append(
                f"  the gap exceeds the tolerated {report.tolerated_gap:.3f}, "
                "so the score on the originals is partly recognition"
            )
        return lines

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
            "cost_band": self.band.tolerance,
            "pareto": [point.to_dict() for point in self.pareto()],
            "deltas": [found.to_dict() for found in self.deltas()],
            "noise": None if self.noise is None else self.noise.to_dict(),
            "contamination": (
                None if self.contamination is None else self.contamination.to_dict()
            ),
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


def _is_depth_zero(records: Sequence[SampleRecord]) -> bool:
    """Whether every run in this set stopped at depth zero.

    Read off the serialised runs rather than taken on the label's word. A row
    that satisfies the control requirement has to have been a control.
    """
    for record in records:
        attempts = record.run.get("attempts", [])
        if not attempts:
            return False
        if any(int(attempt.get("max_depth", 0)) != 0 for attempt in attempts):
            return False
        if record.n_sub_calls:
            return False
    return True


def system_row(
    label: str, records: Sequence[SampleRecord], *, policy: GradingPolicy
) -> ReportRow:
    """One row for one system, from that system's own records.

    Used to put several systems on one table. The row carries the corpus hash
    and the grading policy it was measured under, so `ResultTable.check` can
    refuse a comparison across corpora or graders rather than printing one.
    """
    if not records:
        raise ReportRefusalError(f"row {label!r} has no records to summarise")
    hashes = {record.corpus_hash for record in records}
    if len(hashes) != 1:
        raise ReportRefusalError(
            f"row {label!r} pools records from more than one corpus"
        )
    cost, unpriced = _sum_cost([record.cost_usd for record in records])
    return ReportRow(
        label=label,
        is_depth_zero=_is_depth_zero(records),
        corpus_hash=hashes.pop(),
        policy=policy,
        scores=tuple(record.final_score for record in records),
        cost_usd=cost,
        n_unpriced=unpriced,
        wall_clock_s=sum(record.wall_clock_s for record in records),
        usage=_sum_usage([record.usage for record in records]),
        n_calls=sum(record.n_calls for record in records),
    )


def build_comparison(
    systems: Sequence[tuple[str, Sequence[SampleRecord]]],
    *,
    title: str,
    policy: GradingPolicy,
    noise: NoiseFloor | None = None,
    band: CostBand | None = None,
    reference_label: str | None = None,
    tolerated_gap: float = 0.1,
) -> ResultTable:
    """One table over several systems measured on the same corpus.

    This is the table the multi-agent audit says has to exist. A depth-zero
    control next to a recursive system answers whether the recursion helped;
    it does not answer whether either of them beats chain-of-thought with
    self-consistency, and until that row is on the same table the recursive
    result is compared only against itself.

    Verdict counts and integrity flags are taken from the first system, which
    is the one the table is about. Pooling them across systems would produce a
    recursion verdict distribution for a set of runs most of which never had
    the option to recurse.
    """
    if not systems:
        raise ReportRefusalError("a comparison with no systems reports nothing")
    rows = tuple(
        system_row(label, records, policy=policy) for label, records in systems
    )
    head = list(systems[0][1])
    counts: dict[HarnessVerdict, int] = {}
    raw_counts: dict[Verdict, int] = {}
    for record in head:
        counts[record.harness_verdict] = counts.get(record.harness_verdict, 0) + 1
        raw_counts[record.run_verdict] = raw_counts.get(record.run_verdict, 0) + 1
    return ResultTable(
        title=title,
        rows=rows,
        verdicts=VerdictCounts(counts=counts, raw_counts=raw_counts),
        integrity=_integrity_of(head),
        noise=noise,
        contamination=contamination_report(head, tolerated_gap=tolerated_gap),
        band=band or CostBand(),
        reference_label=reference_label,
    )


def _integrity_of(records: Sequence[SampleRecord]) -> IntegrityReport:
    return IntegrityReport(
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


def build_report(
    records: Sequence[SampleRecord],
    *,
    title: str,
    policy: GradingPolicy,
    system_label: str = "rlm0 escalating",
    noise: NoiseFloor | None = None,
    band: CostBand | None = None,
    tolerated_gap: float = 0.1,
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

    return ResultTable(
        title=title,
        rows=tuple(rows),
        verdicts=VerdictCounts(counts=counts, raw_counts=raw_counts),
        integrity=_integrity_of(records),
        noise=noise,
        contamination=contamination_report(records, tolerated_gap=tolerated_gap),
        band=band or CostBand(),
    )
