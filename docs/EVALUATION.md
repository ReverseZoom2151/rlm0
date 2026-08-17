# Evaluation protocol

Committed in advance, so that it cannot be adjusted after the numbers are in.
No benchmark result has been published from this repository and none will be
without a depth-zero row beside it.

The point of writing this down before running anything is narrow. A protocol
chosen after seeing results is not a protocol, it is a description of which
comparison happened to look best. Everything here is checkable against
[`src/rlm0/harness/`](../src/rlm0/harness/), and where the code does not yet
implement a part of it, that is said.

## What is being measured

Not whether recursive language models work. That question is badly posed,
because the answer varies by task shape: recursion is worth a great deal on
dense aggregation and is measurably harmful on retrieval and on anything that
fits the window. The independent reproduction
([arXiv:2603.02615](https://arxiv.org/abs/2603.02615)) shows both effects in one
experiment, with the base model at 100 percent on S-NIAH lookup against 85 at
depth one and 70 at depth two, and 0 percent on OOLONG aggregation against 42.1
at depth one.

So the question is per task and per regime: on this shape of work, did paying
for recursion buy anything, and how much did it cost. The corpus is split along
exactly that axis, and every headline figure is reported broken down by it. A
pooled accuracy number across both regimes cannot say which regime a system is
in, which is the only interesting question here.

## The depth-zero control, and why it is not optional

Every run carries the attempt that would have answered the same question without
recursion. It is produced by the same function, `RLM._drive`, with sub-calls
switched off, using the same prompt assembly, the same parser, the same
environment and the same observation formatting. `Run` will not construct
without it. `ResultTable.check` will not render a table without a depth-zero row
and raises rather than warning, because a warning next to a printed number reads
as a caveat on a result and the result is the thing that must not exist.

The control is not a separate experiment run afterwards. It is the first attempt
of every run in the set, graded identically, which is the only way the two rows
are guaranteed to have seen the same everything. A control run as a second job,
with its own prompt and its own harness invocation, measures the difference
between two scaffolds and reports it as the value of recursion.

The reason this is worth insisting on is that it is missing from essentially
every published evaluation of the technique, and the reason it is missing is
that producing it was extra work. Two benchmark papers, AMA-Bench
([arXiv:2602.22769](https://arxiv.org/abs/2602.22769)) and ContextBench
([arXiv:2602.05892](https://arxiv.org/abs/2602.05892)), report scaffolds
underperforming a plain long-context baseline, which is what makes insisting on
the control look like judgement rather than modesty.

## Why the control cannot be depth zero alone

Depth zero is the right control for isolating recursion. It is the wrong control
for the question of whether the whole apparatus is worth building.

*The Illusion of Multi-Agent Advantage*
([arXiv:2606.13003](https://arxiv.org/abs/2606.13003)) found that automatically
designed multi-agent systems rarely beat a strong single-agent baseline once
cost is accounted for, and that chain-of-thought with self-consistency beat them
at under a tenth of the cost. A scaffold that beats its own ablation and loses
to CoT with self-consistency has not demonstrated anything worth deploying.

So the table has at least three rows, and CoT with self-consistency is included
at matched cost. Matched cost, not matched call count: self-consistency at k
samples is cheap per sample and the comparison is only fair if it is given the
budget the recursive system actually spent.

Two non-recursive mechanisms also have to be beaten on their own ground, because
both claim the territory this project is arguing about. Addressable Recall
Compaction ([arXiv:2607.25066](https://arxiv.org/abs/2607.25066)) reports 99.40
percent on NIAH against 88.12 for the best baseline it tested, at the lowest
serving latency of everything measured, and it should win on lookup. VISTA
([arXiv:2606.30005](https://arxiv.org/abs/2606.30005)) reports 50.7 on
LOCA-Bench against 22.7 for ReAct, which is a prompt-level interface change
beating a runtime by 28 points. Losing to either is a result and should be
published as one.

The baselines have to be well elicited. METR's guidance on capability
elicitation is the counterweight to everything above: a weak baseline understates
the model, and that is a methodological error in the same family as a missing
control, just pointing the other way. A depth-zero attempt given a worse prompt,
fewer iterations or a tighter output cap than the recursive attempt is a
strawman, and the section-level prompt invariant in
[`prompt.py`](../src/rlm0/prompt.py) exists to make that specific cheat visible:
exactly three named sections may differ between the variants, and the test suite
asserts the rest are byte-identical.

## Grading: two axes, never blended

A system that returns the right answer having read the wrong documents got
lucky. On a long-context benchmark that is not rare, because the answer space is
small and the distractors are near misses of it. Reporting only answer accuracy
makes luck indistinguishable from a solve.

[`grading.py`](../src/rlm0/harness/grading.py) scores the answer and the evidence
separately and derives the composite rather than averaging.
`SampleScore.score` is zero unless the answer is correct, and is then weighted by
the F1 of the cited document set against the set the corpus says had to be read.
A right answer citing wrong documents therefore scores zero: precision is zero,
recall is zero, F1 is zero, and the gate is multiplicative.

`supported` is the stricter flag and is what the harness verdicts are computed
from. It requires a correct answer, complete evidence recall by default, and
precision at or above 0.5, the last of which stops a system from citing the
entire context to guarantee recall for free.

Answer matching is exact after normalisation, and the normalisation folds only
case, surrounding quotes, a trailing full stop, thousands separators, and the
integer-written-as-float case. Nothing else. The answers here are opaque
identifiers and integers, so there is nothing for a fuzzy match to usefully
forgive, and partial credit for a near miss would reward precisely the failure
mode the corpus is built to punish.

The required evidence set is not the same as the set containing the answer
string. On a supersession chain you have to read the stale records to know they
are stale. On an aggregation you have to read every member of the cohort to know
the count. Grading against that set is what separates a solve from a lucky
guess.

`ScoreSummary` reports `answer_accuracy` alongside `supported_accuracy` and
exposes the difference as `luck_gap`. The first number is the one everyone else
publishes, and it is included only so the gap is visible.

## The corpus

Deterministic from a seed, adversarial by construction, and self-verifying.
[`corpus.py`](../src/rlm0/harness/corpus.py) is 900 lines of closing shortcuts:
opaque identifiers so nothing is guessable from a name, distractors whose
closeness to the answer is a single tunable parameter, supersession chains
shuffled so recency of position never tracks recency of record, decoy cohorts
including a one-character twin of the target, and facts scattered within
documents rather than placed at the top.

Two details are worth naming. First, the generator re-derives every answer from
the text it emitted, by regex, the way a solver would, and raises
`GroundTruthError` on any disagreement. Re-deriving from the objects the text
was rendered from would only prove the renderer was called; a rendering bug
would become a silently wrong label. Second, the argmax family is constructed so
the subject holding the single largest record is not the subject with the
largest total, so a system that answers by finding the biggest number rather
than by summing lands on a specific recorded distractor.

Corpus identity is the SHA-256 of the canonical serialisation, not of the seed,
because a change to the generator that keeps the seed produces different text.
The runner refuses to resume across a corpus change and the report refuses to
pool records from more than one corpus.

The corpus generated here is not a substitute for the published benchmarks. It
exists so that the deterministic layers can be exercised end to end and so that
distractor difficulty is a knob rather than an accident. OOLONG
([arXiv:2511.02817](https://arxiv.org/abs/2511.02817)) remains the load-bearing
external benchmark for aggregation, with ATLAS, HELMET, MRCR v2, AA-LCR and
BrowseComp-Plus as the surrounding set. The aggregation claim specifically has
to be demonstrated on AGGBench
([arXiv:2602.01355](https://arxiv.org/abs/2602.01355)), where DFA is the
baseline to beat.

## What the report refuses

[`report.py`](../src/rlm0/harness/report.py) raises `ReportRefusalError` rather
than rendering, in five cases.

No rows, or a row with no scored samples, because an average over nothing is not
a figure. No depth-zero row, for the reason above. Rows measured on different
corpora, because putting them side by side compares the corpora as much as the
systems. Rows graded under different policies, because the difference between
them is then partly the grader. Rows covering different sample sets, because
neither number is a measurement of the other's task set.

Every one of these is a comparison that appears in published work in this area.
The refusals are not defensive programming; they are the specific errors
observed while reading twenty implementations.

Cost is printed as `unpriced (n of m)` when any call could not be priced, never
as zero. Several surveyed implementations accumulate unpriced calls as zero,
which is how a cost table comes to read as complete while omitting exactly the
calls it could not account for.

`IntegrityReport` flags three things cheaply: answers produced with no model
call recorded, answers citing no evidence at all, and answers correct without
the evidence to support them. The first two catch a solver that is not doing the
work. The third is what luck looks like in aggregate.

## Verdicts

`Run.recursion_verdict` can only see whether the deeper attempt produced an
answer where the control did not, because the run does not hold the ground
truth. The harness does, so `classify` in `report.py` refines it into four
outcomes: `NOT_ATTEMPTED` when depth zero answered and the run stopped, `HELPED`
when escalation turned an unsupported result into a supported one, `HARMED` when
depth zero was supported and escalation replaced it with something that was not,
and `WASTED` when escalation was paid for and supported correctness did not
move.

Correctness there means supported correctness. An escalation that arrives at the
right string by reading the wrong documents has not been shown to help, and
counting it as help is how a technique acquires a reputation it did not earn.

The verdict distribution is the number this project can publish that nobody else
currently has: how often recursion was not needed, how often it helped, and how
often it was paid for and did not. The report prints
`paid for and did not help` as an explicit count.

## Noise floors

*How Much Coordination Gain Is Real?*
([arXiv:2606.20695](https://arxiv.org/abs/2606.20695)) found that many reported
coordination gains fall inside run-to-run noise. Any depth-one over depth-zero
delta reported from this repository has to be measured against a paired noise
floor, and paired is the operative word: the two attempts share a corpus, a
sample, a model and a seed, so the correct baseline for the difference is the
distribution of differences between two independent depth-zero runs on the same
samples, not the variance of either arm on its own.

Concretely, the floor is established by running the depth-zero configuration
twice over the same corpus with different sampling seeds, scoring both, and
taking the per-sample paired difference. A depth-one gain smaller than that
distribution's spread is not a result. The reproduction paper's own caveat is
relevant here too: 20 samples per condition in a single run gives large effect
sizes with small n, and this project should not repeat that shape while citing
it.

The harness preserves the records needed for this calculation, but it does not
yet provide a noise-floor command or report field. It remains a release gate
for a public delta.

## What is reported alongside accuracy

A 2026 evaluation that reports accuracy alone is not taken seriously, and the
canonical citation for why is *AI Agents That Matter*
([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)). The Holistic Agent
Leaderboard ([arXiv:2510.11977](https://arxiv.org/abs/2510.11977), ICLR 2026) is
the current standard: cost-controlled by default, with accuracy-cost Pareto
frontiers, analysed across models by scaffolds by benchmarks.

So every row of a published table from this repository carries, at minimum:

- Answer accuracy and supported accuracy, with the gap between them.
- Evidence precision, recall and F1.
- Total USD, or `unpriced` with a count, never zero.
- Wall clock.
- Call count, split by root and sub.
- Cache read ratio over billed input. Zero across a fan-out is a bug rather than
  a fact about the workload, and it is the single best diagnostic available for
  whether prefix caching is doing anything.
- The same figures broken down by regime.
- The verdict distribution.

Plus, outside the table: the corpus hash, the seed, the grading policy, the
models, the budget summary, the harness version, and the exact invocation. All
of those are in `manifest.json` already, written by
[`runner.py`](../src/rlm0/harness/runner.py).

Every aggregate must be recomputable from the raw per-sample records, which is
why the full serialised `Run` is persisted per sample rather than a summary of
it. Aggregates alone are unauditable, which is the state most published
evaluations of this technique are in.

The right frame for an eventual result is an honest ablation of an RLM scaffold.
HAL already provides cost-controlled scaffold comparisons. A HAL adapter is
future work, not a capability claimed by this repository.

## Reporting a negative result

If depth zero wins on the retrieval regime, that is the expected outcome and is
published as such. If depth one does not beat depth zero on aggregation either,
that is the more interesting result and is published without softening. If the
verdict distribution shows escalation mostly `WASTED`, that is the headline.

The commitment is not that recursion will be shown to work. It is that whatever
is shown will have the control beside it.
