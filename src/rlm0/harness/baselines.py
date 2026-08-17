"""The systems rlm0 has to beat before any of its numbers mean anything.

A depth-zero control answers one question: did the recursion in this run buy
anything over the same scaffold without recursion. It does not answer the
question a reader actually has, which is whether the whole apparatus beats
something simpler and cheaper. The audit of six automatic multi-agent design
frameworks (arXiv:2606.13003) found that it usually does not: chain-of-thought
with self-consistency beat every one of them, frequently at higher accuracy for
under a tenth of the cost, and the automated searches kept rediscovering
CoT-SC under more elaborate names. So a depth-zero row alone is not a baseline,
it is a self-comparison, and a table containing only self-comparisons cannot
say the system is worth running.

The opposite error is just as easy and is the one METR's elicitation guidance
warns about: a baseline written to lose understates whatever is being measured,
and a win over it is worth nothing. Every prompt in this module is therefore
written to make the baseline win if it can. Concretely:

- The task prompt names the four traps the corpus is built from, in plain
  language, before the model sees a single document. Naming a distractor
  construction is the single highest-value elicitation move available on an
  adversarial corpus, and it costs nothing.
- The answer format is exact and terminal, so a correct answer is never lost
  to formatting. Exact-match grading punishes a good answer wrapped in a
  sentence, and losing a solve to the parser is a measurement error, not a
  capability finding.
- Abstention is available. UNKNOWN is not scored as a wrong answer, so the
  baseline is never forced to guess, and a self-consistency vote is never
  polluted by a guess it did not want to make.
- The evidence requirement is stated with its scoring rule attached, including
  that citing everything scores as citing nothing, because a baseline that
  loses on evidence precision for want of being told the rule is a strawman.
- Only one prompt section differs between the direct and the reasoning
  baselines, which is the same discipline `rlm0.prompt` applies to the
  depth-zero control: if the baselines differed in tone or in task framing,
  the measured gap between them would include the prompt.

Every solver here reports what it spent. A baseline that under-reports its own
cost inverts the accuracy-cost comparison it exists to inform, which would be
worse than having no baseline at all, so cost and calls come from the provider
response and nothing is estimated or dropped.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rlm0.harness.corpus import Document, SolverTask
from rlm0.harness.grading import normalise
from rlm0.harness.runner import Attempted, SolverResult
from rlm0.ports import LMClient
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run

__all__ = [
    "ABSTENTION",
    "REASONING_SECTIONS",
    "SELF_CONSISTENCY_PATHS",
    "AlwaysWrongSolver",
    "BaselineConfig",
    "BaselineReply",
    "CoTSelfConsistencySolver",
    "DirectSolver",
    "RetrievalSolver",
    "baseline_prompt_sections",
    "build_baseline_prompt",
    "parse_baseline_reply",
    "rank_documents",
]

ABSTENTION = "UNKNOWN"
"""What the model writes instead of guessing.

Read as no answer rather than as a wrong one. A baseline forced to guess is a
baseline whose accuracy figure is partly a measurement of how the prompt
handles uncertainty, and on a corpus of near-miss distractors a guess lands on
a distractor often enough to matter.
"""


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """What a baseline was configured with, carried into its run record.

    `context_char_limit` is a stated precondition rather than a truncation
    point. A baseline that silently drops half the documents and then reports
    an accuracy figure has measured a different task, so the direct solver
    refuses instead, and the refusal appears in the run as an attempt that
    made no calls and cost nothing.
    """

    model: str
    max_tokens: int = 1024
    context_char_limit: int = 400_000

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if self.context_char_limit < 1:
            raise ValueError("context_char_limit must be at least 1")


_ROLE = """\
You are answering one question about a set of documents. Every document you
need is reproduced in full below. Nothing you remember from anywhere else
applies to this material, and an answer that does not come from the text below
is wrong even when it looks right."""


_HAZARDS = """\
Four things in this material are built to catch a careless reader. Knowing
them in advance is worth more than any amount of care applied afterwards.

1. Identifiers are opaque and near misses are deliberate. SUBJ-4K2P9Q and
   SUBJ-4K2P9R are different subjects, and there will be records for both.
   Compare identifiers character by character before accepting a match.
2. The same shape of statement is made under several different attribute
   names, such as retention window or clearance margin. Only the exact
   attribute the question names counts. The others are there to be rejected.
3. A record can be superseded by a later one, stated as "Record A supersedes
   record B". When supersessions are present the answer is the record nothing
   supersedes. Position in the context says nothing about which that is: the
   current record is deliberately not the last one you will read.
4. Cohort membership is declared, on a line reading "cohort: CHT-XXXXXX". A
   document that looks like a member but declares a different cohort is not a
   member, and there is a cohort whose identifier differs by one character."""


_EVIDENCE = """\
Cite the documents you actually used, by the identifier on their DOCUMENT
line. Cite every document the answer rests on, which includes documents that
only establish that a record was superseded, and every member of a cohort you
had to count or total. Do not cite documents you did not use: your citations
are scored for precision as well as for recall, so citing everything scores
the same as citing nothing."""


_FORMAT = f"""\
End your reply with exactly two lines, in this order, with nothing after them:

ANSWER: <the answer and nothing else>
EVIDENCE: DOC-XXXXXXXX, DOC-YYYYYYYY

Write a numeric answer as digits alone, with no units, no thousands
separators and no sentence around it. Write an identifier answer exactly as it
appears in the text. If the documents genuinely do not settle the question,
write ANSWER: {ABSTENTION} rather than guessing; a guess is scored as a wrong
answer and an abstention is not."""


_DIRECT_REASONING = """\
Answer directly. Do not write out your working: go straight to the two final
lines. You may look back over the documents as many times as you need before
you commit, but what you emit is the answer, not the route to it."""


_STEPWISE_REASONING = """\
Work the problem in writing before you commit, in this order.

1. Restate exactly which identifier and which attribute the question names.
2. List every document that mentions that identifier, with its DOCUMENT id,
   and for each one say whether the identifier is an exact match or a near
   miss. Reject the near misses explicitly rather than silently.
3. Apply whatever resolution the question needs: follow the supersession
   chain to the record nothing supersedes, follow the routing chain hop by
   hop, or enumerate the declared cohort members one line at a time.
4. Only then write the two final lines.

Enumerating and then eliminating is slower than pattern matching and it is the
only method that survives this material. Write the intermediate list out; a
count or a chain held in your head is where the mistakes happen."""


SELF_CONSISTENCY_PATHS: tuple[str, ...] = (
    "Work forwards: start from the identifier in the question and follow it "
    "through the documents in the order they appear.",
    "Work backwards: find every candidate answer first, then eliminate the "
    "candidates whose identifier or attribute does not match exactly.",
    "Work by elimination: go document by document and decide for each one "
    "whether it is relevant at all before you use any of them.",
    "Work by transcription: copy out every line that mentions the identifier "
    "the question names, then reason only over what you copied.",
    "Work by contradiction: assume the most obvious answer is a planted "
    "distractor, and try to prove it is one before you accept it.",
)
"""Distinct reasoning framings, one per self-consistency sample.

`LMClient.complete` exposes no temperature, deliberately, so diversity here
cannot come from sampling parameters. It comes from asking for a genuinely
different route to the same answer, which is a defensible substitute: the
point of self-consistency is that independent routes agreeing is evidence, and
routes that differ in method are more independent than routes that differ only
in a random seed. The framings live in the final user message, so the system
prompt and the documents stay byte identical across samples and the provider
prefix cache still applies.
"""


REASONING_SECTIONS: frozenset[str] = frozenset({"reasoning"})
"""The only section allowed to differ between the baselines.

Same rule as `rlm0.prompt.SUB_CALL_SECTIONS`, for the same reason: if the
direct and the reasoning baselines differed anywhere else, the measured gap
between them would include the difference in framing rather than isolating the
reasoning instruction.
"""


@dataclass(frozen=True, slots=True)
class _Section:
    name: str
    body: str


def baseline_prompt_sections(*, stepwise: bool) -> tuple[_Section, ...]:
    """The baseline system prompt as named sections, in order.

    Exposed so a test can assert the invariant directly rather than by
    diffing two blobs and hoping.
    """
    return (
        _Section("role", _ROLE),
        _Section("hazards", _HAZARDS),
        _Section("evidence", _EVIDENCE),
        _Section(
            "reasoning", _STEPWISE_REASONING if stepwise else _DIRECT_REASONING
        ),
        _Section("format", _FORMAT),
    )


def build_baseline_prompt(*, stepwise: bool) -> str:
    """The system prompt for one baseline."""
    return "\n\n".join(
        section.body for section in baseline_prompt_sections(stepwise=stepwise)
    )


_ANSWER_RE = re.compile(r"^[ \t]*ANSWER[ \t]*:(?P<value>.*)$", re.MULTILINE)
_EVIDENCE_RE = re.compile(r"^[ \t]*EVIDENCE[ \t]*:(?P<value>.*)$", re.MULTILINE)
_DOC_RE = re.compile(r"DOC-[A-Z0-9]{4,10}")


@dataclass(frozen=True, slots=True)
class BaselineReply:
    """One model reply, reduced to what the harness can score."""

    answer: str | None
    cited_doc_ids: tuple[str, ...]
    abstained: bool = False

    @property
    def usable(self) -> bool:
        return self.answer is not None


def parse_baseline_reply(text: str, known_doc_ids: Sequence[str]) -> BaselineReply:
    """Read the two terminal lines out of a reply.

    The last ANSWER line wins, because a model that reasons in writing often
    narrates the format before it uses it. Cited identifiers are filtered to
    documents that were actually in the context: an identifier the model
    invented is not evidence, and the runner refuses a result that cites one
    outright, so dropping it is the only way the row exists at all. Dropping
    it is generous to the baseline, since a hallucinated citation would
    otherwise cost it evidence precision, and generous is the correct
    direction for a baseline to err in.
    """
    known = set(known_doc_ids)
    answers = list(_ANSWER_RE.finditer(text))
    cited: tuple[str, ...] = ()
    evidence = list(_EVIDENCE_RE.finditer(text))
    if evidence:
        found = _DOC_RE.findall(evidence[-1].group("value"))
        seen: list[str] = []
        for doc_id in found:
            if doc_id in known and doc_id not in seen:
                seen.append(doc_id)
        cited = tuple(seen)
    if not answers:
        return BaselineReply(answer=None, cited_doc_ids=cited)
    value = answers[-1].group("value").strip()
    if not value:
        return BaselineReply(answer=None, cited_doc_ids=cited)
    if value.strip().strip("*`").upper() == ABSTENTION:
        return BaselineReply(answer=None, cited_doc_ids=cited, abstained=True)
    return BaselineReply(answer=value, cited_doc_ids=cited)


def _render(documents: Sequence[Document]) -> str:
    return "\n\n".join(doc.render() for doc in documents)


def _user_message(task: SolverTask, documents: Sequence[Document], suffix: str) -> str:
    """The documents first, then the question, then the framing.

    The question comes after the documents so that everything above it is
    identical across the samples of one self-consistency vote, which is what
    lets the provider prefix cache serve the repeat and keeps the reported
    cost of the vote honest about what it actually charged.
    """
    parts = [_render(documents), f"QUESTION: {task.question}"]
    if suffix:
        parts.append(suffix)
    return "\n\n".join(parts)


@dataclass(frozen=True, slots=True)
class _Call:
    """One completion and the record that accounts for it."""

    reply: BaselineReply
    record: CallRecord


def _issue(
    client: LMClient,
    config: BaselineConfig,
    *,
    system: str,
    user: str,
    known_doc_ids: Sequence[str],
    cache_prefix: bool,
) -> _Call:
    """Make one call and account for it from the provider's own numbers.

    Nothing here derives usage or cost from string length. The whole point of
    putting a baseline on the same table as the system under test is that the
    two cost figures are commensurable, and an estimate on one side of the
    comparison makes both sides meaningless.
    """
    response = client.complete(
        system=system,
        messages=[{"role": "user", "content": user}],
        model=config.model,
        max_tokens=config.max_tokens,
        cache_prefix=cache_prefix,
    )
    record = CallRecord(
        role=Role.ROOT,
        depth=0,
        model=response.model,
        usage=response.usage,
        wall_clock_s=response.wall_clock_s,
        cost_usd=response.cost_usd,
        cached_prefix=response.cached_prefix,
    )
    return _Call(
        reply=parse_baseline_reply(response.text, known_doc_ids), record=record
    )


def _depth_zero_result(
    task: SolverTask,
    *,
    answer: str | None,
    cited: Sequence[str],
    calls: Sequence[CallRecord],
    budget_summary: str,
    detail: str = "",
    outcome: Outcome | None = None,
) -> SolverResult:
    """Wrap a baseline's work in a `Run` whose control is the run itself.

    A baseline has no deeper attempt to compare against, so its depth-zero
    attempt is both the control and the result. That is not a formality: it is
    what makes the row comparable to the control row of an escalating run,
    which is the comparison the whole table is built around.
    """
    if outcome is None:
        outcome = (
            Outcome.ANSWERED if answer is not None else Outcome.ITERATIONS_EXHAUSTED
        )
    attempt = Attempt(
        max_depth=0,
        outcome=outcome,
        calls=tuple(calls),
        wall_clock_s=sum(call.wall_clock_s for call in calls),
        answer=answer if outcome.produced_answer else None,
        detail=detail,
    )
    run = Run(
        task=task.question, attempts=(attempt,), budget_summary=budget_summary
    )
    attempted = Attempted(answer=run.answer, cited_doc_ids=tuple(cited))
    return SolverResult(run=run, final=attempted, baseline=attempted)


@dataclass
class DirectSolver:
    """One call, no reasoning instruction, where the context fits.

    The cheapest thing that could possibly work, and on retrieval-shaped items
    the published evidence says it is also the thing that wins. It is here to
    stop a recursive system claiming credit for solving a lookup.

    Elicitation: it gets the identical task prompt, hazard list, evidence rule
    and answer format as every other baseline. The only section that differs
    is the one telling it not to write out its working, which is the variable
    under test.
    """

    client: LMClient
    config: BaselineConfig
    label: str = "direct single-shot"

    def describe(self) -> str:
        return (
            f"{self.label} [model={self.config.model}, "
            f"max_tokens={self.config.max_tokens}, "
            f"fits<={self.config.context_char_limit} chars]"
        )

    def _budget(self) -> str:
        return (
            f"baseline ceiling: 1 call, {self.config.max_tokens} output tokens, "
            f"no escalation"
        )

    def solve(self, task: SolverTask) -> SolverResult:
        context = task.context()
        if len(context) > self.config.context_char_limit:
            # Refusing beats truncating. A truncated context is a different
            # task, and an accuracy figure measured on it would be reported
            # as if it were measured on this one.
            return _depth_zero_result(
                task,
                answer=None,
                cited=(),
                calls=(),
                budget_summary=self._budget(),
                outcome=Outcome.ERRORED,
                detail=(
                    f"context of {len(context)} chars exceeds the stated fit "
                    f"limit of {self.config.context_char_limit}; this baseline "
                    "does not truncate"
                ),
            )
        call = _issue(
            self.client,
            self.config,
            system=build_baseline_prompt(stepwise=False),
            user=_user_message(task, task.documents, ""),
            known_doc_ids=task.doc_ids,
            cache_prefix=False,
        )
        return _depth_zero_result(
            task,
            answer=call.reply.answer,
            cited=call.reply.cited_doc_ids,
            calls=(call.record,),
            budget_summary=self._budget(),
            detail="abstained" if call.reply.abstained else "",
        )


@dataclass(frozen=True, slots=True)
class _Vote:
    """One sample of the self-consistency vote."""

    answer: str
    cited_doc_ids: tuple[str, ...]


def _modal(votes: Sequence[_Vote]) -> tuple[_Vote, ...]:
    """The largest group of votes that agree, after normalisation.

    Grouping uses the grader's own normalisation, so two samples that would be
    scored identically are counted as agreeing. Anything looser would let a
    vote be won by formatting; anything stricter would split a majority across
    "7" and "7.".

    Ties go to the group whose first vote came earliest. That is arbitrary,
    and it is chosen because it is arbitrary in a way that is stable and
    independent of the answers: breaking a tie by picking the numerically
    smaller answer, or by asking the model again, would put a thumb on the
    scale in a direction that correlates with the corpus.
    """
    order: list[str] = []
    groups: dict[str, list[_Vote]] = {}
    for vote in votes:
        key = normalise(vote.answer)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(vote)
    if not order:
        return ()
    best = max(order, key=lambda key: (len(groups[key]), -order.index(key)))
    return tuple(groups[best])


def _majority_evidence(group: Sequence[_Vote]) -> tuple[str, ...]:
    """Documents cited by a strict majority of the winning group.

    Not the union, which would buy evidence recall for free by citing anything
    any sample happened to mention, and would cost precision that the grader
    charges for. Not the intersection either, which one careless sample can
    empty. A majority of the samples that already agreed on the answer is the
    set the vote actually supports.
    """
    tally: dict[str, int] = {}
    for vote in group:
        for doc_id in set(vote.cited_doc_ids):
            tally[doc_id] = tally.get(doc_id, 0) + 1
    selected = sorted(doc for doc, count in tally.items() if count * 2 > len(group))
    if selected:
        return tuple(selected)
    return tuple(group[0].cited_doc_ids) if group else ()


@dataclass
class CoTSelfConsistencySolver:
    """Chain of thought, sampled several ways, answering by majority.

    This is the baseline the multi-agent audit found beating six automatic
    framework searches, usually at under a tenth of their cost. If rlm0 does
    not beat this on the accuracy-cost Pareto then rlm0 beats nothing, so it
    is built to win.

    What makes it strong, in order of how much it is worth:

    - The stepwise instruction asks for enumeration and explicit elimination
      of near misses, which is the method this corpus is built to require.
      A generic "think step by step" leaves the model to discover that, and
      the discovery is most of the task.
    - The samples take deliberately different routes rather than differing by
      a random seed, because `LMClient` exposes no temperature and because
      agreement between different methods is stronger evidence than agreement
      between reruns of one.
    - Abstentions do not vote. A sample that says UNKNOWN is not evidence for
      any answer, and letting it vote for a guess would make the vote worse
      than its own members.
    - Every sample shares the system prompt and the documents byte for byte,
      so the provider prefix cache serves the repeats. That lowers the real
      cost of the vote, which matters because the comparison being made here
      is on cost as much as on accuracy.

    Cost is reported per call from the provider response, so the vote costs
    what `samples` calls cost and says so.
    """

    client: LMClient
    config: BaselineConfig
    samples: int = 5
    label: str = "cot self-consistency"
    paths: tuple[str, ...] = field(default_factory=lambda: SELF_CONSISTENCY_PATHS)

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("a self-consistency vote needs at least one sample")
        if not self.paths:
            raise ValueError("at least one reasoning path is required")

    def describe(self) -> str:
        return (
            f"{self.label}@{self.samples} [model={self.config.model}, "
            f"max_tokens={self.config.max_tokens}, modal answer, "
            f"majority evidence]"
        )

    def _budget(self) -> str:
        return (
            f"baseline ceiling: {self.samples} calls, "
            f"{self.config.max_tokens} output tokens each, no escalation"
        )

    def solve(self, task: SolverTask) -> SolverResult:
        system = build_baseline_prompt(stepwise=True)
        calls: list[CallRecord] = []
        votes: list[_Vote] = []
        n_abstained = 0
        for index in range(self.samples):
            path = self.paths[index % len(self.paths)]
            call = _issue(
                self.client,
                self.config,
                system=system,
                user=_user_message(task, task.documents, path),
                known_doc_ids=task.doc_ids,
                cache_prefix=True,
            )
            calls.append(call.record)
            if call.reply.abstained:
                n_abstained += 1
            if call.reply.answer is not None:
                votes.append(
                    _Vote(
                        answer=call.reply.answer,
                        cited_doc_ids=call.reply.cited_doc_ids,
                    )
                )
        group = _modal(votes)
        answer = group[0].answer if group else None
        detail = (
            f"{len(group)} of {self.samples} samples agreed, "
            f"{n_abstained} abstained"
        )
        return _depth_zero_result(
            task,
            answer=answer,
            cited=_majority_evidence(group),
            calls=calls,
            budget_summary=self._budget(),
            detail=detail,
        )


@dataclass
class AlwaysWrongSolver:
    """The floor of the table, and the only row whose cost is honestly zero.

    Every benchmark needs a row that shows what nothing buys. This one makes
    no model call, spends nothing, and answers with a token no sample in the
    corpus can have as its answer, since answers are positive integers or
    opaque identifiers. Its accuracy is the number a grader has to give zero,
    and its integrity flags are the ones a grader has to raise, so it doubles
    as a check that the harness still fails what it should fail.
    """

    label: str = "always wrong (floor)"
    answer: str = "-1"

    def describe(self) -> str:
        return f"{self.label} [no model calls]"

    def solve(self, task: SolverTask) -> SolverResult:
        return _depth_zero_result(
            task,
            answer=self.answer,
            cited=(),
            calls=(),
            budget_summary="baseline ceiling: 0 calls, nothing is spent",
            detail="constant answer, no context read",
        )


def _terms(text: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[A-Za-z0-9-]+", text)]


def rank_documents(
    question: str, documents: Sequence[Document]
) -> tuple[tuple[float, Document], ...]:
    """Score documents against the question, best first.

    Saturating term frequency weighted by inverse document frequency, which is
    the load bearing half of BM25 and is what makes this a real retrieval
    baseline rather than a substring search. On this corpus the question's
    opaque identifier is rare by construction, so it dominates the score and
    the ranking is genuinely competitive on the retrieval families. It is
    equally genuinely hopeless on aggregation, where the answer needs every
    member of a cohort and a top-k cut cannot return them all. That contrast
    is the reason this row is on the table.

    Ties break on document order, which is already shuffled by the generator,
    so the ranking never inherits position as a signal.
    """
    tokenised = [(_terms(doc.render()), doc) for doc in documents]
    n_docs = len(tokenised) or 1
    doc_freq: dict[str, int] = {}
    for terms, _ in tokenised:
        for term in set(terms):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    query = [term for term in _terms(question) if len(term) > 2]
    scored: list[tuple[float, Document]] = []
    for terms, doc in tokenised:
        counts: dict[str, int] = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        total = 0.0
        for term in query:
            tf = counts.get(term, 0)
            if not tf:
                continue
            df = doc_freq.get(term, 0)
            # Rarity in this document set, floored at zero so a term present
            # everywhere contributes nothing rather than going negative.
            idf = max(0.0, ((n_docs - df + 0.5) / (df + 0.5)))
            total += (tf / (tf + 1.2)) * idf
        scored.append((total, doc))
    order = {doc.doc_id: i for i, doc in enumerate(documents)}
    scored.sort(key=lambda pair: (-pair[0], order[pair[1].doc_id]))
    return tuple(scored)


@dataclass
class RetrievalSolver:
    """Retrieve first, then answer over what was retrieved.

    The non-recursive competitor. It is the architecture most systems in this
    space are actually replacing, and leaving it off the table would let a
    recursive result take credit for beating an approach nobody compared it
    against.

    Elicitation: the same prompt as the reasoning baseline, plus one honest
    sentence saying the documents were pre-selected and may be incomplete, so
    the model abstains rather than confabulating from a bad retrieval. Without
    that sentence the row measures the retriever's recall and calls it the
    model's accuracy.

    Retrieval itself makes no model call and costs nothing, which is reported
    as exactly that. The one call it does make is over `top_k` documents
    rather than the whole context, which is where the cost advantage of this
    architecture comes from and why it belongs in a cost-matched comparison.
    """

    client: LMClient
    config: BaselineConfig
    top_k: int = 5
    label: str = "retrieval then answer"

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")

    def describe(self) -> str:
        return (
            f"{self.label}@k={self.top_k} [model={self.config.model}, "
            f"max_tokens={self.config.max_tokens}, tf-idf ranking]"
        )

    def _budget(self) -> str:
        return (
            f"baseline ceiling: 1 call over {self.top_k} retrieved documents, "
            f"{self.config.max_tokens} output tokens, no escalation"
        )

    def solve(self, task: SolverTask) -> SolverResult:
        ranked = rank_documents(task.question, task.documents)
        selected = [doc for _, doc in ranked[: self.top_k]]
        note = (
            f"These {len(selected)} documents were selected from "
            f"{len(task.documents)} by a lexical retriever, so the set may be "
            f"incomplete. If what you were given does not settle the question, "
            f"answer {ABSTENTION} rather than answering from a partial set."
        )
        call = _issue(
            self.client,
            self.config,
            system=build_baseline_prompt(stepwise=True),
            user=_user_message(task, selected, note),
            known_doc_ids=[doc.doc_id for doc in selected],
            cache_prefix=False,
        )
        return _depth_zero_result(
            task,
            answer=call.reply.answer,
            cited=call.reply.cited_doc_ids,
            calls=(call.record,),
            budget_summary=self._budget(),
            detail=(
                f"retrieved {len(selected)} of {len(task.documents)} documents"
                + (", abstained" if call.reply.abstained else "")
            ),
        )
