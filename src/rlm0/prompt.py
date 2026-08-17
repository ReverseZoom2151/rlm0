"""What the root model is told, in two variants that differ in one place.

Two things in here are load bearing and both are easy to get wrong.

The first is the sub-call size. A sub-call is a whole model with a whole
context window, so the unit of work handed to it should be a document set, not
a sentence. The guidance here is roughly 500K characters per call, which is
about the largest slice a long-context model will actually read, and it means a
thirty million character context is covered by tens of calls rather than tens
of thousands. One implementation surveyed for this project capped the slice at
500 characters. That is a thousandfold divergence and it changes what the
sub-model is: at 500K it is an analyst that can answer a question about a set
of documents, and at 500 it is a classifier that can label a snippet. A
classifier cannot aggregate, so the root model is forced to loop over the whole
context one snippet at a time, which is exactly the brute-force behaviour that
implementation's own README complains about. The number is not a tuning
parameter, it is the difference between decomposition and enumeration.

The second is that the depth-zero prompt is the control the entire project is
measured against. If it differs from the recursive prompt in tone, in section
order, or in the strategies it recommends, then the measured difference between
the two attempts includes the scaffolding and no longer isolates the recursion.
So the prompt is assembled from named sections, exactly three of which are
allowed to depend on whether sub-calls exist, and the rest are byte identical by
construction rather than by careful editing. `tests/test_prompt.py` asserts it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MIN_SUB_CALL_CHARS",
    "SUB_CALL_CHAR_BUDGET",
    "SUB_CALL_SECTIONS",
    "ContextShape",
    "PromptSection",
    "build_system_prompt",
    "build_turn_prompt",
    "system_prompt_sections",
]

SUB_CALL_CHAR_BUDGET = 500_000
"""Characters of context to put in one sub-call. See the module docstring."""

MIN_SUB_CALL_CHARS = 50_000
"""Below this a sub-call stops being an analyst and becomes a classifier.

Set an order of magnitude under the default rather than just above the
pathological 500, because the failure is gradual: the smaller the slice, the
more the root model has to do itself, and there is no sharp edge to detect.
"""

_MAX_LISTED_CHUNKS = 100


@dataclass(frozen=True, slots=True)
class ContextShape:
    """The only thing the model is told about the context: its size.

    Contents never appear here. The whole point of holding the context in the
    REPL is that it does not enter the window, and a shape object that carries
    a preview is the most natural place for that guarantee to be lost.
    """

    total_chars: int
    chunk_lengths: tuple[int, ...] = ()
    kind: str = "str"

    def __post_init__(self) -> None:
        if self.total_chars < 0:
            raise ValueError("total_chars cannot be negative")
        if any(n < 0 for n in self.chunk_lengths):
            raise ValueError("chunk lengths cannot be negative")
        if self.chunk_lengths and sum(self.chunk_lengths) != self.total_chars:
            raise ValueError(
                "chunk lengths must sum to total_chars; a shape that does not "
                "add up is one the model will plan a split against and miss"
            )

    @classmethod
    def of(cls, context: str | Sequence[str]) -> ContextShape:
        if isinstance(context, str):
            return cls(total_chars=len(context), kind="str")
        lengths = tuple(len(chunk) for chunk in context)
        return cls(
            total_chars=sum(lengths),
            chunk_lengths=lengths,
            kind="list[str]",
        )

    @property
    def n_chunks(self) -> int:
        return len(self.chunk_lengths)

    def describe(self) -> str:
        """The shape paragraph that opens the prompt."""
        if not self.chunk_lengths:
            return (
                f"Your `context` variable is a {self.kind} holding "
                f"{self.total_chars} characters."
            )
        shown = list(self.chunk_lengths[:_MAX_LISTED_CHUNKS])
        listing = ", ".join(str(n) for n in shown)
        hidden = self.n_chunks - len(shown)
        tail = f"... and {hidden} more" if hidden else ""
        return (
            f"Your `context` variable is a {self.kind} of {self.n_chunks} "
            f"chunks holding {self.total_chars} characters in total. The "
            f"chunk lengths in order are: [{listing}]{tail}."
        )


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One named piece of the system prompt.

    Named so that the two variants can be compared section by section instead
    of by diffing two blobs and hoping.
    """

    name: str
    body: str


SUB_CALL_SECTIONS: frozenset[str] = frozenset({"tools", "strategies", "sizing"})
"""The only sections allowed to depend on whether sub-calls exist."""


_ROLE = """\
You are answering a query about a context that is far too large for you to
read. The context has already been loaded into a Python REPL as a variable
named `context`. You will never see the context itself. You act on it only by
writing code, and you will be asked for your next action again after every
turn until you give a final answer."""


_PROTOCOL = """\
To run code, write a fenced block tagged `repl`:

```repl
print(type(context), len(context))
```

Rules for those blocks:
- Everything you emit outside a fenced block is thinking, and never runs.
- Blocks run in order, in one long-lived session. Names you bind on one turn
  are still bound on the next, so variables are your working memory.
- Write code that does something. A turn that only restates a plan costs a
  call and buys nothing."""


_TRUNCATION = """\
You see only a truncated view of whatever your code prints. Printing a large
slice does not deliver it to you, it delivers the first part of it and a
notice. So do the reduction inside the REPL and print the result of the
reduction: counts, indices, keys, short extracts, the answer itself. Treat the
truncation as a rule about where work happens rather than as an obstacle to
route around."""


_FINAL = """\
Finishing. When, and only when, you have the answer, end a turn with:

RLM0_FINAL_V1({"protocol_version": 1, "status": "answered", "answer":
"your complete answer", "evidence": ["short supporting facts or locations"],
"answer_artifact": null})

The JSON object must contain exactly those five fields. `evidence` is a list
of short strings naming what you actually found, and `answer_artifact` is null
unless a host-provided answer artifact is available. This protocol is how a
result becomes reportable.

For compatibility, FINAL(your complete answer) and FINAL_VAR(name) are still
accepted, but they are recorded as recovered legacy completions. FINAL_VAR
returns the current value of that REPL variable as the answer, which is how an
answer longer than your output limit gets out. Pass a bare variable name.

Both are read from your prose, not from your code. A directive written inside
a ```repl block, inside a string, or inside backticks is ignored, so quoting
these instructions is safe. A directive is also ignored on a turn where you
have not yet run any code, because an answer produced before looking at the
context is a guess from memory, not a reading of the context."""


_CLOSING = """\
Plan briefly, then carry the plan out in the same turn. Do not describe what
you are about to do and stop. Your first turn sets the decomposition, and a
decomposition chosen carelessly is rarely recovered from later, so spend that
turn on establishing the real structure of the context rather than on guessing
where the answer might be. Answer the original query explicitly at the end."""


def _tools(sub_calls: bool, sub_call_chars: int) -> str:
    if sub_calls:
        return f"""\
The REPL session holds:
1. `context`, as described above.
2. `llm_query(prompt: str) -> str`, which sends one prompt to a sub-model and
   returns its reply as a string. The sub-model reads only what you put in
   that prompt. It cannot see the REPL, it cannot see your conversation, and
   it remembers nothing between calls, so every call has to be self contained.
   It reads about {sub_call_chars} characters comfortably.
3. `rlm_batch([[prompt, handle], ...]) -> list[str]`, for independent sub-
   questions that can be answered from named REPL variables. Each `handle`
   names a variable holding the slice that child should read. The host reserves
   the whole batch before it starts, warms the shared cache prefix once, and
   returns answers in the same order. Use it only when the children do not
   depend on one another.
4. `print()`, which is the only way anything reaches you.

`llm_query` is what makes reading possible at all. Code can find and count and
split, but any judgement about what the text means has to be made by a model
that has read the text, and that model is the sub-model."""
    return """\
The REPL session holds:
1. `context`, as described above.
2. `print()`, which is the only way anything reaches you.

There is no sub-model on this attempt. Every judgement about what the text
means has to be made either by code you write or by you, from an extract small
enough to survive truncation, so choose what you print with that in mind."""


def _strategies(sub_calls: bool) -> str:
    if sub_calls:
        return """\
Three strategies that work. Adapt them, do not copy them.

1. Iterative buffering. Walk the context in order, carrying forward what you
   have learned so far, when the answer depends on accumulated state such as a
   count, a chronology, or a running comparison.

```repl
buffer = "nothing yet"
for i, section in enumerate(context):
    buffer = llm_query(
        f"Question: {query}\\n"
        f"Known so far: {buffer}\\n"
        f"Section {i} of {len(context)} follows. Update what is known and "
        f"reply with the updated state only.\\n\\n{section}"
    )
print(buffer[:800])
```

2. Split and aggregate. Cut the context into as few parts as the sub-model can
   hold, ask each the same question, then run one more pass over the replies,
   when the answer is somewhere specific and you do not know where.

```repl
per_call = 500_000
groups, cur, size = [], [], 0
for chunk in context:
    if size + len(chunk) > per_call and cur:
        groups.append(cur)
        cur, size = [], 0
    cur.append(chunk)
    size += len(chunk)
if cur:
    groups.append(cur)
print("groups:", len(groups))
answers = [
    llm_query(f"{query}\\n\\nAnswer only from these documents, and say "
              f"NOT FOUND if they do not contain it.\\n\\n" + "\\n\\n".join(g))
    for g in groups
]
merged = llm_query(f"{query}\\n\\nPartial answers:\\n" + "\\n".join(answers))
print(merged[:800])
```

3. Split on structure. Find the seams the document already has, headers,
   record separators, dates, and split on those instead of on a character
   count, when the context is structured. A split that respects structure
   gives the sub-model whole units and stops answers falling across a boundary.

```repl
import re
parts = re.split(r"\\n#+ ", context if isinstance(context, str) else
                 "\\n".join(context))
print(len(parts), [p[:60] for p in parts[:5]])
notes = [llm_query(f"{query}\\n\\nSection:\\n{p}") for p in parts]
```"""
    return """\
Three strategies that work. Adapt them, do not copy them.

1. Iterative buffering. Walk the context in order, carrying forward what you
   have learned so far, when the answer depends on accumulated state such as a
   count, a chronology, or a running comparison.

```repl
import re
buffer = []
for i, section in enumerate(context):
    hits = re.findall(r"(?i)the term you care about", section)
    if hits:
        buffer.append((i, len(hits), section[:200]))
print(len(buffer), buffer[:5])
```

2. Split and aggregate. Cut the context into as few parts as you can, compute
   the same reduction over each, then combine the reductions, when the answer
   is somewhere specific and you do not know where.

```repl
per_group = 500_000
groups, cur, size = [], [], 0
for chunk in context:
    if size + len(chunk) > per_group and cur:
        groups.append(cur)
        cur, size = [], 0
    cur.append(chunk)
    size += len(chunk)
if cur:
    groups.append(cur)
print("groups:", len(groups))
scores = [sum(g_text.lower().count("keyword")
              for g_text in group) for group in groups]
print(scores)
```

3. Split on structure. Find the seams the document already has, headers,
   record separators, dates, and split on those instead of on a character
   count, when the context is structured. A split that respects structure
   gives you whole units and stops answers falling across a boundary.

```repl
import re
parts = re.split(r"\\n#+ ", context if isinstance(context, str) else
                 "\\n".join(context))
print(len(parts), [p[:60] for p in parts[:5]])
```"""


def _sizing(sub_calls: bool, sub_call_chars: int) -> str:
    if sub_calls:
        return f"""\
Sizing your sub-calls. Aim for about {sub_call_chars} characters of context
per call, and treat anything under {MIN_SUB_CALL_CHARS} as a mistake unless
you have a reason. The sub-model is a full model, not a snippet labeller: give
it a set of documents and a real question and it will answer the question. A
context of ten million characters is about twenty calls at this size, and
about twenty thousand calls at a size that fits in a paragraph. If you find
yourself writing a loop with thousands of iterations, the slice is too small
and the loop is doing the reading that the sub-model was there to do.

Sizing down is worth it only for a reason you can name: an aggregation that
needs per-document granularity, a boundary you must not cross, an answer you
want located rather than summarised. Cost scales with the number of calls, so
fewer and larger is cheaper as well as better."""
    return f"""\
Sizing your passes. There is no sub-model on this attempt, so the equivalent
decision is how much you reduce per pass. Work over groups of roughly
{sub_call_chars} characters and reduce each group to something small enough to
print, rather than looping over the context in tiny pieces and printing as you
go. A loop with thousands of iterations will produce more output than you can
see, and truncation will hide the part that mattered.

Reducing further is worth it only for a reason you can name: an aggregation
that needs per-document granularity, a boundary you must not cross, an answer
you want located rather than summarised. Each pass costs you a turn, so fewer
and larger is cheaper as well as better."""


def system_prompt_sections(
    shape: ContextShape,
    *,
    sub_calls: bool,
    sub_call_chars: int = SUB_CALL_CHAR_BUDGET,
) -> tuple[PromptSection, ...]:
    """The system prompt as named sections, in order.

    Exposed rather than kept private so the invariant that matters can be
    tested directly: every section whose name is not in SUB_CALL_SECTIONS is
    identical between the two variants, and the section order is the same.
    """
    if sub_call_chars < MIN_SUB_CALL_CHARS:
        raise ValueError(
            f"sub_call_chars={sub_call_chars} is below MIN_SUB_CALL_CHARS="
            f"{MIN_SUB_CALL_CHARS}. A slice that small turns the sub-model "
            "into a snippet classifier and forces the root model to brute "
            "force the context one piece at a time."
        )
    return (
        PromptSection("role", _ROLE),
        PromptSection("shape", _shape_section(shape)),
        PromptSection("tools", _tools(sub_calls, sub_call_chars)),
        PromptSection("protocol", _PROTOCOL),
        PromptSection("truncation", _TRUNCATION),
        PromptSection("strategies", _strategies(sub_calls)),
        PromptSection("sizing", _sizing(sub_calls, sub_call_chars)),
        PromptSection("final", _FINAL),
        PromptSection("closing", _CLOSING),
    )


def _shape_section(shape: ContextShape) -> str:
    return (
        f"{shape.describe()}\n\n"
        "Those sizes are everything you have been told. No part of the "
        "context has been shown to you, so any claim you make about what it "
        "says has to come from code you actually ran."
    )


def build_system_prompt(
    shape: ContextShape,
    *,
    sub_calls: bool,
    sub_call_chars: int = SUB_CALL_CHAR_BUDGET,
) -> str:
    """The system prompt for one attempt.

    `sub_calls=False` builds the depth-zero control. It is the same prompt
    with the three sub-call dependent sections swapped, never a different
    prompt.
    """
    sections = system_prompt_sections(
        shape, sub_calls=sub_calls, sub_call_chars=sub_call_chars
    )
    return "\n\n".join(section.body for section in sections)


_TURN = """\
Your next action, working towards this query: {query}

Continue in the REPL, which holds `context` and everything you have bound so
far, by writing a ```repl block. State your answer only when you have it."""

_FIRST_TURN_SAFEGUARD = """\
You have not run any code yet and have not seen one character of the context.
Anything you could answer right now would come from memory rather than from
the context, and would be rejected. Spend this turn establishing the actual
structure of what you are holding: its type, its size, how it divides, and
where the query's terms occur. The split you choose now is the one you will
live with.

"""

_WRAP_UP = """\
This is your last turn. Give the best answer your buffers support, using
FINAL(...) or FINAL_VAR(...), and say plainly what you were unable to check.

"""


def build_turn_prompt(query: str, *, iteration: int = 0, wrap_up: bool = False) -> str:
    """The user message for one turn.

    Carries no sub-call wording at all, in either variant. Turn framing that
    varied with depth would reintroduce, once per turn, exactly the difference
    the section split exists to keep out of the comparison.
    """
    if iteration < 0:
        raise ValueError("iteration cannot be negative")
    body = _TURN.format(query=query)
    if wrap_up:
        return _WRAP_UP + body
    if iteration == 0:
        return _FIRST_TURN_SAFEGUARD + body
    return (
        "Above is your own record of this session: what you ran and what came "
        "back.\n\n" + body
    )
