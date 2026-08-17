# Architecture

This describes how rlm0 is put together and, where it matters more, why each
boundary is where it is. It is written against the code as it stands in August
2026. Several modules are being worked on concurrently, so the sections that
describe something not yet wired up say so in place rather than in a footnote.

## The property everything else exists to protect

One function drives every REPL in the system. `RLM._drive` in
[`src/rlm0/runtime.py`](../src/rlm0/runtime.py) runs the depth-zero attempt,
the depth-two attempt, and the grandchild three levels down. Those three
differ in two arguments, `depth` and `max_depth`, and in nothing else. They
share the prompt builder, the parser, the observation formatter, the sandbox
factory and the budget.

That is not a tidiness preference. A `Run` claims its first attempt is the
counterfactual for the ones after it, and the claim only holds if both went
through the same machinery. A depth-zero path written separately, as a
single-shot completion or a trimmed loop, converts the run's headline
measurement from a measurement of recursion into a measurement of the
scaffolding. Several of the implementations surveyed for this project have
exactly that shape, a `use_repl` flag selecting between two unrelated code
paths, and their reported gains cannot be attributed to anything in
particular.

Everything below is downstream of keeping that one function single.

## The contract, and what it refuses

[`src/rlm0/run.py`](../src/rlm0/run.py) holds no behaviour. It is the project's
argument expressed as types, and its job is to make certain runs
unconstructible.

A `Run` refuses to exist unless its first attempt is bounded at depth zero.
The ordering is not bookkeeping. The runtime genuinely tries depth zero first
and escalates only on failure, so the control is a by-product of how the work
is done rather than a second experiment somebody has to remember to run. Most
published evaluations of this idea are missing the control, and the reason is
that producing it was extra work.

A `Run` also refuses a missing budget summary, and refuses attempts whose
depth bounds are not monotonically increasing. Reordering them is how the
cheapest attempt stops being the one that ran first.

The escape hatch is `BaselineWaiver`, and it is deliberately expensive. It
names an approver, it demands a reason with at least six distinct words of
more than two characters, and it is refused outright if attached to a run that
has its control anyway. A waived run reports `UNTESTED` forever rather than
reading as a success. A rule with no escape gets bypassed rather than obeyed,
so the escape exists and costs something.

At the level below, `Attempt` refuses an answer on an outcome that is not
`ANSWERED`. A conclusion reached after the budget ran out did not survive its
own stop condition, so it is kept in `detail` where a person can read it and a
scorer cannot count it. `Attempt` also refuses a depth-zero attempt containing
sub-calls, and refuses any attempt holding a call deeper than its own bound,
which is how a bound that was not actually enforced becomes a construction
error rather than a silent mislabel.

`CallRecord` keeps `role` and `depth` as separate fields because they answer
different questions. Depth says how far down the call sat, role says whether it
was driving a REPL or being handed a string. A depth-one call can be either.
Conflating them is how a cost table stops being able to say where the money
went.

`cost_usd` is `float | None` throughout, and `None` propagates. An `Attempt`
whose calls include one unpriced call reports `None` for the whole attempt
rather than a partial sum, and a `Run` does the same across attempts. A total
that silently omits what it could not price reads as complete, and it is the
number a budget would be checked against.

`recursion_verdict()` is a measurement over data the run already holds. It can
only see whether the deeper attempt produced an answer where the control did
not. Whether that answer is correct is the harness's question, because the
harness owns the ground truth.

## The seams

[`src/rlm0/ports.py`](../src/rlm0/ports.py) states four Protocols and the value
types that cross them: `LMClient`, `Sandbox`, `Budget`, `DepthPolicy`. They are
Protocols rather than base classes because every one has a test double that
should not have to inherit anything, and because naming a seam is how somebody
else implements it.

The constraint that recurs across all four is that a layer must report what it
spent. A layer that can hide its own cost makes the run-level accounting a
fiction. `LMClient` implementations must return provider-reported token counts
and must not derive usage from string length. `Budget` reserves before dispatch
and settles after. `Sandbox` is the one seam whose contract is about
containment rather than accounting, and it is covered separately in
[THREAT_MODEL.md](THREAT_MODEL.md).

`EscalationContext` is deliberately small: task, context size, attempts so far,
last outcome, last answer, spend, elapsed time, remaining budget, and a
free-form `signals` dictionary. A policy that needs more than this is probably
reaching for the task label, which is how a router becomes overfitted to a
benchmark.

Three seams the runtime needs are not yet in `ports.py`. `Prompter`,
`TurnParser` and `ObservationFormatter` are declared as local Protocols in
`runtime.py` and taken as constructor arguments. That is stated in the module
docstring as a temporary position: inventing an import to a concrete module
would put the orchestrator back in the business of knowing who renders its
prompts. It also means there is currently no adapter in the package that
connects `prompt.py`, `parse.py` and `observation.py` to those Protocols. See
[ROADMAP.md](ROADMAP.md).

## The escalation loop

`RLM.complete` runs attempts until the policy says stop or `max_attempts` is
reached. The first attempt is always bounded at depth zero and there is no
argument that skips it. A caller who genuinely cannot run the control has to
construct the `Run` themselves with a `BaselineWaiver`; this entry point has no
way to produce one.

After each attempt the runtime asks the policy for the next bound and then
refuses anything at or below the bound just run. That would make the attempts
non-monotonic, which `Run` rejects at construction, and would mean the first
attempt was no longer the cheapest. It is treated as "stop" rather than as an
error, because the policy seam exists to be replaced by strangers.

The policies in [`src/rlm0/policy.py`](../src/rlm0/policy.py) are all blind to
what the task is about. `Never` runs depth zero only, and is a first-class
policy rather than a degenerate case: the best-engineered open implementation
of this idea reports that its model never once chose to recurse across every
measured run, so a system that only ever runs depth zero reproduces those
results exactly and for less money. `Fixed` runs one deeper attempt
unconditionally, which is what an ablation needs and what serving a query does
not. `Escalating` is the default and steps deeper only when the previous
attempt did not answer and the budget can fund a whole further attempt.

`can_fund_another_attempt` is the part worth reading. A granted reservation
says one more call is allowed, which is a different question from whether a
deeper attempt fits. A deep attempt needs at least a root turn and the sub-call
it exists to make, so a ceiling with one call left funds a deep attempt that
stops before it can differ from the shallow one. Escalating into a ceiling that
cannot fund the deeper attempt buys a truncated deep trajectory, which costs
more than the shallow attempt and answers less. The published cost tail is made
almost entirely of trajectories like that.

`Escalating.stop_on_error` treats `ERRORED` as terminal, because an error is
the environment failing rather than the task being hard, and the deeper attempt
would run in the same environment with more calls in it.

Inside one attempt, `_turns` loops up to `max_iterations`. Each turn reserves
budget, calls the model, parses the reply, and either returns an answer, runs
code and formats the observation, or nudges the model when it did neither. Two
calls are reserved per turn, this one and the wind-down this turn may turn out
to need, because a ceiling that grants its last call to a working turn leaves
nothing to ask for a final answer with.

Sub-call wiring is unconditional on which sandbox was chosen. In one surveyed
implementation recursion was unreachable for every remote environment because
the host-call bridge sat inside a branch only the local one entered, and
nothing said so; runs completed, numbers were published, and the recursion
never happened. Here the wiring is attempted for every attempt bounded above
zero, and a sandbox that cannot service host calls raises
`RecursionUnavailableError` rather than producing a flat run wearing a depth
label.

When a sub-call is serviced, the child is handed a variable *name* from the
parent's environment, not its contents. The host reads that variable out of the
parent sandbox and binds it into the child's, so the slice never crosses a
prompt. A surveyed implementation passes the child a single string that is both
its query and its entire context, which reproduces at depth one the exact
problem the REPL exists to solve and makes depth two pointless.

## The sandbox channel

Both sandboxes are the same class with a different `_spawn`.
[`_channel.py`](../src/rlm0/sandbox/_channel.py) holds the host half:
framing, deadlines, host calls, truncation and restart.
[`_guest.py`](../src/rlm0/sandbox/_guest.py) is the guest half, which is never
imported and is shipped as text over stdin into a Python that has none of this
project installed.

The wire is newline-delimited JSON, `ensure_ascii=True`, one message per line.
It is JSON because the host must never deserialize sandbox output with a format
that can execute code. A surveyed implementation calls `pickle.loads` on
sandbox output in the host process, which hands the host away the moment
anything achieves execution inside the boundary.

Messages carry ids and are correlated by them. The design this started from ran
one request at a time with no correlation field and skipped lines it could not
parse, so a desynchronized stream degraded into plausible-looking wrong answers
instead of an error. Here a framing mismatch is a `ProtocolError` and is always
fatal for the channel.

Reads happen on a thread feeding a queue, so a deadline is a `queue.get`
timeout rather than a blocking read the host cannot get out of. That is what
makes `while True:` in model-written code survivable on Windows as well as on
Linux.

The guest arrives via a one-line loader passed as `-c`, which reads a byte
count and then the source from stdin. Command-line length limits are low on
Windows, and a mounted host path into the container is a seam this design
refuses to open.

A `host_call` message pauses the guest's own deadline and extends the host's,
by exactly the time the host spent. The execution budget bounds the model's
code; charging a provider's latency against it would make timeouts depend on
how busy an API was that afternoon.

Timeouts are enforced twice. The guest arms a repeating interval timer that
raises a `BaseException` subclass, so neither `except Exception` nor a loop that
swallows everything holds the interpreter past its budget, and re-arming turns
swallowing the deadline into a busy wait rather than an escape. The host keeps
a hard kill in reserve two seconds later, because a loop inside a C extension
never reaches a bytecode boundary and will not notice a Python signal handler.
On Windows there is no interval timer at all, so the hard kill is the only
bound there, and the environment does not survive it.

When the sandbox has to be replaced, `execute` returns a failed `ExecResult`
rather than raising, and the message says plainly that every variable is gone
and the context needs re-seeding. Silently continuing with an empty environment
would let a later step read a missing variable as an empty one.

Two caps hold the architecture together. `STDOUT_CHAR_CAP` (8,000 characters,
applied in the guest) keeps a runaway print from becoming a multi-gigabyte line
on the pipe. `DEFAULT_STDOUT_CAP` (4,000 characters, applied in
[`observation.py`](../src/rlm0/observation.py)) is what the model actually sees.
If the model can print a large slice and read it, the REPL is a delivery
mechanism for context and the whole pattern collapses into one very long
single-shot prompt. Raising that cap to the point where a document fits is the
single change that quietly undoes the design. When truncation fires, the notice
says so in as many words, because that is the only place the model is taught
the lesson at the moment it is needed.

Variables are reported by name and never by value, for the same reason. A
single runtime that helpfully echoes the environment puts the whole context
back in the window and nothing about the run looks different afterwards except
the bill.

## The budget ledger

[`src/rlm0/budget.py`](../src/rlm0/budget.py) exists because the number of calls
in a recursive run is decided by a model, at runtime, inside a tree whose width
nobody declared. The published cost distributions have standard deviations at
roughly double their means, and the tail is not random: cost explodes exactly on
the trajectories where the model cannot find the answer and keeps spawning
children to look for it. Cost and quality fail together, which removes the usual
comfort that an expensive run at least bought something.

Two choices carry the module. The budget is shared across the whole run rather
than granted per level, because a per-level budget multiplies with fan-out: give
each of eight children the parent's allowance and the tree is authorised for
eight times the parent's spend without anybody having written that number down.
And permission is taken before dispatch, not checked after, because a counter
inspected after a call only reports, and a fan-out that checks between
dispatches has already landed half its calls by the time it notices.

Reservations are all or nothing. `CallReservation` has no field for "how many of
the twenty", so a partial grant would be indistinguishable on the wire from a
full one, and the caller that wrote code fanning out over twenty slices would
discover mid-loop that eight of them have no answer.

Refusal is a signal rather than an exception. A refused runtime tells the model
what is left and winds down to a final answer, which is strictly better than an
exception unwinding a tree that was halfway to an answer. A granted reservation
can also carry a soft-threshold advisory, because a model told it has two calls
left writes a different next block than one refused at the wall.

Unpriced calls fail closed. When a USD ceiling is set and an unpriced settlement
arrives, the budget records it, warns once, and past `max_unpriced_calls`
refuses every subsequent reservation with a reason naming the problem. The
default is zero, so the first unpriced call trips it. The alternatives were
considered and rejected in the class docstring: counting unpriced calls at zero
is the defect being fixed, raising from `settle` destroys the result of a call
already paid for, and substituting a guessed price makes the ledger a fiction
that reads as a measurement.

The class is honest about what a ceiling can promise. The call ceiling is
exact, because batch size is known before dispatch. Tokens and USD are exact
only to the extent the caller's estimates are, and can be overshot by the error
on calls already in flight, but not repeatedly, because once the settled ledger
crosses the line every further reservation is refused. The wall-clock ceiling
stops new dispatch and does not abort a running call, which is the sandbox
layer's job.

`Unbounded` exists so that no run record can imply a ceiling nobody set. Its
summary leads with the word unbounded and `Run` stores that string verbatim. A
`RunBudget` with every ceiling unset is refused at construction, because its
summary would be a line full of the word "unset" that a reader could skim as
bounded.

There is no refund or release in the `Budget` seam. `reserve` and `settle`
exist; nothing returns an unused reservation. The consequence is described
under known gaps in [ROADMAP.md](ROADMAP.md), and it matters because the
runtime reserves two calls per turn and settles one.

## Prompt and parse

[`prompt.py`](../src/rlm0/prompt.py) builds the system prompt from named
sections. Exactly three of them, `tools`, `strategies` and `sizing`, may depend
on whether sub-calls exist. The rest are byte-identical between the two variants
by construction rather than by careful editing, and `tests/test_prompt.py`
asserts it. If the depth-zero prompt differed in tone, section order or
recommended strategy, the measured difference between the two attempts would
include the scaffolding and would no longer isolate the recursion. Turn prompts
carry no sub-call wording at all in either variant, for the same reason once per
turn.

The other load-bearing number in that file is the sub-call size. The guidance is
roughly 500,000 characters per call, with a hard floor at 50,000. One surveyed
implementation caps the slice at 500 characters. That is a thousandfold
divergence and it changes what the sub-model is: at 500K it is an analyst that
can answer a question about a set of documents, at 500 it is a classifier that
can label a snippet. A classifier cannot aggregate, so the root model is forced
to loop over the whole context one snippet at a time, which is the brute-force
behaviour that implementation's own README complains about.

[`parse.py`](../src/rlm0/parse.py) handles four failure modes, every one of them
observed in a surveyed implementation and every one of them silent. A non-greedy
paren match truncates every answer containing a parenthesis, so the argument is
matched by balancing instead, with a repair for the case where the model wrote
an unbalanced answer. A directive matched anywhere fires on a model quoting the
instructions back, so code fences and inline backtick spans are masked out
before the search, with offsets preserved so the answer can still be sliced from
the original text. A sentinel variable fires whenever the model uses a common
name for scratch, so there is no sentinel: a variable becomes the answer only
when the model names it in `FINAL_VAR`. And a final answer accepted before any
code has run is the model answering from priors, so it is rejected with a
`Rejection` the caller can log rather than dropped.

A message carrying both runnable code and a directive runs the code and refuses
the directive. The orderings are not symmetric: deferring costs one turn and the
model can repeat itself, whereas accepting throws away code the model wrote
because it thought it still needed it.

Nothing in the module raises on malformed input. A model that emits nonsense
should get another turn, not a traceback in the orchestrator.

## History compaction

Root-side tokens grow with the length of the conversation rather than with the
work, so a long run pays for its own transcript on every turn.
`compact_history` moves the oldest turns into REPL variables, where they remain
reachable by code, and replaces them with one line naming what moved and which
buffers the model still owns. The inventory is the part that has to survive: a
model that forgets it already built `summaries` builds it again, and rebuilding
a buffer is the most expensive mistake available to it. The most recent turns
are never folded regardless of budget, because the next action is chosen almost
entirely from what just happened.

`compact_history` returns the stashes for the caller to write with
`Sandbox.set_variable` and does not touch the sandbox itself, which is how it
gets tested with no sandbox in the loop.

## Providers

[`providers/`](../src/rlm0/providers/) is the only place in the project that
talks to a model API. The SDKs are optional extras loaded through `importlib`,
so nothing in the package has a static dependency on either one and the full
test suite runs with neither installed and no network.

Prefix caching is the reason this layer is written carefully. Every sub-call in
a fan-out shares a prefix by construction, differing only in the slice at the
end. The Anthropic client places two breakpoints when asked: one on the system
block, one on the last message of the stable head, meaning index -2. That
placement is chosen for the fan-out shape specifically. The usual multi-turn
advice, marking the final message, would write a distinct cache entry per child
and read none of them.

`cached_prefix` is reported from what the provider said, never from the fact
that we asked. Asking and receiving are different events, and the gap between
them is the bug the diagnostic exists to catch. The module docstring lists what
callers must not do, in rough order of how often each actually happens, and the
first item is interpolating anything variable into the system block.

The Anthropic docstring also records a constraint that inverts the naive design:
a cache entry becomes available only once the first response has begun, and
parallel requests sharing a prefix do not hit each other's cache. A cold fan-out
of N children pays N prefix writes at 1.25x base input, which is worse than not
caching at all. Sub-calls are sequential today so the problem does not arise;
the note exists so that adding parallel fan-out without a warming barrier does
not silently make every run more expensive.

OpenAI caches automatically and offers no way to mark a block, so `cache_prefix`
is accepted and ignored rather than emulated. A client that pretended to honour
the flag would make the flag useless as a diagnostic everywhere else. What is
not ignored is `prompt_tokens_details.cached_tokens`, which is a real
measurement and populates `cache_read_tokens`. The arithmetic differs between
the two providers: OpenAI's `prompt_tokens` is inclusive of cached tokens and
Anthropic's `input_tokens` is not, so the cached count is subtracted out in the
OpenAI client and only there.

Request-shape differences between chat and reasoning models are a registry keyed
by exact model name in [`params.py`](../src/rlm0/providers/params.py), not a
substring test on the name. The registry is wrong in a loud, immediate way that
names the parameter and is fixed by one line the caller can pass in. A substring
test is wrong in both directions and gets more wrong over time.

A response carrying no usage object raises `ProviderResponseError` rather than
defaulting to zeros, because zero tokens makes the call free in every downstream
total.

Retry lives in the library layer, which none of the surveyed implementations do.
`Retry-After` is honoured when sent, and a wait longer than
`max_retry_after_s` is re-raised rather than slept through, because the run
layer is the thing holding a wall-clock budget. Retried calls cannot
double-count, and that is structural rather than remembered: a failed attempt
raises and produces no response, so usage is read exactly once from the response
that finally came back.

## The harness

[`harness/`](../src/rlm0/harness/) is four pieces. The corpus generator, the
grader, the report, and the runner. `EVALUATION.md` covers the protocol; what
follows is where the boundaries sit.

`Solver` is deliberately minimal so a competing implementation can be measured
on the same corpus without adopting any of this project's internals. What it is
not is optional: a solver must return a `Run`, which means it must have
attempted depth zero, attributed its calls, and named the budget it executed
under. That is the price of appearing in a table.

The runner checks the solver against its own accounting rather than trusting it.
An answer that appears in the result but not in the `Run` is an answer that was
not paid for. A citation naming a document not in the context is refused.
Records are written and fsynced per sample, so a sweep that dies at sample
ninety is still worth ninety samples, and resuming across a corpus change is
refused because pooling results measured on different text produces an aggregate
that measures nothing.

The corpus generator re-derives every answer from the text it just emitted, by
regex, the way a solver would, and refuses to hand back a corpus whose ground
truth disagrees. Re-deriving from the objects the text was rendered from would
only prove the renderer was called.

`ResultTable.check` raises rather than warns. A warning next to a printed number
is read as a caveat on a result; the result is the thing that must not exist.
It refuses a table with no depth-zero row, and refuses rows measured on
different corpora, under different grading policies, or over different sample
sets.

## Layering, in one pass

`run.py` is the bottom and depends on nothing. `ports.py` depends only on
`run.py`. `budget.py`, `policy.py`, `observation.py` and the providers depend on
`ports.py` and `run.py`. `prompt.py` and `parse.py` depend on nothing at all,
which is what lets both be tested with no model, no sandbox and no clock.
`sandbox/` depends on `ports.py` for its result and error types and on nothing
above it. `runtime.py` sits on top and imports the seams rather than the
implementations, with the single exception of `Escalating` as the default
policy. The harness sits beside the runtime rather than above it, and talks to a
`Solver` rather than to `RLM`.

The direction of every arrow is towards the contract. The point is that a claim
made by `run.py` cannot be undermined by anything further up, because nothing
further up can construct a `Run` that does not satisfy it.

## What is not wired up yet

Stated here rather than left to be discovered.

The runtime calls `set_host_call_handler` on the sandbox to bind the host side
of a sub-call, and `ChannelSandbox` exposes that operation as `bind_host_call`.
The names do not match, so as the code stands neither `DockerSandbox` nor
`SubprocessSandbox` satisfies the structural `SubCallSandbox` check and any
attempt bounded above depth zero raises `RecursionUnavailableError`. That is
the loud failure the design asked for rather than a silent flat run, but it
means the recursive path is not currently reachable with a real sandbox.

`RLM._probe_budget` asks the budget for a zero-call reservation as a read-only
probe, and both `RunBudget.reserve` and `Unbounded.reserve` raise `ValueError`
for `n_calls < 1`. `RunBudget` has a `remaining()` method shaped for exactly
this purpose and the runtime does not use it, because `remaining()` is not on
the `Budget` Protocol.

The prompt module names the sub-call `llm_query` with one argument and the
context variable `context`; the runtime registers `rlm_call` with arity two and
binds the context to `CONTEXT`. There is no adapter connecting `prompt.py`,
`parse.py` and `observation.py` to the `Prompter`, `TurnParser` and
`ObservationFormatter` Protocols the runtime takes.

Each of these is a small piece of wiring and all three are in flux. They are
recorded because the layering above is otherwise accurate and it would be easy
to read this document as describing a system that runs end to end today.
