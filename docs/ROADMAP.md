# Roadmap

What is built, what is not, and what is blocked on what. Written against the
code as of August 2026, with several modules under concurrent development, so
anything marked in flux should be checked against the source before it is
relied on.

The README's Limitations section is the short version. This is the long one.

## Built

The contract in [`run.py`](../src/rlm0/run.py) and the seams in
[`ports.py`](../src/rlm0/ports.py) are done and tested. So are the budget
ledger, the depth policies, the prompt assembly, the turn parser, the
observation formatter and history compaction, the sandbox channel with both
backends, the Anthropic and OpenAI clients with pricing and retry, and all four
pieces of the harness.

The runtime loop exists and its shape is settled: one function drives every
depth, the control runs first, and sub-call wiring is unconditional on the
sandbox choice. What is not settled is the wiring between it and the concrete
prompt, parse and sandbox modules, which is the first item below.

## In flux, and blocking the recursive path

Three pieces of wiring are mismatched right now. None is hard to fix and all
three matter, so they are listed rather than left to be discovered.

**The host-call handler name.** `RLM._wire_sub_calls` requires a sandbox
exposing `set_host_call_handler`. `ChannelSandbox` exposes the same operation as
`bind_host_call`. As the code stands, neither `DockerSandbox` nor
`SubprocessSandbox` satisfies the structural `SubCallSandbox` check, so any
attempt bounded above depth zero raises `RecursionUnavailableError`. That is the
loud failure the design asked for rather than a silent flat run wearing a depth
label, but it means the recursive path is not reachable with a real sandbox
today. Everything else in this section is downstream of resolving this one.

**The budget probe.** `RLM._probe_budget` reserves zero calls as a read-only
query, and both `RunBudget.reserve` and `Unbounded.reserve` raise `ValueError`
below one call. `RunBudget` already has `remaining()`, shaped for exactly this,
and the runtime does not use it because `remaining()` is not on the `Budget`
Protocol. The fix is either to put it there or to make a zero-call reservation
legal, and the choice should be made in `ports.py` rather than worked around in
the runtime.

**No adapter between the runtime Protocols and the concrete modules.**
`runtime.py` declares `Prompter`, `TurnParser` and `ObservationFormatter`
locally and takes them as constructor arguments, deliberately, so the
orchestrator does not import who renders its prompts. But nothing in the package
implements them over `prompt.py`, `parse.py` and `observation.py`. The names also
disagree: the prompt names the sub-call `llm_query` with one argument and the
context variable `context`, while the runtime registers `rlm_call` with arity two
and binds `CONTEXT`. Reconciling those is part of the same piece of work, and
`ports.py` should probably grow the three seams while it happens.

## Known gaps in what is built

**No refund in the budget lifecycle.** `Budget` has `reserve` and `settle` and
nothing that returns an unused reservation. The runtime reserves two calls per
turn, this one and the wind-down it may need, and settles one, so the held-back
call stays in flight for the life of the run. Over an eight-iteration attempt
against a twenty-call ceiling, that is a materially tighter effective bound than
the one written down. The published budget work
([arXiv:2606.04056](https://arxiv.org/abs/2606.04056)) names the lifecycle as
reserve, reconcile and refund; the third verb is the one missing here. Adding it
means a reservation handle, which changes the `Budget` Protocol.

**Static over-reservation.** `_estimate_tokens` is characters over four, which
is a bad token count and a fine reservation hint, and it is never recorded as
usage. The same Rust work concedes 4 to 6x static over-reservation as the state
of the art, and [RELATED_WORK.md](RELATED_WORK.md) identifies a tighter fan-out
estimator as one of the few things this project could still claim. A runtime
that knows its own call tree width in advance should beat a static factor.
Nothing has been built towards it.

**Sub-calls are sequential.** This is not a limitation so much as a deliberate
deferral. Parallel fan-out without a warming barrier makes every fan-out
quietly more expensive, because a cache entry becomes available only after the
first response begins and parallel requests sharing a prefix do not hit each
other's cache. A cold fan-out of N children pays N prefix writes at 1.25x base
input, which is worse than not caching at all. The constraint is recorded in
[`anthropic_client.py`](../src/rlm0/providers/anthropic_client.py) so that
adding concurrency does not undo the cost model by accident. Parallel sub-calls
also need a review of the channel, which currently assumes one outstanding host
call.

**Graceful degradation is partly built.** The wind-down path exists: a refused
reservation spends a held-back call telling the model to stop and summarise, and
the reply goes into `detail` rather than becoming an answer. What does not exist
is anything that degrades progressively, meaning a run that notices it is
approaching its ceiling and narrows its own fan-out rather than continuing at
full width until refused. `RunBudget.advisory()` produces the text for it and
nothing consumes that text yet. A targeted search found nothing published on
this at all, so it is one of the few genuinely open contributions available
here.

**No noise floor computation.** The evaluation protocol requires paired noise
floors, and the harness does not compute one. It is a procedure over
`records.jsonl` rather than a feature, but it needs to exist before any delta is
reported.

**Windows loses the guest-side deadline.** There is no interval timer on
Windows, so only the host's hard kill applies and the sandbox environment does
not survive a timeout there. The default sandbox is a Linux container where the
path always exists, so this affects development on Windows rather than
production, and it is stated in `_guest.py` rather than hidden.

## Not built

**A microVM sandbox backend.** The highest-value item on this list. The 2026
consensus is that shared-kernel container isolation is no longer adequate for
model-written code, so the Docker backend here is a floor rather than a ceiling,
and a kernel escape defeats every control in
[THREAT_MODEL.md](THREAT_MODEL.md) at once. `microsandbox` is Apache-2.0 libkrun
microVMs booting under 200ms and self-hostable, which makes this tractable
rather than research. The channel abstraction is already the right shape: a
microVM backend is a third `_spawn` and nothing above `ChannelSandbox` needs to
know. Blocked on nothing except the work.

**Google and Gemini provider clients.** Neither exists. The seam is
`LMClient.complete` and the two existing clients are the pattern, so this is
mostly a question of getting the usage fields and the caching semantics right,
which is where both existing clients spend most of their code. Gemini's caching
model differs from both Anthropic's explicit breakpoints and OpenAI's automatic
prefix matching, and the client should report what the provider says rather than
what was asked for, the same as the others.

**Prices for the OpenAI reasoning line.** Deliberately absent.
[`pricing.py`](../src/rlm0/providers/pricing.py) carries two OpenAI entries,
`gpt-4o` and `gpt-4o-mini`, and stops there. The reasoning models are the
conspicuous omission: their prices have moved more than once, and a stale entry
for an expensive model is the worst possible error in this file, because it is
wrong in the direction of looking cheap and it never announces itself. Calls to
them report `cost_usd=None`, which shows up in `PriceTable.unpriced_models` and
in a warning, and a caller who knows the current rate supplies it through
`with_overrides`. Guessing would be one line and would make every downstream
total a fiction, so the entries stay out until somebody can state them without
hedging. The same reasoning applies to the Anthropic table, where the sticker
price is used rather than the introductory rate, because overstating a discount
with an expiry produces a report that silently becomes wrong on a known date.

**Running inside HAL rather than beside it.** The Holistic Agent Leaderboard
([arXiv:2510.11977](https://arxiv.org/abs/2510.11977)) already does
cost-controlled evaluation with accuracy-cost Pareto frontiers, and its scaffold
axis is this project's ablation axis. The claim that nobody can compare these
systems honestly was true of the twenty RLM repositories surveyed and false of
the wider field. The right move is to run inside HAL and reframe the
contribution as an honest ablation of an RLM scaffold rather than as a harness.
That means a HAL-compatible adapter over `Solver`, and it means the harness here
becomes the offline, deterministic layer rather than the publication path.

**A CoT self-consistency baseline solver.** Required by the evaluation protocol
and not written. Depth zero alone is the wrong control for the question of
whether the apparatus is worth building, because chain-of-thought with
self-consistency beat automatically designed multi-agent systems at under a
tenth of their cost ([arXiv:2606.13003](https://arxiv.org/abs/2606.13003)). It
is a `Solver` implementation, so it does not touch the runtime, but it has to
produce a `Run` with a depth-zero attempt like everything else, which is a small
design question in itself.

**Contamination-resistant perturbation of the corpus.** The generator controls
its own corpus, so entity and value perturbation with structure held fixed is
cheap ([arXiv:2605.19999](https://arxiv.org/abs/2605.19999) argues that is what
contamination resistance means). Not started.

**No published benchmark result.** There is none, and there will not be one
without a depth-zero row beside it. This is not a scheduling statement. Given
what is listed above, in particular that the recursive path is not currently
reachable with a real sandbox and that no baseline beyond depth zero exists,
publishing a number now would be publishing a number about a system that is not
finished.

## Ordering

The first three items under "in flux" gate everything, because until they are
resolved no end-to-end run happens at all. After that the order is: the missing
refund in the budget lifecycle, because it makes the ceilings mean what they
say; the microVM backend, because it is what makes the threat model hold against
a real corpus; the CoT baseline, because without it no result is publishable
under the protocol in [EVALUATION.md](EVALUATION.md); and then HAL.

Parallel fan-out, the fan-out estimator and progressive degradation are the
interesting work and all three come after the boring work above them.
