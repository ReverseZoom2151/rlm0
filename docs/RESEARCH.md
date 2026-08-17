# Optional research APIs

`rlm0.research` is a local, library-first layer for experiments that do not
belong in the default serving path. The normal runtime still starts at depth 0
and may escalate to depth 1. Importing this package does not change that policy,
download data, call a provider, or publish a result.

The APIs exist to make exploratory variants inspectable. They preserve the
ordinary `Run` contract, keep configuration and data identities explicit, and
leave external validation separate from implementation.

## Records, replay, and artifacts

`ResearchRun`, `ResearchTrial`, and `ResearchStage` are immutable records for
multi-run experiments. A research record carries a depth-0 control, a declared
budget, fingerprints for configuration and budget, and contiguous stage
metadata. Every trial must name the same task as that control and use the same
budget fingerprint. A strategy cannot use a flexible experiment structure to
discard or weaken the baseline required by the main runtime.

`EventLog` writes append-only JSONL records with a SHA-256 hash chain and
flushes each appended record. A `research_complete` record seals the log:
further appends, a chain with events after completion, or a chain lacking a
final completion record are rejected. `read_events` validates the complete
chain, and `replay` rebuilds the submitted `ResearchRun` without provider
calls. This is integrity checking and deterministic record reconstruction, not
a replay of model execution.

`ArtifactStore` is a local content-addressed store. It writes objects
atomically, bounds each object and the whole store, verifies bytes on read, and
passes `ArtifactRef` values between stages. An artifact reference is a digest,
size, and media type, never a host path mounted into an untrusted environment.

## Context preparation

The screen API is deliberately conservative. Checks return `safe`, `unsafe`,
or `unknown`; malformed check output, a missing check, and a checker exception
all produce `unknown`. It is a research seam inspired by RLM-JB-style
screening, not a security guarantee or a prompt-injection classifier.

PEEK-style maps divide a context into bounded sections and store short,
caller-supplied summaries. A map identity includes the context hash, map
builder version, model, prompt fingerprint, schema version, and limits.
`MapStore` writes private map files atomically and rejects an identity mismatch.
The package does not provide a provider-backed summariser, so the caller owns
model selection, prompt construction, and any cost.

## Search and recombination

SRLM search accepts an injected candidate factory and makes several independent
depth-0 `Run` records. It selects an exact-answer plurality, then breaks ties
by fewer calls, lower priced cost, lower wall time, and declaration order. It
does not infer confidence from model prose, fabricate a control, or dispatch a
provider itself.

Verifier-backed recombination accepts candidate answers with cited evidence and
provenance, then calls an injected verifier. The verifier sees the candidate,
not hidden gold labels. A missing or failing verifier result is rejected; the
module does not turn a verifier into a universal judge for open-ended tasks.

## Bounded multi-stage variants

Chained RLM executes fresh roots supplied by an injected factory. Each root
receives the original task and context plus only a bounded `SUMMARY`,
`BLACKBOARD`, `NEXT`, and content-addressed artifact metadata from the prior
stage. Strict `RLM0_CHAIN_V1` handoffs require an artifact update, including at
the final stage. Each root remains a normal `Run` with its own depth-0 control.
The module does not claim a reproduction or validation of the Chained RLM
paper.

The agent-harness API executes a declared `HarnessNode` tree through an
injected executor. It validates maximum depth, node count, children per node,
and sibling concurrency before the first executor call. A single semaphore
bounds concrete executor calls across every depth of the tree; records still
return in declared order. The executor gets artifact references only, not
credentials or host paths. This is a bounded plan executor, not an
implementation or reproduction of a model-generated, asynchronous Recursive
Agent Harness script.

## Prompt optimisation

RLMOpt-inspired APIs create immutable prompt candidates with typed sections.
They pin train, validation, and test fingerprints, permit selection on
validation bytes only, compute a Pareto frontier, and use uncertainty-aware
regression gates. An optional cost estimator creates a permit before a candidate
can be evaluated; the permit is settled against actual cost or refunded after a
failure. Candidates that cannot be reserved are recorded as refused without
calling an evaluator. `evolve` is deterministic given its seed. It has no
operation that applies a selected candidate to the shipping runtime: promoting
a prompt remains a reviewed release change.

## Local anomaly evaluation

`rlm0.benchmarks.anomalyxl.AnomalyXL` is a narrow local adapter for
time-series anomaly questions. A caller supplies `manifest.json` and `data.jsonl`;
the manifest pins the data hash, revision, split, and row count. Predictions are
one strict JSON object, with no prose recovery. The adapter reports task-shaped
metrics for localization, classification with evidence, magnitude,
multi-channel events, and lead-lag. Lead-lag labels must state a positive
series length, and their lag cannot exceed it; incomplete local labels are
rejected before scoring.

It is a clean-room local format inspired by TimeRLM and AnomalyXL. It does not
vendor, fetch, score, or claim parity with the official benchmark. It becomes
useful for a public comparison only after its local data and scorer are made
available and independently reviewed.

## What remains external

These APIs have unit and integration coverage against local data and fakes. No
public provider run, benchmark number, cache saving, sandbox security result,
or paper reproduction follows from that. See [EVALUATION.md](EVALUATION.md) for
the release gates and [RELATED_WORK.md](RELATED_WORK.md) for the sources that
informed these optional strategies.
