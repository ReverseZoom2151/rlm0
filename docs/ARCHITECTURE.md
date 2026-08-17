# Architecture

rlm0 is built around one execution loop. The loop first runs with sub-calls
disabled, then may run at a deeper bound using the same prompt assembly,
parsing, observation formatting, sandbox interface, provider interface, and
budget. This keeps the depth-zero attempt useful as a control for recursion.

## Execution path

```text
CLI or library caller
        |
        v
assembly.build_rlm
        |
        +--> provider client: Anthropic, OpenAI, Gemini, or fake
        +--> sandbox: Docker, experimental microVM, or subprocess
        +--> RunBudget: reserve, settle, release
        +--> prompt, parser, and observation bridge
        |
        v
RLM.complete
        |
        +--> depth 0 attempt
        +--> policy decision
        +--> optional deeper attempt
        |
        v
Run record + evaluation harness
```

[`assembly.py`](../src/rlm0/assembly.py) is intentionally the only concrete
composition point. The runtime itself depends on Protocols from
[`ports.py`](../src/rlm0/ports.py), so tests and competing solvers can replace
providers, sandboxes, budgets, or policies without inheriting a base class.

## Run contract

[`run.py`](../src/rlm0/run.py) holds the immutable public record:

- The first ordinary attempt must be depth zero.
- Every call records its role and depth.
- A run records its budget summary and propagates an unpriced cost as `None`.
- An attempt cannot claim a normal answer after a terminal stop condition.

`BaselineWaiver` exists for imported or unusual records. It does not let the
assembled runtime skip the control and it permanently labels the record
untested.

The run record measures whether a deeper attempt produced an answer after depth
zero did not. It does not establish correctness. The evaluation harness owns
ground truth and evidence grading.

## Runtime and policy

[`runtime.py`](../src/rlm0/runtime.py) drives each attempt through bounded
turns. A model may return code, a final directive, or neither. Code executes in
the sandbox, bounded output becomes an observation, and malformed turns receive
another bounded turn.

The default escalating policy begins at depth zero and can request a deeper
attempt only after an unsuccessful result and a budget check. `Never` and
`Fixed` are useful evaluation policies. Depth 1 is the ordinary ceiling;
deeper recursion requires an explicit experimental flag in both the policy and
the assembled runtime.

The runtime reserves a finalization call before a turn that may need it. It
settles calls after provider responses and releases reservations that were not
used. On a bound refusal it asks for a bounded wind-down instead of throwing
away the trajectory. A refused batch is reduced to the largest reservable
prefix, with the deferred suffix made explicit to the model.

The runtime supports `rlm_batch([[query, handle], ...])` for independent child
questions. It reserves the complete batch before dispatch, uses a bounded worker
pool, lets one child warm the cache prefix, preserves input order in its results,
and returns unused initial holds. If the full batch cannot be reserved, it
reduces width in guest order and returns explicit deferred entries for the
suffix. Ordinary `llm_query` remains sequential. This
distinction matters because a cold parallel fan-out can duplicate prefix-cache
writes.

## Prompt, parser, and observations

[`prompt.py`](../src/rlm0/prompt.py) builds named prompt sections. The depth
zero and recursive variants differ only in the sections that describe
sub-calls. [`parse.py`](../src/rlm0/parse.py) recognizes code, `FINAL(...)`,
and `FINAL_VAR(...)` while ignoring quoted or fenced instructions. The assembly
bridge adapts these functional modules to the runtime Protocols.

`RLM0_FINAL_V1` is the reportable completion protocol. It carries a version,
answer, evidence list, and optional answer artifact. `FINAL(...)` and
`FINAL_VAR(...)` remain readable for compatibility, but each is recorded as a
recovered completion so an evaluation can distinguish it from the strict path.

[`observation.py`](../src/rlm0/observation.py) caps output before it returns to
the model. The cap prevents the sandbox from becoming an unbounded delivery
channel for the original context.

## Sandboxes and host calls

The sandbox channel uses newline-delimited JSON over standard input/output.
The host never unpickles sandbox output. API credentials remain outside the
sandbox, and sub-calls are serviced by the host over the existing channel.

- `DockerSandbox` is the practical isolated backend: no network, constrained
  filesystem, user, capabilities, and resources.
- `MicroVMSandbox` is experimental. It uses Docker with a registered OCI
  microVM runtime, then probes for a guest kernel distinct from the host.
- `SubprocessSandbox` is for trusted local development only. It is not an
  isolation boundary.

The microVM backend has no demonstrated Podman or libkrun support in this
repository. It must not be described as such until its runtime discovery,
launch path, and live-backend tests exist. See [THREAT_MODEL.md](THREAT_MODEL.md).

## Providers and accounting

Provider clients turn native responses into shared usage records. Anthropic,
OpenAI, and Gemini are optional extras. Provider-reported usage is required;
the clients do not estimate usage from text. Pricing is an explicit table and
unknown prices remain unpriced.

Prefix-cache reporting also comes from provider responses. The Anthropic
adapter places stable cache boundaries, but this is not proof of a cache hit.
The cache-warming barrier applies to batch fan-out. It prevents a cold batch
from paying duplicate prefix writes before a cache entry exists.

## Evaluation

The harness has a deterministic synthetic corpus, evidence-aware grading, run
records, and report refusal rules. It includes direct, retrieval, and CoT
self-consistency baselines. OOLONG and RULER S-NIAH adapters are implemented
as library APIs with pinned-data requirements.

The harness rejects comparisons across different corpora, samples, or grading
policies, and requires a depth-zero row. The report layer can compute a paired
noise floor, but the CLI does not yet schedule replicates for it or provide HAL
integration.

## Optional research layer

[`research/`](../src/rlm0/research/) sits beside the serving runtime. Its
strategies receive or produce ordinary immutable `Run` records and add
experiment-only contracts around them. They do not replace `assembly.build_rlm`
or change the default depth policy.

```text
ordinary Run records
        |
        +--> ResearchRun / ResearchTrial / ResearchStage
        |         |
        |         +--> hash-chained JSONL event log --> validated record replay
        |         +--> bounded content-addressed artifacts
        |
        +--> optional strategies
                  |
                  +--> screen and PEEK-style context maps
                  +--> SRLM search and verifier recombination
                  +--> fresh-root chained handoffs
                  +--> declared bounded agent-harness plans
                  +--> split-safe prompt optimisation
```

`ResearchRun` keeps a depth-zero control and fingerprints its configuration and
budget. Its trials must use that control's task and budget identity. `EventLog`
is append-only and hash chained, then seals at completion; replay rejects a
missing or nonterminal completion and reconstructs persisted records without
rerunning a provider. `ArtifactStore` is atomic and bounded, and stages carry
artifact digests instead of exposing host paths to strategy executors.

The strategies are dependency-injected. SRLM receives candidate runs from a
factory, verifier recombination receives a verifier, Chained RLM receives fresh
roots, and the agent-harness path receives a declared plan plus an executor.
This keeps their budgets, provider choices, and sandbox assembly outside the
research contracts. The agent-harness executor applies one global concurrency
bound to work at every tree depth, while prompt optimisation reserves estimated
evaluation cost before dispatch. None of these APIs is a claimed paper
reproduction or a ready-made autonomous orchestration system. Details and API
limits are in [RESEARCH.md](RESEARCH.md).

[`benchmarks/anomalyxl.py`](../src/rlm0/benchmarks/anomalyxl.py) is similarly
separate from the published benchmark adapters. It accepts only a local,
manifest-hash-locked JSONL dataset and strict JSON predictions. It is informed
by TimeRLM and AnomalyXL task shapes, but does not download, vendor, or claim
official-data or scorer parity.

## Layer boundaries

`run.py` and `ports.py` define contracts. `budget.py`, `policy.py`, prompt,
parse, observation, providers, and sandboxes implement independent layers.
`runtime.py` consumes the ports. `assembly.py` composes concrete layers.
`harness/` evaluates a minimal solver interface instead of reaching into the
runtime. This boundary lets a non-rlm0 baseline produce a comparable `Run`.
