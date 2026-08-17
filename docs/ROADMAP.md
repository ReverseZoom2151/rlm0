# Roadmap

This file distinguishes completed work from work that remains. It is written
against the code in this repository, not against an earlier design document.

## Implemented foundation

- Runtime assembly that connects the prompt, parser, observer, budget, provider,
  and sandbox layers.
- A depth-zero first execution path with recursive escalation through the same
  runtime loop.
- Reservation, settlement, and release in the shared budget ledger.
- Docker, subprocess, and experimental OCI-microVM sandbox choices.
- Anthropic, OpenAI, and Gemini clients.
- A deterministic, evidence-aware synthetic corpus; direct, retrieval, and CoT
  self-consistency baselines; OOLONG and RULER S-NIAH adapters.
- A CLI, CI matrix, packaging metadata, security policy, and integration tests.
- Optional local research APIs: immutable `ResearchRun` records, hash-chained
  event logs and replay, a bounded content-addressed artifact store,
  conservative context screening, and PEEK-style context maps.
- Optional strategy APIs: SRLM candidate search, verifier-backed recombination,
  fresh-root Chained RLM handoffs, declared bounded agent-harness plans, and
  split-safe RLMOpt-style prompt evolution.
- A local, manifest-hash-locked AnomalyXL-style adapter for precise time-series
  anomaly tasks. It is not the official dataset or scorer.

## Next: make the public surface exact

1. Exercise the Docker plus OCI-runtime microVM backend on a real KVM host.
2. Keep the public microVM claim limited to Docker plus a registered OCI runtime
   until Podman/libkrun discovery, launch support, and live tests exist.
3. Keep the built-wheel smoke test current as the CLI gains commands.

## Next: evaluate batch fan-out under real providers

`rlm_batch` now atomically reserves a complete batch, lets one child warm the
prefix, dispatches a bounded worker pool, preserves result order, records the
estimation error, and refunds unused holds. What remains is live-provider
evidence: cache read/write diagnostics, cancellation behavior, and a measured
comparison against sequential `llm_query` at the same budget.

## Next: use the budget to degrade deliberately

The runtime reserves a finalization call, emits a model-visible advisory before
budget refusal, winds down after an actual budget refusal, and reduces a refused
batch to the largest reservable prefix. It still cannot cancel a provider call
already in flight. Add a recorded degradation ladder for partial evidence
gathering, deadlines, rate limits, and unpriced usage.

## Evaluation before results

No public benchmark number exists. Before publishing one:

1. Run direct prompting, depth-zero RLM, recursive RLM, and CoT
   self-consistency at matched cost.
2. Schedule the existing paired noise-floor calculation from the CLI, then add
   multiple sampling seeds, confidence intervals, and per-sample paired deltas.
3. Run the pinned OOLONG and RULER adapters, then evaluate a precise
   localization task. The local AnomalyXL-style adapter is ready for
   caller-supplied, hash-pinned data, but official TimeRLM/AnomalyXL data and
   scorer parity remain external validation work.
4. Add entity/value perturbations to the synthetic corpus.
5. Publish raw trajectories, manifests, costs, evidence scores, and negative
   results. Do not publish a headline without the depth-zero and strong
   nonrecursive rows beside it.

AGGBench and LOCA-Bench are intentionally not yet adapters; their blockers are
recorded in [`benchmarks/registry.py`](../src/rlm0/benchmarks/registry.py).

## Security and release work

- Exercise the microVM backend on a real KVM host and test the guest kernel,
  network, filesystem, capabilities, environment, and process limits live.
- Add adversarial tests for context injection, forged protocol frames,
  credential discovery, oversized output, filesystem access, and non-cooperative
  native code.
- Integrate the existing conservative RLM-JB-style screen into an explicit
  research command only after its policy, model, and false-positive costs are
  specified. Its parse failures already produce `unknown`, not `safe`.
- Expose the existing versioned, hash-chained event log and replay APIs through
  inspect/replay commands before the first external evaluation release.

## Optional research policies

These are implemented as separate, local APIs, not default runtime behavior:
Chained RLM-style fresh roots and blackboards, SRLM-style nonrecursive program
search, PEEK-style context maps, verifier-backed recombination for objectively
scored tasks, declared bounded agent-harness plans, and prompt optimisation.
They preserve ordinary run and reporting contracts but require a caller to
provide the concrete provider, sandbox, budget, data, and any external
validation. See [RESEARCH.md](RESEARCH.md).
