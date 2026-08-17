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
budget refusal, and winds down after an actual budget refusal. It still does
not narrow a planned batch before reserving it or cancel already-running
children. Add a recorded degradation ladder for width reduction, partial
evidence gathering, deadlines, rate limits, and unpriced usage.

## Evaluation before results

No public benchmark number exists. Before publishing one:

1. Run direct prompting, depth-zero RLM, recursive RLM, and CoT
   self-consistency at matched cost.
2. Add paired noise-floor computation, multiple sampling seeds, confidence
   intervals, and per-sample paired deltas.
3. Run the pinned OOLONG and RULER adapters, then evaluate a task with precise
   evidence localization such as TimeRLM's AnomalyXL if its data and scorer can
   be pinned.
4. Add entity/value perturbations to the synthetic corpus.
5. Publish raw trajectories, manifests, costs, evidence scores, and negative
   results. Do not publish a headline without the depth-zero and strong
   nonrecursive rows beside it.

AGGBench and LOCA-Bench are intentionally not yet adapters; their blockers are
recorded in [`benchmarks/registry.py`](../src/rlm0/benchmarks/registry.py).

## Security and release work

- Exercise the microVM backend on a real KVM host and test the guest kernel,
  network, filesystem, capabilities, environment, and process limits live.
- Pin the sandbox image by digest for reproducible security testing.
- Add adversarial tests for context injection, forged protocol frames,
  credential discovery, oversized output, filesystem access, and non-cooperative
  native code.
- Consider an optional RLM-JB-style context screen. Its parse failures must
  produce `unknown`, not `safe`.
- Add inspect/replay commands and a versioned JSONL event schema before the
  first external evaluation release.

## Optional research policies

These are separate policies, not default runtime behavior: Chained RLM-style
fresh roots and blackboards, SRLM-style nonrecursive program search, PEEK-style
context maps, verifier-backed recombination for objectively scored tasks, and
prompt optimization. Each must use the same run, budget, sandbox, and reporting
contracts as the default runtime.
