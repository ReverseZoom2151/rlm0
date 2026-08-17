# Changelog

## Unreleased

- Added integrated runtime assembly, provider and sandbox selection, benchmark
  adapters, and an evidence-aware evaluation harness.
- Added bounded reservation, release, and wind-down behavior to the runtime.
- Added Gemini and experimental OCI-microVM backends, plus public OOLONG and
  RULER S-NIAH adapter APIs.
- Added direct, retrieval, and CoT self-consistency baselines to the local
  evidence-aware harness.
- Corrected public documentation to separate implemented behavior from prepared
  fan-out work and to credit related RLM implementations.
- Bound every research trial to its depth-zero control's task and budget
  identity, sealed completed research event logs, and bounded agent-harness
  execution across the full tree.
- Added pre-dispatch permits for prompt-optimisation evaluations and stricter
  local AnomalyXL lead-lag label validation.
- Extended the installed-wheel smoke test to research inspection commands.
