<h1 align="center">rlm0</h1>

<p align="center"><strong>Recursive Language Models with a depth-zero control, one run-wide budget, and isolated code execution.</strong></p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/typed-mypy%20strict-blue.svg" alt="mypy strict" />
  <a href="https://arxiv.org/abs/2512.24601"><img src="https://img.shields.io/badge/arXiv-2512.24601-b31b1b.svg" alt="RLM paper" /></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">Paper</a> &bull;
  <a href="https://github.com/alexzhang13/rlm">Reference implementation</a> &bull;
  <a href="docs/RELATED_WORK.md">Related work</a> &bull;
  <a href="docs/EVALUATION.md">Evaluation protocol</a>
</p>

rlm0 is a Python runtime for [Recursive Language Models](https://arxiv.org/abs/2512.24601).
It binds long context to a variable in a persistent Python environment, lets a
model inspect it with code, and can service focused sub-model calls without
putting the whole context in a prompt.

Each run starts with the same loop at depth 0, where sub-calls are unavailable.
If it escalates, the depth-zero and recursive attempts remain in one `Run`
record with call attribution, provider-reported usage, and one shared budget.

This is a clean-room implementation, not a reproduction of the RLM paper. It
ships no model weights and no published benchmark result.

## Why depth zero comes first

Recursion is not uniformly useful. The small independent reproduction
[*Think, But Don't Overthink*](https://arxiv.org/abs/2603.02615) found its base
model scored 100% on S-NIAH lookup in 3.6 seconds, while depth 1 scored 85% in
89.3 seconds. On OOLONG aggregation, the same study found the base model at 0%
and depth 1 at 42.1%. The study used 20 examples per condition, so treat those
numbers as motivation, not a general result.

rlm0 makes that trade-off visible per run:

```text
task
  |
  v
depth 0: persistent environment, sub-calls unavailable
  |
  +-- answered --> stop
  |
  +-- no answer --> depth 1+: same loop, sub-calls available
                         |
                         +-- answer, stop, or exhaust a bound

both attempts remain in one Run
all calls carry a role and depth
all reservations draw from one run-wide budget
```

## Quick start

Python 3.11 or newer is required. The test suite uses fakes and needs no API
key.

```bash
git clone https://github.com/ReverseZoom2151/rlm0.git
cd rlm0
pip install -e ".[dev,anthropic]"
python -m pytest

export ANTHROPIC_API_KEY=...
rlm0 sandbox --require docker
rlm0 run "Which supplier appears in both reports?" \
  --context reports/ --provider anthropic --model claude-sonnet-5 \
  --max-usd 0.50 --max-calls 20 --record run.json
```

`.[openai]` and `.[gemini]` install the other supported provider SDKs. Use
`rlm0 run --provider fake` to exercise the assembled runtime without contacting
a provider. `rlm0 eval` runs the deterministic synthetic corpus; public
benchmark adapters are library APIs and require their pinned data locally.

## Current capability status

| Capability | Status | Notes |
| --- | --- | --- |
| Depth-zero control and escalation | Implemented | One assembled runtime drives both attempts. |
| Run-wide reserve, settle, and release | Implemented | Unpriced provider usage fails closed under a USD ceiling. |
| Docker sandbox | Implemented | Network disabled; credentials remain on the host. |
| MicroVM sandbox | Experimental | Docker plus a registered Kata-style OCI runtime; it verifies a separate guest kernel. |
| Subprocess sandbox | Implemented, unsafe | Intended only for context you wrote yourself. |
| Anthropic, OpenAI, Gemini clients | Implemented | Usage comes from provider responses. |
| Evidence-aware synthetic evaluation | Implemented | Scores answer and cited evidence separately. |
| OOLONG and RULER S-NIAH adapters | Implemented | Data must be downloaded and pinned by the caller. |
| CoT self-consistency baseline | Implemented | Use it at matched cost in evaluation. |
| Parallel sub-call fan-out | Implemented | `rlm_batch` reserves the whole batch, warms its prefix, and preserves result order. |
| Cache-warming barrier and fan-out estimator | Implemented | Batch dispatch records the reservation estimate against observed use. |
| Published benchmark result | Not available | No result has been run or released. |

## Safety model

The context may be hostile text and the model writes code over it. Use Docker
or the experimental microVM backend for material you do not control. The
subprocess backend runs with your user, filesystem, and network access; it is
not a security boundary.

API credentials stay on the host. Sandbox messages use JSON framing, never
pickle, and host callbacks cross standard input/output rather than a network
socket. See [SECURITY.md](SECURITY.md) and the detailed
[threat model](docs/THREAT_MODEL.md).

The experimental microVM backend currently targets Docker with a registered OCI
microVM runtime such as Kata. Podman and libkrun are not supported claims until
they have dedicated discovery, launch, and live-backend tests.

## Evaluation

The harness records answer correctness and evidence support separately. It
refuses to render a comparison when rows use different corpora, samples, or
grading policies, and it requires a depth-zero row.

Before any public result, compare at least:

- Direct long-context prompting.
- Depth-zero RLM.
- Recursive RLM.
- CoT self-consistency at matched cost.
- A task-appropriate nonrecursive baseline.

The full protocol, data requirements, and negative-result policy are in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Related work and scope

The original RLM paper is the starting point, not the entire field. The design
also draws on TimeRLM's evidence-grounded long-context evaluation, zigrlm's
network-closed host callbacks and deterministic tracing, RLM_agent's state and
budget work, Chained RLM's fresh-root handoffs, Recursive Agent Harnesses,
RLMOpt, and lambda-RLM. These systems establish prior art for most individual
mechanisms. [RELATED_WORK.md](docs/RELATED_WORK.md) explains the differences,
including where their evidence is incomplete or their approach is a better fit.

## License and citation

Released under the [MIT License](LICENSE). If this repository informs research,
cite the original work:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025},
  url={https://arxiv.org/abs/2512.24601}
}
```
