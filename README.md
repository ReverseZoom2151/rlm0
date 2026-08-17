<h1 align="center">rlm0</h1>

<p align="center"><strong>A bounded runtime for Recursive Language Models</strong></p>

<p align="center">
  <a href="https://github.com/ReverseZoom2151/rlm0/actions/workflows/ci.yml"><img src="https://github.com/ReverseZoom2151/rlm0/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-3776AB.svg" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/typed-mypy%20strict-2A6DB2.svg" alt="mypy strict" />
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-2EA44F.svg" alt="MIT License" /></a>
  <a href="https://arxiv.org/abs/2512.24601"><img src="https://img.shields.io/badge/arXiv-2512.24601-B31B1B.svg" alt="RLM paper" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> &bull;
  <a href="docs/ARCHITECTURE.md">Architecture</a> &bull;
  <a href="docs/EVALUATION.md">Evaluation</a> &bull;
  <a href="docs/RESEARCH.md">Research APIs</a> &bull;
  <a href="docs/RELATED_WORK.md">Related work</a> &bull;
  <a href="https://github.com/alexzhang13/rlm">Reference implementation</a>
</p>

rlm0 keeps long context in an isolated Python environment. A model can search,
compute, and call submodels over selected slices without loading the full source
into its prompt.

Every task starts at depth 0, with subcalls disabled. If that attempt does not
answer, the same runtime may try depth 1. Both attempts share one budget and
remain together in the final run record, so the cost and effect of recursion
are visible for that task.

> [!NOTE]
> This is a clean-room RLM implementation, not a reproduction of the paper.
> The repository ships no model weights and makes no benchmark claim yet.

## Why rlm0

| Capability | What it does |
| --- | --- |
| Depth 0 control | Runs the nonrecursive REPL loop first on every task |
| Shared budget | Reserves, settles, and refunds calls across the entire recursion tree |
| Isolated execution | Runs model-authored Python without placing provider keys in the sandbox |
| Bounded fanout | Reserves each batch before dispatch, warms its cache prefix, and preserves result order |
| Inspectable runs | Records every call with its model, role, depth, usage, cost, and wall time |
| Evidence-aware evaluation | Scores answers and cited evidence separately, with matched sample and cost checks |

## How it runs

```text
long context
     |
     v
+-------------------+       code, stdout, variables
| isolated Python   | <------------------------------+
| context = source  |                                |
+---------+---------+                                |
          |                                          |
          v                                          |
   depth 0 model ------------------------------------+
   subcalls disabled
          |
          +-- answered --------------------------> Run
          |
          +-- no answer
                 |
                 v
          depth 1 model
          subcalls available
                 |
                 +-------------------------------> Run

one run record, one budget, every call tagged by role and depth
```

Depth 1 is the normal ceiling. Higher depths require both an experimental
policy and the `--experimental-depth` flag.

## Quick start

Python 3.11 or newer is required.

```bash
git clone https://github.com/ReverseZoom2151/rlm0.git
cd rlm0
pip install -e ".[dev]"

python -m pytest
rlm0 doctor
rlm0 benchmark --list
```

Install the SDK for the provider you intend to use:

```bash
pip install -e ".[anthropic]"  # or .[openai] / .[gemini]
```

Check the execution boundary before sending a task:

```bash
rlm0 sandbox --require docker
```

Run against a file or directory of context:

```bash
export ANTHROPIC_API_KEY="..."

rlm0 run "Which supplier appears in both reports?" \
  --context reports/ \
  --provider anthropic \
  --model YOUR_MODEL \
  --max-usd 0.50 \
  --max-calls 20 \
  --record runs/suppliers.json
```

The final block printed by `rlm0 run` is always the run summary. It includes
the depth 0 attempt, the recursive attempt when one occurred, and the budget
used by each.

## CLI

| Command | Purpose |
| --- | --- |
| `rlm0 run` | Answer one task over local files or directories |
| `rlm0 eval` | Run the deterministic synthetic evaluation suite |
| `rlm0 benchmark` | Run a pinned local OOLONG, RULER S-NIAH, or AnomalyXL-style dataset |
| `rlm0 cost` | Estimate the worst case permitted by a configuration |
| `rlm0 sandbox` | Verify the requested execution backend |
| `rlm0 doctor` | Inspect providers, benchmarks, and local prerequisites without spending |
| `rlm0 research` | Validate and inspect local research records without rerunning providers |

The CLI downloads no benchmark data. `rlm0 benchmark --list` prints the pinned
source and revision expected by each adapter.

## Defaults

| Setting | Default |
| --- | --- |
| Sandbox | Docker |
| Policy | Escalate from depth 0 to depth 1 after an unsuccessful attempt |
| Maximum depth | 1 |
| REPL turns per attempt | 8 |
| Attempts per run | 4 |
| Budget | $2, 40 calls, 900 seconds |
| Unpriced calls under a USD ceiling | Refused after the first settlement |

Use `--unbounded` only when an unbounded run is intentional. The flag conflicts
with every budget ceiling, and the run record names the choice.

## Sandboxes

| Backend | Intended use | Security boundary |
| --- | --- | --- |
| Docker | Default for untrusted context | Container boundary with network disabled |
| MicroVM | Experimental use on a KVM host | Separate guest kernel through a registered Docker OCI runtime |
| Subprocess | Local development with context you wrote | None |

Provider credentials stay on the host. Host callbacks cross the sandbox over
JSON messages on standard input and output, so the guest needs neither a
network socket nor an API key.

The microVM backend currently supports Docker with a registered OCI runtime
such as Kata. The project does not claim Podman or libkrun support. See
[SECURITY.md](SECURITY.md) and the full [threat model](docs/THREAT_MODEL.md).

## Evaluation

The harness grades answer correctness and evidence support separately. It
refuses to compare runs that used different corpora, sample sets, or grading
policies, and it will not render a result table without a depth 0 row.

A public result must include:

- direct long-context prompting;
- depth 0 RLM;
- recursive RLM;
- chain-of-thought self-consistency at matched cost;
- a nonrecursive baseline suited to the task.

OOLONG and RULER S-NIAH adapters are available through `rlm0 benchmark`. The
data must be acquired separately and pinned by hash. No public result has been
run from this repository.

The complete protocol and negative-result policy are in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Optional research APIs

[`rlm0.research`](src/rlm0/research/) is a separate, local and API-first
layer for controlled research experiments. It does not alter the normal
depth-0 then depth-1 runtime or make any public benchmark result.

It provides immutable `ResearchRun` records, hash-chained JSONL replay logs
that seal once complete, and a bounded content-addressed artifact store. Every
trial must match its control's task and budget identity. On top of those contracts are
conservative context screening, PEEK-style maps, SRLM candidate search,
verifier-backed recombination, fresh-root Chained RLM handoffs, declared
bounded agent-harness plans, and split-safe prompt optimisation with
pre-dispatch evaluation permits.

The local `anomalyxl-local` adapter accepts a caller-supplied, hash-locked
JSONL dataset for precise anomaly tasks. It is informed by TimeRLM and
AnomalyXL task shapes; it neither downloads nor claims compatibility with the
official dataset or scorer. See [docs/RESEARCH.md](docs/RESEARCH.md) for the
contracts, limits, and intended use of each API.

## Project status

rlm0 is alpha software. The runtime, CLI, provider adapters, Docker sandbox,
experimental microVM path, synthetic harness, public benchmark adapters, and
optional research APIs are implemented and tested on Python 3.11 through 3.13.

The remaining work needs external systems or data: live provider measurements,
public benchmark runs, microVM testing on KVM, and HAL integration. These items
are tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

## Prior work

The RLM method comes from [*Recursive Language Models*](https://arxiv.org/abs/2512.24601)
by Alex L. Zhang, Tim Kraska, and Omar Khattab. Their
[reference implementation](https://github.com/alexzhang13/rlm) defines the
category and remains the general-purpose upstream project.

rlm0 is a clean-room implementation. Its design also reflects work in TimeRLM,
zigrlm, RLM_agent, Chained RLM, Recursive Agent Harnesses, RLMOpt, and
lambda-RLM. [RELATED_WORK.md](docs/RELATED_WORK.md) records the specific prior
art, evidence, and differences.

## License and citation

Released under the [MIT License](LICENSE). If this repository informs research,
cite the original RLM paper:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025},
  url={https://arxiv.org/abs/2512.24601}
}
```

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md).
