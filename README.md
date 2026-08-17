<h1 align="center">rlm0</h1>

<p align="center"><strong>A cost-bounded Recursive Language Model runtime with depth-0 controls</strong></p>

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
  <a href="#research-lineage">Research lineage</a> &bull;
  <a href="docs/CITATIONS.md">Citations</a> &bull;
  <a href="https://github.com/alexzhang13/rlm">Reference implementation</a>
</p>

rlm0 runs Recursive Language Models over long context through a sandboxed Python
REPL. The source stays in the environment as a variable. The model writes code
to search it, compute over it, and call submodels on selected slices.

The default policy starts each task at depth 0 with subcalls disabled. If that
attempt does not answer, the same runtime may try depth 1. Both attempts use one
budget and stay together in the final `Run`, which records what recursion cost
and whether it helped.

> [!NOTE]
> This is a clean-room RLM implementation, not a reproduction of the paper.
> The repository ships no model weights and makes no benchmark claim yet.

## Why rlm0

| Capability | What rlm0 records or enforces |
| --- | --- |
| Paired depth control | Keeps the depth-0 attempt beside every recursive attempt in the same run |
| Tree-wide budget | Reserves, settles, and refunds calls across the complete recursion tree |
| Sandboxed execution | Keeps provider credentials on the host while model-authored Python runs in the selected backend |
| Deterministic fanout | Reserves a batch before dispatch, bounds parallel work, and returns results in declared order |
| Reproducible evaluation | Matches samples and cost, scores cited evidence, and refuses reports without depth 0 |

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

Depth 1 is the normal ceiling. Higher depths require an experimental policy and
the `--experimental-depth` flag.

## Quick start

Python 3.11 or newer is required. rlm0 is not published on PyPI, so install it
from a checkout. Choose the extra for the provider you plan to use.

```bash
git clone https://github.com/ReverseZoom2151/rlm0.git
cd rlm0
pip install ".[anthropic]"  # or .[openai] / .[gemini]
```

Inspect the local setup before spending anything:

```bash
rlm0 doctor
rlm0 benchmark --list
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

For development, install the test tools alongside a provider extra:

```bash
pip install -e ".[dev,anthropic]"
python -m pytest
```

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

[`rlm0.research`](src/rlm0/research/) is a local, API-first layer for controlled
experiments. Importing it does not change the normal depth-0 then depth-1 policy.

| API | Purpose |
| --- | --- |
| `contracts`, `events`, `artifacts` | Immutable research records, sealed hash-chained replay logs, and bounded content-addressed storage |
| `screen`, `peek` | Conservative tri-state context checks and reusable context maps |
| `srlm`, `verifier` | Nonrecursive candidate search and evidence-carrying recombination |
| `chained` | Fresh roots connected by bounded handoffs and durable artifacts |
| `agent_harness` | Declared agent trees with one concurrency bound across every depth |
| `optimize` | Split-safe prompt evolution with pre-dispatch evaluation permits |

The local `anomalyxl-local` adapter accepts a caller-supplied, hash-locked
JSONL dataset for precise anomaly tasks. TimeRLM and AnomalyXL informed its task
shapes. The adapter does not download the official dataset or claim scorer
parity. [RESEARCH.md](docs/RESEARCH.md) documents each contract and limit.

## Research lineage

rlm0 shares no code with these projects. The papers below introduced or tested
the methods that correspond to implemented modules.

| Paper | What it informs here |
| --- | --- |
| [Recursive Language Models](https://arxiv.org/abs/2512.24601) | Core REPL and subcall runtime |
| [Think, But Don't Overthink](https://arxiv.org/abs/2603.02615) | Depth-0-first policy and the experimental ceiling above depth 1 |
| [Recursive Language Models Meet Uncertainty](https://arxiv.org/abs/2603.15653) | SRLM candidate search and the nonrecursive comparison |
| [PEEK](https://arxiv.org/abs/2605.19932) | Reusable context maps |
| [Chained Recursive Language Models](https://arxiv.org/abs/2608.05124) | Fresh-root handoffs, blackboards, and artifacts |
| [Recursive Agent Harnesses](https://arxiv.org/abs/2606.13643) | Declared recursive agent trees |
| [RLMOpt](https://arxiv.org/abs/2608.10471) | Prompt evolution and deterministic selection guards |
| [TimeRLM](https://arxiv.org/abs/2608.03391) | Precise time-series tasks and the local AnomalyXL-style adapter |
| [Recursive Language Models for Jailbreak Detection](https://arxiv.org/abs/2602.16520) | Conservative context screening |

Full BibTeX entries are in [CITATIONS.md](docs/CITATIONS.md). The broader
comparison, including work that argues against parts of this design, is in
[RELATED_WORK.md](docs/RELATED_WORK.md).

## Project status

rlm0 is alpha software. The runtime, CLI, provider adapters, Docker sandbox,
experimental microVM path, synthetic harness, public benchmark adapters, and
optional research APIs are implemented and tested on Python 3.11 through 3.13.

The remaining work needs external systems or data: live provider measurements,
public benchmark runs, microVM testing on KVM, and HAL integration. These items
are tracked in [docs/ROADMAP.md](docs/ROADMAP.md).

## Citation

Use [`CITATION.cff`](CITATION.cff) to cite rlm0 itself. The core runtime
implements the method introduced by Zhang, Kraska, and Khattab, so research
that uses it should also cite the original RLM paper:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025},
  url={https://arxiv.org/abs/2512.24601}
}
```

If you use an optional research module, add the corresponding paper from
[CITATIONS.md](docs/CITATIONS.md).

## License

rlm0 is released under the [MIT License](LICENSE).

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md).
