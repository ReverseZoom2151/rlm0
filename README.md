<h1 align="center">rlm0</h1>

<p align="center"><strong>A Recursive Language Model that runs depth 0 on every task, so recursion always has a baseline to beat.</strong></p>

<p align="center"><sub>Clean-room implementation of <a href="https://arxiv.org/abs/2512.24601">Zhang, Kraska &amp; Khattab (2025)</a>. Tree-wide budget reservation, prefix caching across sub-calls, and a network-isolated sandbox.</sub></p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" />
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+" />
  <img src="https://img.shields.io/badge/typed-mypy%20strict-blue.svg" alt="mypy strict" />
  <a href="https://arxiv.org/abs/2512.24601"><img src="https://img.shields.io/badge/arXiv-2512.24601-b31b1b.svg" alt="arXiv" /></a>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.24601">Paper</a> &bull;
  <a href="https://github.com/alexzhang13/rlm">Reference implementation</a> &bull;
  <a href="docs/RELATED_WORK.md">Related work</a> &bull;
  <a href="#what-a-run-looks-like">Example run</a>
</p>

A clean-room implementation of the paper [*Recursive Language
Models*](https://arxiv.org/abs/2512.24601) (Zhang, Kraska and Khattab, MIT
CSAIL, 2025), built around the finding the paper reports and the field mostly
skips: the REPL is what handles length, and recursion is a task-conditional
accelerator on top of it.

This is not a reproduction. There are no trained weights here, no published
benchmark numbers, and no claim that any figure from the paper has been
reproduced.

## What is an RLM?

Language models degrade well before they hit their context limit. A Recursive
Language Model sidesteps that by keeping the prompt out of the window
entirely: the text is bound to a variable in a persistent Python REPL, and the
model writes code to slice, search and partition it, sending only selected
pieces to sub-LLM calls.

## What rlm0 adds

Depth 0 is the same REPL loop with sub-calls switched off. rlm0 runs it first
on every task and escalates only when it fails. Both attempts land in one
record, so every run reports what recursion cost and whether it helped.

```text
  task
    │
    ▼
  depth 0 ── REPL, context in a variable, sub-calls off
    │
    ├── answered ────────────────────────►  stop        verdict: not_attempted
    │
    └── failed
          │
          ▼
        depth 1 ── same REPL, same prompt, sub-calls on
          │
          ├── answered ────────────────────►  stop      verdict: helped
          │
          └── failed ──────────────────────►  escalate, or wind down

  every call tagged with role and depth   ·   budget reserved before dispatch
  both attempts kept in one Run           ·   cost is None when unpriceable
```

## Why this exists

One table, from an independent reproduction
([arXiv:2603.02615](https://arxiv.org/abs/2603.02615)), explains the whole
project.

| | base model | RLM depth 1 | RLM depth 2 |
| --- | --- | --- | --- |
| S-NIAH (lookup) | **100%**, 3.6s | 85%, 89.3s | 70%, 344.5s |
| OOLONG (aggregation) | **0%** | **42.1%** | 33.7% |

Recursion destroys lookup and creates aggregation out of nothing, in the same
experiment. A strong base model is actively harmed: Kimi K2 falls from 86.6 to
60. That paper's own recommendation is that future work should build better
stopping mechanisms into the REPL.

The rest of the evidence points the same way. Apple's group reports that
recursion is not the primary driver of RLM performance and hurts inside the
native window. Prime Intellect's own evaluation shows a clear regression on
math-python. The best-engineered open implementation reports, in its benchmark
notes, that the model never once chose to recurse in any successful run.

So the technique has a real but narrow sweet spot, and the engineering problem
is knowing which side of it you are on before paying for the wrong answer.

## What is and isn't new here

Being straight about this, because the field moved fast and most of it is
taken. RLM has been absorbed into
[Code as Agent Harness](https://arxiv.org/abs/2605.18747), generalised by
[Recursive Agent Harnesses](https://arxiv.org/abs/2606.13643), and shipped by
Anthropic as Dynamic Workflows. Binding a context to a REPL variable is prior
art ([CaveAgent](https://arxiv.org/abs/2601.01569), January 2026). Budget
reserve-and-reconcile was published with a formal artifact in
[June 2026](https://arxiv.org/abs/2606.04056), and budget conservation under
delegation was proved in
[January](https://arxiv.org/abs/2601.08815). Cost-controlled evaluation with
Pareto reporting already exists as [HAL](https://arxiv.org/abs/2510.11977) at
ICLR 2026.

What appears to be unclaimed, after three literature sweeps:

- **The paired control.** No published runtime runs the no-recursion attempt
  first and keeps both results together.
- **Graceful degradation under a binding budget.** A targeted search found
  nothing. Existing work bounds or aborts; none winds down.
- **A tight fan-out estimator.** The published budget work concedes 4 to 6x
  static over-reservation. A runtime that knows its own fan-out width should
  beat that.
- **The threat model.** Nothing found addresses the hazard specific to this
  architecture, where the untrusted context and the code-writing model share
  one interpreter, so text under analysis can become the program.

Everything above is sourced in [docs/RELATED_WORK.md](docs/RELATED_WORK.md),
including the work that argues against this design.

## Install

```bash
git clone https://github.com/ReverseZoom2151/rlm0.git
cd rlm0
pip install -e ".[dev]"
pytest
```

Python 3.11 or newer. The tests run against fakes, so you need no API key and
no model is called to develop against it.

## Quick start

Install the provider extra you intend to use, then confirm the sandbox before
giving it any context:

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=...
rlm0 sandbox --require docker
rlm0 run "Which supplier appears in both reports?" --context reports/ \
  --provider anthropic --model claude-sonnet-5 --max-usd 0.50 --max-calls 20
```

`.[openai]` and `.[gemini]` install the other supported provider SDKs. Use
`rlm0 eval` for the self-checking synthetic corpus, `rlm0 cost` before a run,
and `rlm0 --help` for all options. Public benchmark adapters are API-first and
require their pinned data on disk; see [docs/EVALUATION.md](docs/EVALUATION.md).

## What a run looks like

```text
task: which supplier appears in both the Q3 audit and the incident log
  depth<=0  iterations_exhausted    $0.0100     2.0s    1 calls (0 sub)
  depth<=1  answered                $0.0130     2.5s    2 calls (1 sub)
  budget: max $0.50, 60s, 20 calls
  recursion helped: $0.0130, +2.5s, +1 sub-calls
```

That last line is arithmetic over the two attempts above it. When depth zero
answers on its own the run stops there and says recursion was never needed,
which on retrieval work is most of the time.

A `Run` will not construct without its depth-zero attempt, without every model
call tagged with the role and depth that made it, or without the budget it ran
under. You can waive the control, because a rule with no escape hatch gets
worked around instead of followed, but a waiver names an approver, needs a real
reason, and leaves the verdict reading `untested` from then on.

## What it refuses to do

Cost comes back as `None` when a call could not be priced, never as zero.
Several implementations I read total unpriced calls as zero, which means their
spending limits can never fire. A limit that quietly never triggers is worse
than no limit, because people believe it.

Budget is reserved before the call goes out. Counting afterwards bounds
nothing, and a fan-out that checks between dispatches has already sent half of
them.

The sandbox keeps the network shut and the API keys outside it. Your context is
attacker-controlled text sharing an interpreter with a model that writes code,
which puts prompt injection one step from code execution. Sub-calls get
serviced by the host across that boundary, so nothing inside ever needs a
socket or a key.

## How this compares

| Approach | Where the text lives | Decides at runtime | Cost shape | Best for |
| --- | --- | --- | --- | --- |
| Single-shot | In the window | No | Flat, one pass of input | Small inputs |
| Long-context cram | In the window | No | Full text on every call | It fits and cost is no object |
| RAG | Retrieved chunks | No | Cheap retrieval plus one call | Factoid lookup over a corpus |
| Map-reduce | Fixed chunking | No | Linear in chunks | Summarisation |
| **RLM at depth 0** | REPL variable | Yes | Root calls only, no fan-out | Search and retrieval over huge text |
| **RLM at depth 1+** | REPL variable | Yes | Root plus a fan-out per turn | Aggregation across the whole text |

Most implementations treat the last two rows as one thing. They are not. Depth
0 already handles text far past the window and wins outright on retrieval, and
the fan-out is what costs money. Choosing between them per task is the job
rlm0 exists to do.

## Limitations

When recursion was always going to be needed, running depth 0 first costs one
extra attempt. That is the price of the baseline, it is real, and the run
record shows it.

A cost ceiling cannot be enforced against calls nobody can price, so the
budget fails closed when it meets one and says why. If you point rlm0 at a
model with no entry in the price table and set a spending limit, it will stop
rather than guess.

Real isolation needs Docker. The subprocess fallback keeps a runaway loop from
taking out the orchestrator, but it runs as you, with your filesystem, and it
is not a boundary against a hostile context.

The control cannot be depth 0 alone. Chain-of-thought with self-consistency
beat automatically-designed multi-agent systems at under 10 percent of their
cost ([arXiv:2606.13003](https://arxiv.org/abs/2606.13003)), so it has to be in
the table at matched cost, well elicited rather than as a strawman. Two
non-recursive methods also have to be beaten on their own ground:
[ARC](https://arxiv.org/abs/2607.25066) on lookup and
[VISTA](https://arxiv.org/abs/2606.30005) on LOCA-Bench.

Real isolation should be a microVM. Shared-kernel containers are no longer
considered adequate for model-written code, so the Docker backend here is a
floor and not a ceiling.

There is no benchmark number yet, and none will be published here without a
depth-0 row beside it.

## Where it stands

The integrated runtime, Docker and subprocess backends, microVM backend,
Anthropic/OpenAI/Gemini clients, evaluation harness, benchmark adapters, CLI,
CI, and documentation are built and tested. There is no published model result
yet. Running one requires a provider credential plus the pinned benchmark data,
and any reported number must include its depth-0 control.

## Prior work

The authors' own implementation is at
[alexzhang13/rlm](https://github.com/alexzhang13/rlm), and it is the reference
for this paradigm. rlm0 shares no code with it.

Reading twenty open implementations end to end shaped most of the design
decisions here, including several taken from repositories that solved a problem
better than the reference does. Sources, credits, and the papers that argue
against this design are in [docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## License and citation

Released under the [MIT License](LICENSE). If this work informs research, cite
the original paper rather than this repository:

```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025},
  url={https://arxiv.org/abs/2512.24601}
}
```
