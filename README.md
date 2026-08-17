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
  <a href="docs/PRIOR_ART.md">Prior art</a> &bull;
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

The headline numbers are real. On OOLONG-Pairs a frontier model scores 0.1 F1
and the recursive version reaches 76.0. At six million input tokens, where no
context window helps at all, questions get answered for under a dollar.

Underneath sits a quieter result that the paper states plainly and few people
repeat. The REPL is what handles length. Recursion helps on dense aggregation,
and on ordinary retrieval depth zero simply wins. A reproduction found depth
two making things worse, with runtime going from 3.6 seconds to 344. Another
group found recursion hurting outright once the text fits in the window.

The best-engineered open implementation I could find reports, in its own
benchmark notes, that the model never recursed in any successful run.
Recursion was configured, available, and never chosen.

So this is a technique with a real but narrow sweet spot, and the engineering
problem is knowing which side of it you are on before you pay for the wrong
answer.

## Install

```bash
git clone https://github.com/ReverseZoom2151/rlm0.git
cd rlm0
pip install -e ".[dev]"
pytest
```

Python 3.11 or newer. The tests run against fakes, so you need no API key and
no model is called to develop against it.

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

There is no benchmark number yet, and none will be published here without a
depth-0 row beside it.

## Where it stands

The contract and the seams are built and tested. The runtime, sandbox,
providers and harness are in progress.

## Prior work

The authors' own implementation is at
[alexzhang13/rlm](https://github.com/alexzhang13/rlm), and it is the reference
for this paradigm. rlm0 shares no code with it.

Reading twenty open implementations end to end shaped most of the design
decisions here, including several taken directly from repositories that solved
a problem better than the reference does. Credits and sources are in
[docs/PRIOR_ART.md](docs/PRIOR_ART.md).

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
