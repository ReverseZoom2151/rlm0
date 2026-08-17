<h1 align="center">rlm0</h1>

<p align="center"><strong>A Recursive Language Model that runs depth 0 on every task, so recursion always has a baseline to beat.</strong></p>

<p align="center"><sub>Clean-room implementation of <a href="https://arxiv.org/abs/2512.24601">Zhang, Kraska &amp; Khattab (2025)</a>. Tree-wide budget reservation, prefix caching across sub-calls, and a network-isolated sandbox.</sub></p>

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

## Where it stands

The contract and the seams are built and tested. The runtime, sandbox,
providers and harness are in progress. There is no benchmark number yet, and
there won't be one published without a depth-zero row next to it.

## Prior work

The idea comes from [Recursive Language
Models](https://arxiv.org/abs/2512.24601) by Alex L. Zhang, Tim Kraska and Omar
Khattab at MIT CSAIL. Their implementation is at
[alexzhang13/rlm](https://github.com/alexzhang13/rlm). This is a clean-room
build and shares no code with it.

Reading twenty open implementations end to end shaped most of the design
decisions here. Credits and sources are in
[docs/PRIOR_ART.md](docs/PRIOR_ART.md).

## License

[MIT](LICENSE).
