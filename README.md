<h1 align="center">rlm0</h1>

<p align="center"><strong>A Recursive Language Model Runtime That Runs the Control First</strong></p>

A recursive language model answers questions about text far larger than its
context window by holding that text in a REPL and writing code to work over it,
calling itself on the pieces. The idea works. What nobody ships is any way to
tell whether the recursion was the part that helped, or what the trajectory
cost when it did not.

rlm0 attempts depth zero first, always. The same environment, the same prompt,
sub-calls unavailable. It escalates only when that fails. So every run carries
the counterfactual, and whether recursion earned its cost becomes something the
run measured rather than something the README claims.

## Why this exists

The published result is real and the numbers are large. On OOLONG-Pairs a
frontier model scores 0.1 F1 and the recursive version reaches 76.0. At six
million input tokens a task that no context window can hold at all gets
answered for under a dollar.

Underneath those headlines is a quieter finding that the paper states plainly
and almost nobody repeats. The REPL is what handles length; recursion helps on
information-dense aggregation. On retrieval work, depth zero wins. An
independent reproduction found depth two degrading rather than improving, with
execution time going from 3.6 seconds to 344. A separate critique showed
recursion actively hurting inside the native window.

The best-engineered open implementation of this idea reports, in its own
benchmark notes, that every successful run used only the root REPL and never
recursed at all. Recursion was configured, available, and never chosen.

So the honest position is that this is a technique with a real and narrow
regime, and the interesting engineering problem is knowing which regime you are
in before you pay for the wrong one.

## Install

```bash
git clone https://github.com/ReverseZoom2151/rlm0.git
cd rlm0
pip install -e ".[dev]"
pytest
```

Python 3.11 or newer. The test suite runs against fakes, so nothing needs an
API key and no model is called to develop against it.

## What a run looks like

```text
task: which supplier appears in both the Q3 audit and the incident log
  depth<=0  iterations_exhausted    $0.0100     2.0s    1 calls (0 sub)
  depth<=1  answered                $0.0130     2.5s    2 calls (1 sub)
  budget: max $0.50, 60s, 20 calls
  recursion helped: $0.0130, +2.5s, +1 sub-calls
```

That last line is the point. It is computed from the two attempts the run
already holds, not asserted. When depth zero answers, the run stops there and
reports that recursion was never needed, which is the common case on retrieval
work and the case existing systems pay for anyway.

A `Run` cannot be constructed without its depth-zero control, without every
model call attributed to the role and depth that issued it, or without the
budget it executed under. The control can be waived, because a rule with no
escape gets bypassed rather than obeyed, but a waiver names an approver, needs
a reason that is actually a reason, and the run reports its verdict as
untested forever after.

## What it refuses to do

Cost is `None` rather than zero when a call could not be priced. Several
surveyed implementations accumulate unpriced calls as zero, which means their
budget ceilings can never fire, and a ceiling that silently never triggers is
worse than no ceiling because it is believed.

Budget is reserved before dispatch rather than counted after. A fan-out that
checks a counter between calls has already landed half of them.

The sandbox keeps the network closed and the credentials outside it. The
context is attacker-controlled text sharing an interpreter with a model that
writes code, so injection and execution are one step apart by construction.
Sub-calls are serviced by the host across the boundary, which is why the
sandbox never needs a socket or a key.

## Where it stands

The contract and the seams are built and tested. The runtime, the sandbox, the
providers and the harness are in progress. No benchmark number exists yet,
honest or otherwise, and none will be published without a depth-zero row beside
it.

## Prior work

The idea is from [Recursive Language Models](https://arxiv.org/abs/2512.24601)
by Alex L. Zhang, Tim Kraska and Omar Khattab at MIT CSAIL, whose reference
implementation is at [alexzhang13/rlm](https://github.com/alexzhang13/rlm).
This is a clean-room implementation and shares no code with it.

The design here was shaped by reading twenty open implementations end to end.
Credited ideas and where they came from are in
[docs/PRIOR_ART.md](docs/PRIOR_ART.md).

## License

[MIT](LICENSE).
