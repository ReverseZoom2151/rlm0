# Threat model

## The hazard

In an ordinary coding agent, injected content arrives through a tool result
into an agent that then decides what to do. The corpus and the program are
different things in different places, and the question is whether the agent can
be talked into acting on what it read.

A recursive language model does not have that separation. The context is bound
to a variable in the same interpreter the model writes code into. The attacker's
text and the model's program occupy one address space by construction. Prompt
injection and code execution are not two steps here; they are one step apart at
most, and on a bad day they are the same step, because the model is writing
Python that reads a string the attacker wrote and the model has been told to
find out what that string says.

This is not a variant of the injection problem. It is a different placement of
it. Three literature sweeps for this project found nothing addressing it
directly. The sources are in [RELATED_WORK.md](RELATED_WORK.md) under
"Security, and the one framing that appears to be unclaimed"; every one of them
assumes the injected content arrives through a tool result into an agent that
then acts.

The nearest prior framing is the Cloud Security Alliance AI Safety Initiative
research note of 22 July 2026, *AI Coding Agent Sandbox Escapes: The Trust
Handoff Flaw*, which documents seven escapes across Cursor, Codex CLI, Gemini
CLI and Antigravity. Its contribution is the framing rather than the list: the
sandbox contained the agent's direct actions, but not what unsandboxed
downstream tools later executed from files the agent had written inside it. The
trust handoff is between the sandboxed writer and the unsandboxed reader.

The relationship to this architecture is worth stating precisely rather than
claimed as novelty. CSA describes a handoff across a boundary that exists. What
happens here is that there is no boundary to hand across in the first place:
the untrusted data and the executing program are already colocated, before any
tool is invoked and before anything is written to a file. The CSA hazard is
adjacent and can also occur here, because a run's transcript, its stashed
history buffers and its final answer all leave the sandbox and are read by
things that are not sandboxed. So the honest position is that this threat model
is the CSA hazard plus a prior one that CSA does not cover, and that the prior
one appears not to have been written down anywhere the search could reach.

That is a claim about a literature search, not a claim of priority. It should be
treated as provisional and re-checked.

## What is in scope

The attacker is whoever wrote the context. They control every byte of the text
bound into the REPL, and they may have written it specifically to be analysed by
a system of this shape. They do not control the task, the prompt, the runtime,
the price table or the model weights.

The assets, in the order they are worth defending:

1. Provider credentials on the host.
2. The host filesystem and anything reachable from it.
3. Network egress from the run, both to the provider and to anywhere else.
4. The integrity of the run's own accounting, meaning the cost figures and the
   depth-zero comparison.
5. The confidentiality of the context itself, where it is confidential.

## What the sandbox holds

Read [`src/rlm0/sandbox/`](../src/rlm0/sandbox/) alongside this. Everything
below is a property of the code as it stands, and where a property is weaker
than it sounds that is said in place.

**There is no in-process execution path, and there will not be one.** Two
implementations and deliberately no third. Every surveyed implementation that
offered an `exec`-in-the-orchestrator option made it the default, because it is
the one that always works, and then shipped a filtered `__builtins__` as the
sandbox. That filter is not a boundary; escaping one of them took three lines.
The absent option is the security control, and it is enforced by there being no
code to select.

**The network is off and stays off.** `DockerSandbox` runs with
`--network none`. There is no interface, no resolver and no route inside the
container, and nothing inside it ever learns that a provider exists. Several
surveyed implementations reach the same starting point and then punch an HTTP
hole through to a proxy on the host so that code inside can make sub-calls.
That is not a smaller hole than a network; it is a socket to an endpoint
holding a credential, reachable by exactly the code the isolation was meant to
contain.

**Credentials are never inside.** The container gets three environment
variables: `HOME`, `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`. Nothing else
crosses. `SubprocessSandbox`, which shares the orchestrator's machine, filters
the environment through an allowlist of seven names rather than a denylist of
key-shaped ones, because the whole point is to be right about the variable
nobody thought of.

**Sub-calls are serviced by the host over the pipe.** Code inside calls a shim
that serialises its arguments to JSON, writes one line to the channel, and
blocks. The host runs the real call with the real key and writes one JSON line
back. The shim holds no credential and opens no socket. This is the mechanism
that makes `--network none` sustainable rather than aspirational, and it is why
`Sandbox.register_host_call` takes a name and an arity and nothing else: the
sandbox is told what may be called, never handed the means to make the call.

**Nothing that can execute crosses the boundary in either direction.** The wire
is newline-delimited JSON. The host never deserializes sandbox output with a
format that can construct an object. A surveyed implementation calls
`pickle.loads` on sandbox output in the host process, which hands the host away
for free the moment anything achieves execution inside the boundary. Against
JSON, a hostile sandbox can at worst return a wrong string.

**The channel is moved off file descriptors 0 and 1 before any model-written
code runs.** This is the control that matters most for the specific hazard in
this architecture, and it is in
[`_guest.py`](../src/rlm0/sandbox/_guest.py). `_take_channel` dups the protocol
onto fresh descriptors, then points fd 0 and fd 1 at the null device. If the
channel stayed on fd 1, a single `os.write(1, ...)` from injected code could
forge a `result` frame and tell the host whatever it liked, including a
fabricated answer. If fd 0 stayed live, a single read could steal the host's
reply to a sub-call. After the swap, code inside the boundary cannot address the
channel by a well-known descriptor number. It is still inside the same
interpreter and can in principle walk the process to find the descriptors, so
this raises the cost of forgery rather than making it impossible; it is a
meaningful control against injected code that is not specifically written
against rlm0, and it should not be described as more than that.

**Messages are correlated by id, and a framing error is fatal.** A
desynchronized stream is not parsed hopefully. The failure mode being avoided is
not a crash, it is a stream that continues to produce plausible-looking wrong
answers.

**Deadlines are enforced on both sides.** The guest raises a `BaseException`
subclass on a repeating interval timer, so `except Exception` in model-written
code cannot swallow it and a loop that catches everything becomes a busy wait
rather than an escape. The host kills two seconds later regardless, because a
loop inside a C extension never reaches a bytecode boundary. On Windows there is
no interval timer, so only the host's kill applies and the environment does not
survive it.

**Container containment defaults are set rather than left to the daemon.**
Non-root user (65534), read-only root filesystem, a `noexec,nosuid,nodev` tmpfs
for scratch, memory and memory-swap set to the same value so pressure fails
instead of spilling into swap, CPU and pid ceilings, `--cap-drop ALL`, and
`no-new-privileges`. The whole run line is a free function, `run_argv`, so the
flags can be asserted in a test on a machine with no daemon. That matters
because the containment properties are the part of this file most likely to be
weakened by a well-meaning edit, and a test that only runs where Docker is
installed is a test that mostly does not run.

**Availability of the sandbox is checked at construction, not at first use.** A
sandbox that reports its absence only when the model finally writes code has
already let the caller set up a long run against a promise it cannot keep. The
Docker probe asks the daemon for its version rather than checking PATH, because
a `docker` binary with no daemon behind it is the common case on a laptop that
just rebooted. A surveyed repo shipped a `check_health` method that nothing ever
called, so its documented fail-closed path could never fire.

**Containers are killed, not detached from.** `docker run` is a client, and
killing the client leaves the container running. The terminator runs
`docker rm --force` on the container name. On the subprocess side the whole
process group is killed where the platform has groups. A benchmark sweep that
leaks one container per task exhausts the host.

**Output volume is capped inside the boundary.** The guest caps stdout at 8,000
characters and counts the rest, so a runaway print cannot become a
multi-gigabyte line on the pipe and cannot be used as a memory-exhaustion
vector against the host.

## What the sandbox does not hold

These are the parts that should be read before deciding whether this is safe
enough for a given corpus.

**A kernel bug is a full escape.** `DockerSandbox` is shared-kernel containment.
The 2026 consensus, recorded in [RELATED_WORK.md](RELATED_WORK.md), is that
shared-kernel container isolation is no longer adequate for model-written code.
A container escape via a kernel vulnerability defeats every control listed
above at once, including the network isolation, because the escapee is then on
the host. This is not mitigated here and cannot be mitigated at this layer. A
microVM backend is on the roadmap and is the intended answer.

**The subprocess backend is not a boundary at all.** `SubprocessSandbox` runs as
the same user, on the same filesystem, with the same network access as the
orchestrator. Code running inside it can read your home directory, read your key
files, open sockets and delete your work. Against an attacker-controlled context
it protects nothing. What it does provide is real and worth having on a machine
with no Docker: it isolates the orchestrator from crashes, from a segfaulting
extension, from memory exhaustion, and from a `while True` that would otherwise
wedge the process, because the interpreter can be killed and replaced without
the run dying. It is opt-in and is never selected for you. Use it for
development, for tests, and for contexts you wrote yourself.

**Secret scrubbing is a backstop and not a control.**
`scrub_secrets` in [`protocol.py`](../src/rlm0/sandbox/protocol.py) blanks
secret-shaped substrings on their way back to the model. It is pattern matching
against eight regexes and it will miss things a determined attacker constructs,
and it will occasionally redact something innocent, which is the correct
direction to be wrong in. The control that matters is that no credential is
inside the boundary in the first place. The scrubber exists because a run may
execute on a host with keys in files and the text leaving the boundary lands in
a transcript that gets logged, cached and replayed. Do not treat a passing
scrubber as evidence that a leak did not happen.

**A cooperative shutdown is not guaranteed to be clean.** `close()` is
idempotent and runs from `__del__`, from an atexit hook and from a signal
handler, and those three race on the same object. The signal handlers chain to
whatever was installed before. If the host process is killed with SIGKILL,
nothing runs and containers survive until something sweeps them.

**The model's own output is not trusted, but the model's answer is not
verified either.** The runtime refuses a final answer produced before any code
has run, because that answer came from priors rather than from the context.
That is the only integrity check on the answer path inside the runtime. The
harness adds evidence grading, but a run outside the harness has nothing
equivalent, and an attacker who can steer the model can steer the answer.

**Denial of service is bounded but not prevented.** The budget refuses further
reservations and the sandbox kills long-running blocks. Neither prevents an
attacker-written context from being expensive to analyse, and the cost tail
described in [`budget.py`](../src/rlm0/budget.py) means an adversarial context
is a plausible way to make a run expensive on purpose.

**The CSA trust handoff applies to what leaves.** Stdout, stderr, stashed
history buffers and the final answer all cross out of the sandbox and are read
by the orchestrator, written to `records.jsonl`, and rendered into reports.
None of those readers treat that text as hostile beyond the scrubber. If a
downstream consumer of a run record executes or interprets what it finds there,
the CSA framing applies directly and this project provides no protection.

## Trust boundaries, listed

| Boundary | What crosses | What is trusted |
| --- | --- | --- |
| Host to guest | Guest source on stdin, `set_var` values, `register` names and arities, `host_result` replies | Guest trusts the host completely |
| Guest to host | `ready`, `result`, `value`, `names`, `ack`, `host_call`, `fatal`, all JSON | Host trusts none of it beyond well-formedness |
| Guest to provider | Nothing. There is no path | n/a |
| Host to provider | The prompt, which contains model output and may contain attacker-influenced text | Provider is trusted with the content, not with the accounting |
| Runtime to harness | A `Run`, checked against the reported answer and citations | Harness trusts nothing the solver says about itself |

The third row is the one that carries the design. The absence of a path from the
guest to the provider is what makes everything else affordable.

## What would change the picture

A microVM backend replaces the first item under "does not hold" and is the
single highest-value change available. Parallel sub-calls would introduce
concurrent host-call servicing on a channel currently written for one
outstanding call, and that needs its own review before it lands. Any feature
that lets model-written code name a file path on the host, or that mounts a host
path into the container, reopens the seam the loader-over-stdin design exists to
avoid.

Anything that makes the observation cap configurable upward is a security change
as well as an architectural one, because the cap is what keeps the attacker's
text out of the model's window.
