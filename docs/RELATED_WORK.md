# Related work

This file exists because the first version of this project's README cited one
paper, and the design rests on about forty. It is organised by what each source
changed here, not by topic, because a bibliography that does not say what it
changed is decoration.

Sources are marked **[V]** where a primary source was read, **[S]** where only a
secondary source was available, and **[?]** where a claim could not be
confirmed. Anything marked [S] should be checked before it is quoted anywhere
that matters.

## The paper this implements

**Recursive Language Models.** Alex L. Zhang, Tim Kraska, Omar Khattab. MIT
CSAIL. [arXiv:2512.24601](https://arxiv.org/abs/2512.24601), v1 31 Dec 2025,
v2 28 Jan 2026, v3 11 May 2026. No v4, and no venue acceptance recorded on
arXiv as of 17 Aug 2026. Reference implementation at
[alexzhang13/rlm](https://github.com/alexzhang13/rlm). **[V]**

The prompt is bound to a variable in a persistent REPL and the model writes
code over it, calling itself on slices. What shaped rlm0 is not the headline
but the ablation underneath it: the paper says in its own words that the REPL
is what handles length and that recursion helps on information-dense inputs.
Its own tables show depth 0 as the best configuration in the entire
Qwen3-Coder CodeQA column, and within 3.3 points of depth 1 on BrowseComp+.

## Why depth 0 runs first

**Think, But Don't Overthink: Reproducing Recursive Language Models.** Daren
Wang. [arXiv:2603.02615](https://arxiv.org/abs/2603.02615), 4 Mar 2026. **[V]**

The motivating result for this entire project, and the single most useful table
in the field. On S-NIAH, a lookup task, the base model scores 100 percent in
3.6 seconds, depth 1 drops to 85 percent at 89.3 seconds, and depth 2 to 70
percent at 344.5 seconds. On OOLONG, an aggregation task, the base model scores
0 and depth 1 reaches 42.1. Recursion destroys lookup and creates aggregation
from nothing, in one experiment. A strong base model is actively harmed: Kimi
K2 falls from 86.6 to 60.

The paper's own recommendation is that future work should design better
stopping mechanisms in the REPL. That is what this is. Caveat worth stating:
20 samples per condition, single run, so the effect sizes are large but n is
small.

**Recursive Language Models Meet Uncertainty.** Alizadeh, Shojaee, Cho,
Farajtabar. Apple. [arXiv:2603.15653](https://arxiv.org/abs/2603.15653), Mar
2026. **[V]** States that recursion is not the primary driver of RLM
performance, and that inside the native context window recursion often makes
things worse. Reaches or beats RLM with program search and no recursion at all.

**Prime Intellect, "RLMs: the paradigm of 2026"**, 1 Jan 2026. **[V]** An
industrial, non-author-affiliated evaluation with genuinely mixed results:
gains on Oolong and verbatim copy, and a clear regression on math-python where
RLM performed significantly worse than a plain model. They also listed depth 0
as unbuilt roadmap, which is part of why this project exists.

**The Illusion of Multi-Agent Advantage.** Jwalapuram, Lin et al. NTU, Meta AI,
Oxford, Tokyo Tech. [arXiv:2606.13003](https://arxiv.org/abs/2606.13003), Jun
2026. **[S]** Automatically designed multi-agent systems rarely beat a strong
single-agent baseline once cost is accounted for, and chain-of-thought with
self-consistency beat them at under 10 percent of the cost.

This changed the evaluation plan rather than the runtime. The control here
cannot be depth 0 alone. It has to include CoT self-consistency at matched
cost, or the comparison is against a strawman and proves nothing.

**How Much Coordination Gain Is Real?** Kaliyev, Maryanskyy.
[arXiv:2606.20695](https://arxiv.org/abs/2606.20695), Jun 2026. **[V]** Many
reported coordination gains fall inside run-to-run noise. Any depth-1 over
depth-0 delta reported here has to be measured against a paired noise floor.

**METR, guidelines for capability elicitation.** **[S]** The counterweight: a
weak baseline understates the model, which is a methodological error in the
opposite direction. The depth-0 control has to be genuinely well elicited.

## The theory, including the part that cuts against this design

**Recursive Models for Long-Horizon Reasoning.** Chenxiao Yang, Nathan Srebro,
Zhiyuan Li. TTIC. [arXiv:2603.02112](https://arxiv.org/abs/2603.02112), Mar
2026, reported as ICML 2026 **[S on the venue]**. Proves any computable problem
admits a recursive decomposition where each subtask needs exponentially smaller
active context, and that this is optimally powerful within agentic systems.

This is the strongest theoretical argument that recursion buys something depth
0 cannot, which is precisely why it belongs here. The position taken by rlm0 is
not that recursion is useless. It is that the regime where it pays is narrower
than the framing suggests, and that a runtime should establish per task which
regime it is in.

**State Representation and Termination for Recursive Reasoning Systems.** Guha,
Mukherjee, Kukreja, Kumar.
[arXiv:2605.06690](https://arxiv.org/abs/2605.06690), May 2026. **[S]** An
order-gap stopping criterion replacing fixed compute budgets with an
evidence-driven signal. Theory with no runtime and no benchmarks, which is the
gap this fills rather than a competitor.

**Context Compaction Theory.** Tirmazi, Markelon, Bishop, Mitzenmacher.
[arXiv:2608.01326](https://arxiv.org/abs/2608.01326), 2 Aug 2026. **[S]** Proves
the minimum compaction budget for answering a query set within a target error
equals the one-way communication complexity of the induced problem. The
framework to reach for when asking whether a cost bound here is too tight to be
sound.

## What absorbed this category

**Code as Agent Harness.**
[arXiv:2605.18747](https://arxiv.org/abs/2605.18747), May 2026. **[S]** Names
the frame that now contains RLM: code makes reasoning executable, action
programmable, and environment state inspectable. Binding a long context to a
REPL variable is one instance of the third clause.

**Recursive Agent Harnesses.** Lumer, Sen, Paul, Subbiah.
[arXiv:2606.13643](https://arxiv.org/abs/2606.13643), 11 Jun 2026. **[V]**
Generalises RLM in its own first sentence, recursing over full agent harnesses
with filesystem and planning rather than bare model calls. Oolong-Synthetic at
4M tokens: 81.36 on GPT-5 against a 71.75 baseline, 89.77 on Claude Sonnet 4.5.
Its configurable default depth is three. The paper does not provide the
cost-matched ablation needed to establish the cost of that recursion.

**Anthropic Dynamic Workflows**, around late May 2026. **[S]** The deployed
version: the model writes an orchestration script, up to 1,000 subagents with
16 concurrent, intermediate results held in script variables so the main
context sees only the answer. That last clause is the RLM thesis shipped as a
product.

**CaveAgent.** Ran, Wan, Lin, Zhang et al.
[arXiv:2601.01569](https://arxiv.org/abs/2601.01569), Jan 2026. **[V]** A
persistent REPL with variable bindings surviving across turns and large data
bound to variables rather than re-transmitted. Prior art on the core mechanism,
framed as statefulness rather than recursion.

The consequence for positioning: "an RLM runtime" is not a distinctive claim in
August 2026, and this project should not make it.

## Mechanisms that solve the same problem without recursion

**Addressable Recall Compaction.** Dang, Ichikawa, Fatima, Shirahata.
[arXiv:2607.25066](https://arxiv.org/abs/2607.25066), 27 Jul 2026. **[V]** An
append-only store keyed by stable hashes, a transcript of citation stubs, and a
`_recall` action to pull exact observations back. Fixed active prompt budget
with exact recoverability. NIAH 99.40 percent against 88.12 for the best
baseline on Qwen3-8B, lowest serving latency of everything tested.

The sharpest competitor here. It is cheaper and lower latency, and it beats
four baseline families. The claim worth testing rather than asserting is that
`_recall` returns items one at a time and never computes over a whole corpus,
so it should lose on aggregation and win on lookup, which is the same boundary
this project is built around.

**VISTA.** Xu, Li, Zhang.
[arXiv:2606.30005](https://arxiv.org/abs/2606.30005), 29 Jun 2026. **[V]**
Gives the model a dashboard of its own context state and lossless
archive-and-recover, with no training. LOCA-Bench 50.7 against 22.7 for ReAct.
A prompt-level interface change beating a runtime by 28 points is a humbling
and necessary comparison.

**Aggregation Queries over Unstructured Text.** Zhu, Xu, Li, Liu, Qiu, Chen,
Jin. [arXiv:2602.01355](https://arxiv.org/abs/2602.01355), Feb 2026. **[V]**
Formalises entity-level aggregation with a strict completeness requirement and
states that Text-to-SQL and RAG fail to achieve it. Introduces AGGBench. This
is where the aggregation claim has to be demonstrated, and DFA is the baseline
to beat.

**When to Retrieve During Reasoning.** Guo, Wu, Yiu. SIGIR 2026.
[arXiv:2604.26649](https://arxiv.org/abs/2604.26649). **[V]** Step-level
uncertainty gating retrieval. MuSiQue 71.2 F1 at 1.8 retrieval calls against
IRCoT's 65.4 at 3.4. The same shape of idea one layer down.

**λ-RLM: The Y-Combinator for LLMs.** Roy, Tutunov, Ji, Zimmer, Bou-Ammar.
[arXiv:2603.20105](https://arxiv.org/abs/2603.20105), 20 Mar 2026. **[V]**
Replaces free-form recursive code with typed combinators over a λ-calculus
runtime, claiming termination, closed-form cost bounds and an optimal partition
rule. Beats standard RLM in 29 of 36 model-task comparisons, up to 21.9 points
and 4.1x lower latency.

The nearest competitor on cost bounds, and it gets there by proof rather than
accounting. Its diagnosis is that an open-ended REPL is hard to verify or
predict, which is an argument against the mechanism this project keeps. The
counter is generality: a combinator library has a coverage gap and a REPL does
not, and rlm0 bounds cost at runtime instead of restricting what can be
written. Whether that trade is worth it is an open question and should be
stated as one.

## Implementations that set the engineering baseline

**zigrlm.** The local `zigrlm-main` implementation uses parallel child work,
deterministic ordering, network-closed container execution, and secret
redaction. It is direct prior art for host callbacks over a local transport and
for deterministic trace ordering. rlm0 should not claim either as novel. Its
different question is whether a run should always retain a no-recursion control
and one budget record.

**RLM_agent.** The local `RLM_agent-main` implementation provides durable agent
state, context handles, repeated-call detection, parent-budget accounting, and
batched work. It is prior art for much of rlm0's state and accounting surface.
Its approach is a stateful local-agent system, not a paired evaluation runtime.

**TimeRLM.** *Recursive Language Models Enable Precise Anomaly Localization in
Long-Context Time-Series.* **[V]** TimeRLM keeps numerical signals in a
persistent environment and grades structured evidence, not only a task answer.
Its AnomalyXL benchmark requires exact anomaly location, type, magnitude, and
lead-lag evidence across long signals. Its published configuration implemented
sub-model support but left it disabled, so the result supports bounded
environmental reasoning and evidence-aware evaluation more directly than a
general recursion claim. The permissive final-answer recovery in its harness is
also a reason for rlm0 to move toward a strict, versioned completion protocol.

**Chained Recursive Language Models.** See the extension section below. It is
the relevant prior art for using fresh roots, a compact blackboard, artifacts,
and handoffs across stages. This is not part of rlm0's default policy.

**RLMOpt.** See the extension section below. It is useful prior art for prompt
optimization with deterministic guards, Pareto selection, and regression
checks. It is not a substitute for a runtime-level evaluation protocol.

## Where this project's contributions were already taken

**Budget reservation.** *Token Budgets: An Empirical Catalog of 63 LLM-Agent
Budget-Overrun Incidents.* [arXiv:2606.04056](https://arxiv.org/abs/2606.04056),
Jun 2026, artifact at
[sajjadanwar0/token-budgets](https://github.com/sajjadanwar0/token-budgets).
**[V]** Publishes the reserve, reconcile and refund lifecycle with an
eight-cluster taxonomy, using Rust affine types so double-spending a budget is
a compile error. One cluster is named delegation-fanout, which is exactly the
RLM failure mode.

*Agent Contracts.* Ye, Tan.
[arXiv:2601.08815](https://arxiv.org/abs/2601.08815), Jan 2026. **[V]**
Establishes budget conservation under delegation: child agents cannot exceed
parent allocations. That is the invariant `RunBudget` enforces, stated formally
a year earlier. No implementation released.

*LiteLLM* ships gateway-scope budget enforcement today. **[S]**

So the mechanism in `budget.py` is not novel. What remains open, and what this
project should claim instead: a tighter fan-out estimator, since the Rust work
concedes 4 to 6x static over-reservation; tree-scoped rather than key-scoped
accounting; and graceful degradation under a binding budget, for which a
targeted search found nothing published at all.

**Sandboxing.** `@anthropic-ai/sandbox-runtime` **[S]** does OS-level
filesystem and egress restriction with Bubblewrap and Seatbelt.
`microsandbox` **[S]** is Apache-2.0 libkrun microVMs booting under 200ms and
self-hostable. `llm-sandbox` **[S]** is a maintained Docker wrapper with 1.1k
stars. The 2026 consensus is that shared-kernel container isolation is no
longer adequate for model-written code, so the Docker sandbox here is the
specific thing the field moved past, and a microVM backend belongs on the
roadmap.

**Evaluation harness.** *Holistic Agent Leaderboard.* Kapoor et al., 30
co-authors. [arXiv:2510.11977](https://arxiv.org/abs/2510.11977), ICLR 2026,
[hal.cs.princeton.edu](https://hal.cs.princeton.edu/). **[V]** Cost-controlled
evaluation by default with accuracy-cost Pareto frontiers, analysed across
models by scaffolds by benchmarks, with 21,730 rollouts and 2.5 billion tokens
of logs released. Its scaffold axis is this project's ablation axis.

The claim that nobody can compare these systems honestly was true of the twenty
RLM repositories surveyed and false of the wider field. The right move is to
run inside HAL and reframe the contribution as an honest ablation of an RLM
scaffold, not as a harness.

Its ancestor, *AI Agents That Matter*
([arXiv:2407.01502](https://arxiv.org/abs/2407.01502)), is the canonical
citation for benchmarks ignoring cost.

**Adaptive depth as a phrase.** *RVLM*
([arXiv:2603.24224](https://arxiv.org/abs/2603.24224)) has an adaptive-depth
router with budget prediction and stall detection, in vision-language. *When to
Think Deeply* ([arXiv:2606.06745](https://arxiv.org/abs/2606.06745)) is this
control loop at the token-budget level: fast attempt, conflict monitoring,
selective escalation. *Dynamic Model Routing and Cascading: A Survey*
([arXiv:2603.04445](https://arxiv.org/abs/2603.04445)) places depth-0-first
precisely as post-generation multi-stage cascading with a response-signal
deferral rule, and notes it surveys routing between models rather than between
strategies.

So the honest claim is cascading applied to a new axis, and the phrase to avoid
is "adaptive depth".

## Prefix caching, and a constraint that changes the design

**Anthropic prompt caching documentation.** **[V]** A cache entry becomes
available only after the first response begins, and **parallel requests sharing
a prefix do not hit each other's cache**. The documented pattern is to issue
one request, wait for the first token, then fire the rest.

This is a hard ceiling on fan-out caching and it inverts the naive design. A
cold fan-out of N children pays N prefix writes at 1.25x base input, which is
worse than not caching at all. A barrier before dispatch is therefore a
correctness fix for the cost model, not an optimisation. `rlm_batch` implements
that barrier now; live-provider measurements remain necessary before making a
cache-savings claim.

Also from the same source: minimum cacheable length is model-dependent rather
than a flat 1024, and below it no cache is created and no error is returned, so
the only way to know is to read the usage fields.

**KVFlow** ([OpenReview 5Iw1nDtYmT](https://openreview.net/pdf?id=5Iw1nDtYmT))
**[S]** observes that LRU eviction is mismatched to agentic workflows because
it ignores known future execution order. **Autellix**
([arXiv:2502.13965](https://arxiv.org/abs/2502.13965)) **[S]** schedules at the
program level. Both are serving-layer work, and both are the intellectual
justification for a runtime knowing its own call tree in advance.

**When KV Cache Reuse Fails in Multi-Agent Systems**
([arXiv:2601.08343](https://arxiv.org/abs/2601.08343)) **[S]** is a negative
result about this exact mechanism and is worth reading before claiming savings.

Reported savings across 500-plus real agent sessions run 41 to 80 percent
rather than the vendor 90 **[S, secondhand and unverified]**, which is why this
project reports its own measured hit rate from provider usage fields instead of
quoting a number.

## Security, and the one framing that appears to be unclaimed

**CSA Research Note, AI Coding Agent Sandbox Escapes: The Trust Handoff Flaw.**
Cloud Security Alliance AI Safety Initiative, 22 Jul 2026. **[V]** Seven
documented escapes across Cursor, Codex CLI, Gemini CLI and Antigravity. The
framing matters more than the list: the sandbox contained the agent's direct
actions but not what unsandboxed downstream tools later executed from files the
agent wrote inside it.

**Microsoft Security, "When prompts become shells"**, 7 May 2026. **[S]**
Includes an agent finding a path around a denylist and then disabling its own
Bubblewrap sandbox.

Also: Cursor CVE-2026-50548 and CVE-2026-50549, both CVSS 9.8; vm2 escapes in
May 2026; Wasmtime advisories in April 2026; and a Grist advisory stating
plainly that Pyodide on Node has no useful sandbox barrier. **[all S]** Taken
together, WebAssembly is not a shortcut here.

**The gap.** A targeted search found nothing addressing the structural hazard
specific to this architecture: the untrusted context and the code-writing model
occupy one interpreter by construction, so text under analysis can become the
program. Every source found assumes injected content arrives through a tool
result into an agent that then acts. The CSA trust-handoff framing is the
nearest neighbour. This is where the threat model here is built, and as far as
three sweeps can tell it is unclaimed.

## Benchmarks

OOLONG ([arXiv:2511.02817](https://arxiv.org/abs/2511.02817)) remains the load
bearing one: aggregation rather than retrieval, and no model above 50 percent
at 128K. ATLAS ([arXiv:2605.28079](https://arxiv.org/abs/2605.28079)) for
length-aware scoring, HELMET
([arXiv:2410.02694](https://arxiv.org/abs/2410.02694)) for downstream tasks,
MRCR v2, AA-LCR, BrowseComp-Plus. A targeted search found nothing superseding
these as of August 2026. **[S]**

**Haystack Engineering**
([arXiv:2510.07414](https://arxiv.org/abs/2510.07414)) **[S]** is the sharpest
critique of HELMET's distractor construction and shaped the corpus generator
here.

**AMA-Bench** ([arXiv:2602.22769](https://arxiv.org/abs/2602.22769)) and
**ContextBench** ([arXiv:2602.05892](https://arxiv.org/abs/2602.05892)) **[S]**
both report scaffolds underperforming a plain long-context baseline. They are
cited here because they make an insistence on running the control look like
judgement rather than modesty.

**LLM Benchmark Datasets Should Be Contamination-Resistant**
([arXiv:2605.19999](https://arxiv.org/abs/2605.19999)) **[S]** argues a
contamination-resistant benchmark applies structure to perturbed inputs rather
than testing recall of a memorised instance. The generator here controls its
corpus, so entity and value perturbation with structure held fixed is cheap and
is on the roadmap.

**Layer-Isolated Evaluation**
([arXiv:2606.11686](https://arxiv.org/abs/2606.11686)) **[S]** argues for
testing deterministic layers behind their own assertions rather than end to end
through a stochastic stack. The sandbox, budget and parsing layers here are
tested that way, with no model in the loop.

## Extensions and task-specific RLM work

**Chained Recursive Language Models.** Mitra, Ulukus. Maryland.
[arXiv:2608.05124](https://arxiv.org/abs/2608.05124), 5 Aug 2026. **[V]**
Repeated fresh reasoning roots carrying a compact summary, blackboard, and
artifacts instead of full history. This is prior art for fresh-root handoffs,
not for rlm0's depth-zero paired control. The paper does not provide enough
cost-matched detail to support a general serving claim.

**RLMOpt.** Satheesha, Pande, Duddempudi, Dandala.
[arXiv:2608.10471](https://arxiv.org/abs/2608.10471), 11 Aug 2026. **[V]**
RLM as the controller for prompt optimisation, with deterministic execution,
Pareto selection, and regression/significance guards. It is adjacent evaluation
and development work, not an inference-runtime replacement.

**PEEK.** Gu, Zhang, **Khattab**, Madden.
[arXiv:2605.19932](https://arxiv.org/abs/2605.19932), May 2026. **[V]** A
constant-sized context map cached across invocations. Khattab's own follow-up
direction, and amortisation across runs rather than depth control within one,
so it is complementary.

**Coding Agents are Effective Long-Context Processors.** Cao, Yin, Dhingra,
Zhou. [arXiv:2603.20432](https://arxiv.org/abs/2603.20432), Mar 2026. **[V on
existence, ? on the head-to-head]** Claims off-the-shelf coding agents beat
published state of the art by 17.3 percent on average. The "just point a coding
agent at a directory" argument, and the competitive baseline this has to beat.
The honest answer is that a coding agent has no cost bound and no depth control.

## Open threads

A search snippet described an implementation with `max_total_cost_usd` and
calls reserved atomically before provider requests, and it could not be
attributed to a project. If that is a community repository, someone has already
built the budget layer here. **[?]**

Whether the reference implementation supports a depth-0 mode or any budget
feature could not be confirmed from its README. **[?]**

Whether Yang, Srebro and Li was accepted at ICML 2026 rests on a secondary
source. **[?]**
