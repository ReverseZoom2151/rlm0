"""The command line, which exists so the cost and the verdict are unavoidable.

Every command that spends money ends by printing `Run.describe()`, because the
argument this project makes is that a recursive run should report what it cost
and whether the recursion earned that cost, from the run itself rather than
from an invoice a month later. A CLI that printed only the answer would undo
that in one line of formatting.

Three rules shape the error paths here, and all three are about what must not
reach a terminal or a log:

* No API key is ever read for display, echoed back, or included in a message.
  The runtime reads its own credentials; this module never looks at them.
* No error path prints the context. The context is the attacker controlled,
  possibly enormous, possibly confidential input, and the natural instinct of
  an exception handler is to include the value it choked on. Failures here
  report paths and sizes.
* Anything that came back from a model or a sandbox is truncated and passed
  through the secret scrubber before it is printed, because a transcript that
  gets logged is a transcript that gets kept.

The runtime itself is built by `rlm0.assembly.build_rlm`, imported lazily so
that `--help`, `cost` and `sandbox` cost nothing and keep working even when a
provider SDK is absent. Two of that function's defaults are deliberately not
softened here. The sandbox choice has no `auto` value, because an `auto` that
falls back to the subprocess backend when Docker is missing would silently move
a run from a boundary to no boundary. And a provider is never guessed: one of
`--provider anthropic`, `openai` or `fake` is always named, so a run against
the wrong account is not one forgotten flag away.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from rlm0 import __version__

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    from rlm0.harness import SolverResult, SolverTask
    from rlm0.run import Run

__all__ = ["CliError", "main"]

EXIT_OK = 0
"""The command did what it was asked to do."""

EXIT_FAILED = 1
"""The work ran and did not succeed: no answer, or a report that refused."""

EXIT_USAGE = 2
"""Argparse's own code for a malformed command line."""

EXIT_CONFIG = 3
"""The configuration cannot be honoured, so nothing was attempted."""

EXIT_UNAVAILABLE = 4
"""The environment is missing something the run needs, such as a sandbox."""

_BUILDER_NAMES = ("build_rlm", "build_runtime", "build", "assemble")
"""Names `rlm0.assembly` might give its constructor, in order of preference."""

_DETAIL_LIMIT = 400
"""Characters of any model or sandbox originated text that reach the terminal."""

_DEFAULT_CONTEXT_LIMIT_MB = 64.0

_FAKE_CODE_TURN = "```repl\nprint(len(context))\n```"
"""The first thing `--provider fake` says: a block, in the shape a model uses.

It runs code before answering because the parser refuses a final answer from a
turn where nothing has run yet, and refuses a turn that both runs code and
answers. Those two rules are what stop a model from answering a question about
a context it never opened, and a fake that sidestepped them would exercise a
path the real system does not have.
"""

_FAKE_FINAL_TURN = (
    "FINAL(the fake provider does not answer questions. It exists so that the "
    "sandbox, the budget and the run record can be exercised end to end "
    "without calling a model.)"
)
"""What it says on every turn after the first."""


class CliError(Exception):
    """A failure with an exit code and a sentence a human can act on."""

    def __init__(self, code: int, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint


class _Runtime(Protocol):
    """What the CLI needs from whatever `rlm0.assembly` hands back."""

    def complete(self, task: str, context: str = "") -> Run: ...


# -- safety helpers -----------------------------------------------------


def _scrub(text: str) -> str:
    """Redact secret shaped substrings, falling back to passing text through.

    The scrubber lives in the sandbox package because that is where untrusted
    output crosses a boundary. If that package cannot be imported the CLI still
    has to be able to report the failure, so the fallback is the raw text with
    the same truncation applied by the caller.
    """
    try:
        from rlm0.sandbox.protocol import scrub_secrets
    except Exception:  # pragma: no cover - only on a broken install
        return text
    return scrub_secrets(text)


def _detail(exc: BaseException, *, limit: int = _DETAIL_LIMIT) -> str:
    """One short, scrubbed line describing a failure.

    Exception messages in this system can carry model output, sandbox stdout,
    or a slice of the context, so the message is collapsed to one line, cut to
    a fixed length and scrubbed before anybody sees it.
    """
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    text = _scrub(text)
    if len(text) > limit:
        text = f"{text[:limit]} [truncated]"
    return text


# -- context loading ----------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise CliError(
            EXIT_CONFIG,
            f"{path} is not UTF-8 text, so it cannot be used as context",
            hint="point --context at text files, or convert the file first",
        ) from exc
    except OSError as exc:
        raise CliError(EXIT_CONFIG, f"could not read {path}: {exc.strerror}") from exc


def _files_under(path: Path, pattern: str) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise CliError(EXIT_CONFIG, f"no such file or directory: {path}")
    return sorted(p for p in path.glob(pattern) if p.is_file())


def load_context(
    paths: Sequence[Path], *, pattern: str, limit_mb: float
) -> tuple[str, int]:
    """Concatenate files into one context string, with each document named.

    Returns the text and the number of files. Nothing about the contents is
    logged or echoed anywhere: a caller that wants to know what went in gets
    the file count and the character count, which is all this function will
    ever say about it.
    """
    documents: list[str] = []
    n_files = 0
    total = 0
    limit = int(limit_mb * 1024 * 1024)
    for root in paths:
        for path in _files_under(root, pattern):
            text = _read_text(path)
            total += len(text)
            if total > limit:
                raise CliError(
                    EXIT_CONFIG,
                    f"context exceeds the {limit_mb:g} MB limit at {path}",
                    hint="raise --context-limit-mb if that is genuinely intended",
                )
            label = path.as_posix()
            documents.append(f"DOCUMENT {label}\n{text}")
            n_files += 1
    return "\n\n".join(documents), n_files


# -- the assembly seam --------------------------------------------------


def _assembly() -> Any:
    try:
        return importlib.import_module("rlm0.assembly")
    except ImportError as exc:
        raise CliError(
            EXIT_CONFIG,
            "rlm0.assembly could not be imported, so no runtime can be built "
            f"({_detail(exc)})",
        ) from exc


def _builder() -> Callable[..., Any]:
    module = _assembly()
    for name in _BUILDER_NAMES:
        candidate = getattr(module, name, None)
        if callable(candidate):
            found: Callable[..., Any] = candidate
            return found
    raise CliError(
        EXIT_CONFIG,
        "rlm0.assembly exposes no runtime builder; the CLI looks for "
        f"{', '.join(_BUILDER_NAMES)}",
    )


def _client(provider: str) -> Any:
    """One provider client, or an error naming what to install.

    No key is read here and none is ever printed. The clients read their own
    credentials from the environment, which is the only place this CLI wants
    them to exist.
    """
    from rlm0.providers.errors import ProviderDependencyError

    try:
        if provider == "anthropic":
            from rlm0.providers import AnthropicClient

            return AnthropicClient()
        if provider == "openai":
            from rlm0.providers import OpenAIClient

            return OpenAIClient()
        if provider == "gemini":
            from rlm0.providers import GeminiClient

            return GeminiClient()
        from rlm0.providers import FakeClient, FakeReply

        return FakeClient(
            replies=(FakeReply(text=_FAKE_CODE_TURN),),
            default_reply=FakeReply(text=_FAKE_FINAL_TURN),
        )
    except ProviderDependencyError as exc:
        raise CliError(EXIT_UNAVAILABLE, _detail(exc)) from exc


def _budget(args: argparse.Namespace) -> Any:
    """The ceiling this run executes under, and never no ceiling by accident.

    With no ceiling named the assembly default is used, which bounds cost,
    calls and wall clock. `--unbounded` is the only way to run without one and
    it conflicts with every ceiling flag, because a budget that is both
    unbounded and bounded is a run record nobody can read.
    """
    from rlm0.budget import RunBudget, Unbounded

    ceilings = {
        "max_usd": args.max_usd,
        "max_seconds": args.max_seconds,
        "max_calls": args.max_calls,
        "max_tokens": args.max_tokens,
    }
    named = {name: value for name, value in ceilings.items() if value is not None}
    if args.unbounded:
        if named:
            raise CliError(
                EXIT_CONFIG,
                "--unbounded was given alongside "
                f"{', '.join('--' + n.replace('_', '-') for n in sorted(named))}",
                hint="a run is either bounded or it is not; pick one",
            )
        return Unbounded()
    if not named:
        return _assembly().default_budget()
    try:
        return RunBudget(**named, max_unpriced_calls=args.max_unpriced_calls)
    except ValueError as exc:
        raise CliError(EXIT_CONFIG, f"the budget is not usable: {exc}") from exc


def _policy(args: argparse.Namespace) -> Any:
    from rlm0.policy import Escalating, Fixed, Never

    try:
        if args.policy == "never":
            return Never()
        if args.policy == "fixed":
            return Fixed(depth=args.max_depth)
        return Escalating(max_depth=args.max_depth)
    except ValueError as exc:
        raise CliError(EXIT_CONFIG, f"--max-depth {args.max_depth}: {exc}") from exc


def _runtime_factory(args: argparse.Namespace) -> Callable[[], _Runtime]:
    """A callable that builds one fresh runtime, budget and all.

    Fresh per call rather than shared, because the budget is per run by design.
    A suite that shared one would spend the whole ceiling on its first few
    samples and report every sample after that as budget exhausted, which reads
    as a finding about the task set and is a finding about the harness.
    """
    builder = _builder()
    _client(args.provider)  # fail now if the SDK is missing, not mid-suite

    def make() -> _Runtime:
        built: _Runtime = builder(
            model=args.model,
            lm=_client(args.provider),
            budget=_budget(args),
            sandbox=args.sandbox,
            policy=_policy(args),
            sub_model=args.sub_model,
            max_iterations=args.max_iterations,
            max_tokens=args.max_output_tokens,
            max_attempts=args.max_attempts,
            exec_timeout_s=args.exec_timeout_s,
            attempt_timeout_s=args.attempt_timeout_s,
        )
        return built

    return make


def _describe_config(args: argparse.Namespace) -> str:
    """The configuration line recorded in a harness manifest."""
    parts = [f"rlm0 {__version__}", f"provider={args.provider}", f"model={args.model}"]
    if args.sub_model:
        parts.append(f"sub_model={args.sub_model}")
    parts.append(f"sandbox={args.sandbox}")
    parts.append(f"policy={args.policy}")
    parts.append(f"max_depth={args.max_depth}")
    for name in ("max_usd", "max_seconds", "max_calls", "max_tokens"):
        value = getattr(args, name)
        parts.append(f"{name}={'unset' if value is None else value}")
    return " ".join(parts)


def _run_guarded(work: Callable[[], Run]) -> Run:
    """Turn the failures a runtime can raise into exit codes and one line each."""
    try:
        return work()
    except CliError:
        raise
    except Exception as exc:
        name = type(exc).__name__
        if name in {"SandboxUnavailableError", "RecursionUnavailableError"}:
            raise CliError(EXIT_UNAVAILABLE, _detail(exc)) from exc
        raise CliError(EXIT_FAILED, f"the run failed: {_detail(exc)}") from exc


# -- run ----------------------------------------------------------------


def cmd_run(args: argparse.Namespace, argv: Sequence[str]) -> int:
    del argv
    context, n_files = load_context(
        args.context, pattern=args.glob, limit_mb=args.context_limit_mb
    )
    factory = _runtime_factory(args)
    print(
        f"task: {args.task}\ncontext: {n_files} file(s), {len(context)} chars",
        file=sys.stderr,
    )
    # Building the runtime is inside the guard because that is where a missing
    # sandbox announces itself, and a stack trace is not a reason a human can
    # act on.
    run = _run_guarded(lambda: factory().complete(args.task, context))

    if args.record is not None:
        _write_record(args.record, run)

    answer = run.answer
    if answer is None:
        print("no answer: every attempt stopped before producing one")
    else:
        print(answer)
    # Last, always. The whole point of the project is that this block is not
    # optional and is not somewhere else.
    print()
    print(run.describe())
    return EXIT_OK if answer is not None else EXIT_FAILED


def _write_record(path: Path, run: Run) -> None:
    from rlm0.harness.runner import run_to_dict

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(run_to_dict(run), indent=2), encoding="utf-8")
    except OSError as exc:
        raise CliError(
            EXIT_FAILED, f"could not write the run record to {path}: {exc.strerror}"
        ) from exc


# -- eval ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _RuntimeSolver:
    """Adapts a runtime to the harness solver interface.

    Every sample gets its own runtime, so every sample gets its own budget and
    its own depth-zero control. The answer reported to the harness is taken
    from the `Run` rather than from anywhere else, which is what lets the
    harness check that the answer being scored is the answer that was paid for.
    """

    factory: Callable[[], _Runtime]
    label: str

    def solve(self, task: SolverTask) -> SolverResult:
        from rlm0.harness import Attempted, SolverResult

        run = self.factory().complete(task.question, task.context())
        baseline = run.baseline
        return SolverResult(
            run=run,
            final=Attempted(answer=run.answer),
            baseline=None if baseline is None else Attempted(answer=baseline.answer),
        )

    def describe(self) -> str:
        return self.label


def cmd_eval(args: argparse.Namespace, argv: Sequence[str]) -> int:
    from rlm0.harness import CorpusSpec, GradingPolicy, generate_corpus
    from rlm0.harness.report import ReportRefusalError
    from rlm0.harness.runner import run_suite

    try:
        spec = CorpusSpec(seed=args.seed, samples_per_family=args.samples_per_family)
    except ValueError as exc:
        raise CliError(EXIT_CONFIG, f"the corpus spec is not usable: {exc}") from exc
    corpus = generate_corpus(spec)
    solver = _RuntimeSolver(
        factory=_runtime_factory(args), label=_describe_config(args)
    )

    print(
        f"corpus {corpus.content_hash[:16]} with {len(corpus.samples)} samples "
        f"from seed {args.seed}",
        file=sys.stderr,
    )
    try:
        result = run_suite(
            corpus,
            solver,
            args.out,
            policy=GradingPolicy(),
            invocation=list(argv),
            resume=not args.no_resume,
        )
    except CliError:
        raise
    except Exception as exc:
        raise CliError(EXIT_FAILED, f"the suite failed: {_detail(exc)}") from exc

    try:
        print(result.report().render())
    except ReportRefusalError as exc:
        raise CliError(
            EXIT_FAILED,
            f"the result table refused to render: {_detail(exc)}",
            hint=f"the per sample records are still in {args.out}",
        ) from exc
    print(f"\nrecords: {args.out}")
    return EXIT_OK


# -- cost ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CostEstimate:
    """The worst case a configuration allows, stated with its assumptions."""

    root_calls: float
    sub_calls: float
    root_usd: float | None
    sub_usd: float | None

    @property
    def total_usd(self) -> float | None:
        """None when either half was unpriceable. Never a partial sum."""
        if self.root_usd is None or self.sub_usd is None:
            return None
        return self.root_usd + self.sub_usd


def worst_case_calls(
    *, max_iterations: int, max_depth: int, fanout: int
) -> tuple[int, int]:
    """How many calls the configuration permits, root and sub, at the ceiling.

    Every attempt from the depth-zero control up to the deepest one is counted,
    because the runtime really does run them in order, and a per level fan-out
    of `fanout` is assumed for every iteration. This is an upper bound and is
    labelled as one: the usual run stops at depth zero and costs a small
    fraction of it.
    """
    root = 0
    sub = 0
    for bound in range(max_depth + 1):
        root += max_iterations
        for depth in range(1, bound + 1):
            sub += max_iterations * fanout**depth
    return root, sub


def estimate_cost(
    *,
    model: str,
    sub_model: str,
    root_calls: float,
    sub_calls: float,
    input_tokens: int,
    output_tokens: int,
) -> CostEstimate:
    from rlm0.providers.pricing import PriceTable
    from rlm0.run import TokenUsage

    table = PriceTable()
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    root_price = table.get(model)
    sub_price = table.get(sub_model)
    return CostEstimate(
        root_calls=root_calls,
        sub_calls=sub_calls,
        root_usd=None if root_price is None else root_price.cost(usage) * root_calls,
        sub_usd=None if sub_price is None else sub_price.cost(usage) * sub_calls,
    )


def cmd_cost(args: argparse.Namespace, argv: Sequence[str]) -> int:
    del argv
    sub_model = args.sub_model or args.model
    root, sub = worst_case_calls(
        max_iterations=args.max_iterations,
        max_depth=args.max_depth,
        fanout=args.fanout,
    )
    total_calls = root + sub
    scale = 1.0
    if args.max_calls is not None and total_calls > args.max_calls:
        scale = args.max_calls / total_calls

    estimate = estimate_cost(
        model=args.model,
        sub_model=sub_model,
        root_calls=root * scale,
        sub_calls=sub * scale,
        input_tokens=args.input_tokens,
        output_tokens=args.output_tokens,
    )

    print("worst case for this configuration, before anything is spent")
    print(f"  models: root {args.model}, sub {sub_model}")
    print(
        f"  assumes {args.input_tokens} input and {args.output_tokens} output "
        f"tokens per call, fan-out {args.fanout}"
    )
    print(
        f"  attempts: depth 0 through {args.max_depth}, "
        f"{args.max_iterations} iterations each"
    )
    if scale < 1.0:
        print(f"  the --max-calls ceiling of {args.max_calls} binds first")
    print(
        f"  calls: {estimate.root_calls:.0f} root, {estimate.sub_calls:.0f} sub, "
        f"{estimate.root_calls + estimate.sub_calls:.0f} total"
    )
    print(f"  root:  {_usd(estimate.root_usd)}")
    print(f"  sub:   {_usd(estimate.sub_usd)}")
    print(f"  total: {_usd(estimate.total_usd)}")

    total = estimate.total_usd
    if total is None:
        raise CliError(
            EXIT_CONFIG,
            "at least one model has no entry in the price table, so this "
            "configuration cannot be costed and is reported as unpriced rather "
            "than as zero",
            hint=(
                "supply the current rate through PriceTable.with_overrides, or "
                "run without a USD ceiling and accept unpriced spend"
            ),
        )
    if args.max_usd is not None and total > args.max_usd:
        print(
            f"\nthe worst case exceeds --max-usd ${args.max_usd:.4f}, so the "
            "budget would bind and the run would wind down rather than finish"
        )
    return EXIT_OK


def _usd(value: float | None) -> str:
    """Unpriced is a word here, exactly as it is everywhere else in rlm0."""
    return "unpriced" if value is None else f"${value:.4f}"


# -- sandbox ------------------------------------------------------------


def cmd_sandbox(args: argparse.Namespace, argv: Sequence[str]) -> int:
    del argv
    try:
        from rlm0.sandbox import SandboxUnavailableError, docker_available
    except Exception as exc:  # pragma: no cover - only on a broken install
        raise CliError(
            EXIT_UNAVAILABLE, f"the sandbox package will not import: {_detail(exc)}"
        ) from exc

    docker = docker_available()
    print(f"docker: {'available' if docker else 'not available'}")

    if args.require == "docker" and not docker:
        raise CliError(
            EXIT_UNAVAILABLE,
            "no Docker daemon answered, and --require docker was given",
            hint=(
                "start Docker, or accept SubprocessSandbox and understand that "
                "it is not a security boundary"
            ),
        )
    if docker and args.require != "subprocess":
        print("backend: DockerSandbox, network none, credentials outside")
        return EXIT_OK

    from rlm0.sandbox import SubprocessSandbox

    try:
        box = SubprocessSandbox()
    except SandboxUnavailableError as exc:
        raise CliError(EXIT_UNAVAILABLE, _detail(exc)) from exc
    try:
        result = box.execute("print(1 + 1)", timeout_s=10.0)
    finally:
        box.close()
    if not result.ok:
        raise CliError(
            EXIT_UNAVAILABLE,
            "the subprocess sandbox started but could not run a trivial "
            f"statement: {_detail(RuntimeError(result.stderr))}",
        )
    print("backend: SubprocessSandbox, usable")
    print(
        "warning: SubprocessSandbox is NOT a security boundary. It runs as you, "
        "on your filesystem, with your network. Do not point it at a context "
        "you did not write. See SECURITY.md."
    )
    return EXIT_OK


# -- argument parsing ---------------------------------------------------


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime")
    group.add_argument(
        "--provider",
        choices=("anthropic", "openai", "gemini", "fake"),
        default="anthropic",
        help=(
            "which provider to call; 'fake' calls nothing and exists to "
            "exercise the wiring (default: %(default)s)"
        ),
    )
    group.add_argument("--model", default="claude-sonnet-5", help="root model name")
    group.add_argument("--sub-model", default=None, help="model for sub-calls")
    group.add_argument(
        "--sandbox",
        choices=("docker", "microvm", "subprocess"),
        default="docker",
        help=(
            "sandbox backend; docker and microvm are isolation backends, "
            "subprocess is an explicit opt-in for contexts you wrote yourself "
            "(default: "
            "%(default)s)"
        ),
    )
    group.add_argument(
        "--policy",
        choices=("escalating", "fixed", "never"),
        default="escalating",
        help="how deep to go after a failed attempt (default: %(default)s)",
    )
    group.add_argument(
        "--max-depth", type=int, default=2, help="deepest attempt permitted"
    )
    group.add_argument(
        "--max-iterations", type=int, default=8, help="REPL turns per attempt"
    )
    group.add_argument(
        "--max-attempts", type=int, default=4, help="attempts per run, all depths"
    )
    group.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="output tokens per model call, not a budget ceiling",
    )
    group.add_argument(
        "--exec-timeout-s", type=float, default=30.0, help="timeout per code block"
    )
    group.add_argument(
        "--attempt-timeout-s",
        type=float,
        default=None,
        help="wall clock ceiling for one attempt",
    )


def _add_budget_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group(
        "budget, shared across the whole run and every depth in it"
    )
    group.add_argument("--max-usd", type=float, default=None)
    group.add_argument("--max-seconds", type=float, default=None)
    group.add_argument("--max-calls", type=int, default=None)
    group.add_argument(
        "--max-tokens", type=int, default=None, help="token ceiling for the whole run"
    )
    group.add_argument(
        "--unbounded",
        action="store_true",
        help=(
            "run with no ceiling of any kind; the run record says so in plain "
            "words and this conflicts with every ceiling above"
        ),
    )
    group.add_argument(
        "--max-unpriced-calls",
        type=int,
        default=0,
        help=(
            "how many calls may settle with no price before the budget refuses; "
            "unpriced is never counted as zero"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlm0",
        description=(
            "Run a recursive language model that attempts depth zero first, so "
            "every run reports what recursion cost and whether it helped."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rlm0 {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="answer one task against a file or directory of context",
        description=(
            "Answers one task and prints the run record. The last block of "
            "output is always Run.describe(), which names the cost of each "
            "attempt and the recursion verdict."
        ),
    )
    run.add_argument("task", help="the question to answer")
    run.add_argument(
        "--context",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="a file or directory of context; may be given more than once",
    )
    run.add_argument(
        "--glob",
        default="**/*",
        help="pattern used when --context names a directory (default: %(default)s)",
    )
    run.add_argument(
        "--context-limit-mb",
        type=float,
        default=_DEFAULT_CONTEXT_LIMIT_MB,
        help="refuse to load more context than this (default: %(default)s)",
    )
    run.add_argument(
        "--record",
        type=Path,
        default=None,
        metavar="PATH",
        help="write the full run accounting to this file as JSON",
    )
    _add_runtime_options(run)
    _add_budget_options(run)
    run.set_defaults(handler=cmd_run)

    evaluate = sub.add_parser(
        "eval",
        help="run the evaluation harness over a generated corpus",
        description=(
            "Generates a self-checking corpus, solves every sample, and prints "
            "the result table. The table refuses to render without its depth "
            "zero row."
        ),
    )
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--samples-per-family", type=int, default=2)
    evaluate.add_argument(
        "--out",
        type=Path,
        default=Path("runs/latest"),
        help="directory for records and the manifest (default: %(default)s)",
    )
    evaluate.add_argument(
        "--no-resume",
        action="store_true",
        help="start over rather than continuing an interrupted run",
    )
    _add_runtime_options(evaluate)
    _add_budget_options(evaluate)
    evaluate.set_defaults(handler=cmd_eval)

    cost = sub.add_parser(
        "cost",
        help="show what a configuration could cost before spending anything",
        description=(
            "Prices the worst case a configuration permits. An unpriced model "
            "is reported as unpriced and exits non-zero, never as zero."
        ),
    )
    cost.add_argument("--fanout", type=int, default=4, help="sub-calls per iteration")
    cost.add_argument("--input-tokens", type=int, default=4000)
    cost.add_argument("--output-tokens", type=int, default=1000)
    _add_runtime_options(cost)
    _add_budget_options(cost)
    cost.set_defaults(handler=cmd_cost)

    sandbox = sub.add_parser(
        "sandbox",
        help="check that a sandbox is available and say which backend",
        description=(
            "Reports which sandbox backend this machine can provide. Docker is "
            "the only one that is a boundary."
        ),
    )
    sandbox.add_argument(
        "--require",
        choices=("any", "docker", "microvm", "subprocess"),
        default="any",
        help="fail unless this backend is the one available",
    )
    sandbox.set_defaults(handler=cmd_sandbox)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse, dispatch, and turn every failure into an exit code and a reason."""
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw)
    handler: Callable[[argparse.Namespace, Sequence[str]], int] = args.handler
    try:
        return handler(args, raw)
    except CliError as exc:
        print(f"rlm0: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"  hint: {exc.hint}", file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        print("rlm0: interrupted", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
