"""The command line, exercised without a network, a key or a real model.

`rlm0.assembly` is stubbed in `sys.modules` rather than imported, for two
reasons. It keeps the suite runnable while that module is still being written,
and it makes the seam explicit: the CLI is only allowed to know that some
callable in that module builds something with a `complete` method, and a test
that supplies exactly that is a test of the contract rather than of the
implementation behind it.

The assertion that matters most here is that the run output ends with the
`Run.describe()` block. Everything else about this CLI is convenience; that is
the project's argument reaching a terminal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from rlm0 import cli
from rlm0.budget import RunBudget
from rlm0.run import Attempt, CallRecord, Outcome, Role, Run, TokenUsage

SECRET = "correct-horse-battery-staple"


def _run(*, answered: bool = True) -> Run:
    call = CallRecord(
        role=Role.ROOT,
        depth=0,
        model="fake-model",
        usage=TokenUsage(input_tokens=100, output_tokens=10),
        wall_clock_s=0.5,
        cost_usd=0.01,
    )
    attempt = Attempt(
        max_depth=0,
        outcome=Outcome.ANSWERED if answered else Outcome.ITERATIONS_EXHAUSTED,
        calls=(call,),
        wall_clock_s=0.5,
        answer="42" if answered else None,
    )
    return Run(
        task="a task",
        attempts=(attempt,),
        budget_summary="max $0.50, 60s, 20 calls",
    )


class _FakeRuntime:
    def __init__(self, run: Run, **options: Any) -> None:
        self._run = run
        self.options = options
        self.seen: list[tuple[str, int]] = []

    def complete(self, task: str, context: str = "") -> Run:
        self.seen.append((task, len(context)))
        return self._run


def _install_assembly(
    monkeypatch: pytest.MonkeyPatch,
    *,
    run: Run | None = None,
    raises: Exception | None = None,
) -> list[_FakeRuntime]:
    """Put a stub `rlm0.assembly` in place and collect the runtimes it builds."""
    built: list[_FakeRuntime] = []
    payload = run if run is not None else _run()

    def build_rlm(**options: Any) -> _FakeRuntime:
        if raises is not None:
            raise raises
        runtime = _FakeRuntime(payload, **options)
        built.append(runtime)
        return runtime

    def default_budget() -> RunBudget:
        return RunBudget(max_usd=1.0)

    module = ModuleType("rlm0.assembly")
    module.build_rlm = build_rlm  # type: ignore[attr-defined]
    module.default_budget = default_budget  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rlm0.assembly", module)
    return built


def _argv(*extra: str) -> list[str]:
    """A run command that names the provider that calls nothing."""
    return ["run", "a task", "--provider", "fake", *extra]


# -- context loading ----------------------------------------------------


def test_context_names_every_document(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta", encoding="utf-8")
    text, n_files = cli.load_context([tmp_path], pattern="**/*", limit_mb=1.0)
    assert n_files == 2
    assert text.count("DOCUMENT ") == 2
    assert "alpha" in text and "beta" in text


def test_context_limit_refuses_and_does_not_echo(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text(SECRET * 1000, encoding="utf-8")
    with pytest.raises(cli.CliError) as caught:
        cli.load_context([tmp_path], pattern="**/*", limit_mb=0.001)
    assert caught.value.code == cli.EXIT_CONFIG
    assert SECRET not in str(caught.value)


def test_missing_context_path_is_a_configuration_error(tmp_path: Path) -> None:
    with pytest.raises(cli.CliError) as caught:
        cli.load_context([tmp_path / "nope"], pattern="**/*", limit_mb=1.0)
    assert caught.value.code == cli.EXIT_CONFIG


# -- run ----------------------------------------------------------------


def test_run_ends_with_the_describe_block(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _install_assembly(monkeypatch)
    (tmp_path / "ctx.txt").write_text("some context", encoding="utf-8")

    code = cli.main(_argv("--context", str(tmp_path)))

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert out.startswith("42")
    assert out.rstrip().endswith(_run().describe().splitlines()[-1])
    assert "recursion not attempted" in out


def test_run_without_an_answer_exits_non_zero_but_still_reports(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install_assembly(monkeypatch, run=_run(answered=False))

    code = cli.main(_argv())

    out = capsys.readouterr().out
    assert code == cli.EXIT_FAILED
    assert "no answer" in out
    assert "budget: max $0.50" in out


def test_run_writes_the_record_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_assembly(monkeypatch)
    record = tmp_path / "nested" / "run.json"

    assert cli.main(_argv("--record", str(record))) == cli.EXIT_OK
    assert '"budget_summary"' in record.read_text(encoding="utf-8")


def test_a_failing_runtime_never_echoes_the_context(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _install_assembly(monkeypatch, raises=RuntimeError(f"boom {SECRET}"))
    (tmp_path / "ctx.txt").write_text(SECRET, encoding="utf-8")

    code = cli.main(_argv("--context", str(tmp_path)))

    captured = capsys.readouterr()
    assert code == cli.EXIT_FAILED
    assert "boom" in captured.err
    assert SECRET not in captured.out


def test_a_missing_sandbox_is_its_own_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from rlm0.ports import SandboxUnavailableError

    _install_assembly(monkeypatch, raises=SandboxUnavailableError("no docker"))

    code = cli.main(_argv())

    assert code == cli.EXIT_UNAVAILABLE
    assert "no docker" in capsys.readouterr().err


def test_missing_assembly_is_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "rlm0.assembly", None)

    code = cli.main(_argv())

    assert code == cli.EXIT_CONFIG
    assert "rlm0.assembly" in capsys.readouterr().err


def test_the_ceilings_reach_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    built = _install_assembly(monkeypatch)

    assert cli.main(_argv("--max-usd", "0.25", "--max-calls", "6")) == cli.EXIT_OK

    budget = built[0].options["budget"]
    assert (budget.max_usd, budget.max_calls) == (0.25, 6)
    assert budget.max_seconds is None


def test_no_ceiling_named_still_bounds_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _install_assembly(monkeypatch)

    assert cli.main(_argv()) == cli.EXIT_OK
    assert built[0].options["budget"].max_usd == 1.0


def test_unbounded_says_so_and_conflicts_with_a_ceiling(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    built = _install_assembly(monkeypatch)

    assert cli.main(_argv("--unbounded")) == cli.EXIT_OK
    assert built[0].options["budget"].summary().startswith("UNBOUNDED")

    code = cli.main(_argv("--unbounded", "--max-usd", "1.0"))
    assert code == cli.EXIT_CONFIG
    assert "--max-usd" in capsys.readouterr().err


def test_the_sandbox_choice_is_passed_through_unsoftened(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _install_assembly(monkeypatch)

    assert cli.main(_argv("--sandbox", "subprocess")) == cli.EXIT_OK
    assert built[0].options["sandbox"] == "subprocess"
    # There is no auto value to fall back to a non boundary with.
    with pytest.raises(SystemExit):
        cli.main(_argv("--sandbox", "auto"))


def test_require_microvm_never_succeeds_just_because_docker_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rlm0.sandbox as sandbox

    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "microvm_available", lambda: False)

    code = cli.main(["sandbox", "--require", "microvm"])

    captured = capsys.readouterr()
    assert code == cli.EXIT_UNAVAILABLE
    assert "microVM runtime" in captured.err
    assert "DockerSandbox" not in captured.out


def test_require_microvm_reports_the_registered_backend(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import rlm0.sandbox as sandbox

    monkeypatch.setattr(sandbox, "docker_available", lambda: True)
    monkeypatch.setattr(sandbox, "microvm_available", lambda: True)

    assert cli.main(["sandbox", "--require", "microvm"]) == cli.EXIT_OK
    assert "MicroVMSandbox" in capsys.readouterr().out


# -- cost ---------------------------------------------------------------


def test_worst_case_counts_every_attempt() -> None:
    root, sub = cli.worst_case_calls(max_iterations=2, max_depth=1, fanout=3)
    # Depth zero attempt: 2 root calls. Depth one attempt: 2 root plus 6 sub.
    assert (root, sub) == (4, 6)


def test_cost_prices_a_known_model(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "cost",
            "--model",
            "claude-sonnet-5",
            "--max-depth",
            "0",
            "--max-iterations",
            "1",
            "--input-tokens",
            "1000000",
            "--output-tokens",
            "0",
        ]
    )
    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "total: $3.0000" in out


def test_cost_refuses_an_unpriced_model(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["cost", "--model", "some-unlisted-model"])

    captured = capsys.readouterr()
    assert code == cli.EXIT_CONFIG
    assert "unpriced" in captured.out
    assert "$0.0000" not in captured.out


def test_cost_says_when_the_ceiling_would_bind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli.main(
        [
            "cost",
            "--model",
            "claude-sonnet-5",
            "--max-usd",
            "0.0001",
            "--input-tokens",
            "100000",
        ]
    )
    assert "exceeds --max-usd" in capsys.readouterr().out


# -- benchmarks and preflight -----------------------------------------


def test_benchmark_list_is_local_and_does_not_need_a_provider(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["benchmark", "--list"])

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert "anomalyxl-local" in captured.out
    assert "oolong-synth" in captured.out
    assert "AGGBench" in captured.out


def test_benchmark_missing_data_stops_before_runtime_assembly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "rlm0.assembly", None)

    code = cli.main(
        [
            "benchmark",
            "oolong-synth",
            "--data-root",
            str(tmp_path / "missing"),
            "--provider",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert code == cli.EXIT_CONFIG
    assert "to obtain it, run:" in captured.err
    assert "rlm0.assembly" not in captured.err


def test_local_anomalyxl_requires_its_manifest_before_runtime_assembly(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "rlm0.assembly", None)

    code = cli.main(
        [
            "benchmark",
            "anomalyxl-local",
            "--data-root",
            str(tmp_path / "missing"),
            "--provider",
            "fake",
        ]
    )

    captured = capsys.readouterr()
    assert code == cli.EXIT_CONFIG
    assert "manifest.json" in captured.err
    assert "rlm0.assembly" not in captured.err


def test_doctor_downloads_nothing_and_does_not_require_a_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = cli.main(["doctor"])

    captured = capsys.readouterr()
    assert code == cli.EXIT_OK
    assert "credentials are read only" in captured.out
    assert "downloads nothing" in captured.out


# -- parser -------------------------------------------------------------


@pytest.mark.parametrize(
    "command", ["run", "eval", "benchmark", "cost", "sandbox", "doctor"]
)
def test_every_subcommand_has_help(command: str) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([command, "--help"])
    assert caught.value.code == 0


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == cli.EXIT_USAGE
