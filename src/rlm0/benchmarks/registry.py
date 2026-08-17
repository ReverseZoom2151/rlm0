"""Which public benchmarks this project can run, and which it deliberately cannot.

The second list is the one that matters. It is easy to publish a benchmark
integration that is a name in a dictionary and a prompt string with no data
behind it, and the survey behind this project found several: one shipped
twenty-seven cases every one of which had a null expected answer, so its
grader could never fire and its score was a formality. The cure is that a
benchmark is either adapted, with a pinned revision and a scorer that has been
checked against the published one, or it is listed here with the reason it was
not, in a form somebody can argue with.

Nothing in this module touches the network or the disk. Listing what exists is
separate from having it, so a user can find out what a benchmark needs before
committing to downloading it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rlm0.benchmarks.niah import RulerNiah
from rlm0.benchmarks.oolong import OolongReal, OolongSynth
from rlm0.benchmarks.suite import BenchmarkAdapter

__all__ = [
    "ADAPTERS",
    "NOT_ADAPTED",
    "UnadaptedBenchmark",
    "describe_catalogue",
    "get",
    "names",
]


ADAPTERS: Mapping[str, Callable[[], BenchmarkAdapter]] = {
    "oolong-synth": OolongSynth,
    "oolong-real": OolongReal,
    "ruler-s-niah": RulerNiah,
}


@dataclass(frozen=True, slots=True)
class UnadaptedBenchmark:
    """A benchmark that was investigated and not adapted, with the reason.

    `blocker` is written to be falsifiable. "Could not obtain it" is not a
    reason; "the only distribution is an anonymous preview link with no
    revision to pin" is, and it stops being true the moment that changes.
    """

    name: str
    paper: str
    why_it_matters: str
    blocker: str


NOT_ADAPTED: tuple[UnadaptedBenchmark, ...] = (
    UnadaptedBenchmark(
        name="AGGBench",
        paper="arXiv:2602.01355",
        why_it_matters=(
            "formalises entity-level aggregation with a strict completeness "
            "requirement and reports that Text-to-SQL and RAG fail it, which "
            "makes it the natural target for this project's aggregation claim"
        ),
        blocker=(
            "as of August 2026 both arXiv versions point only at an anonymous "
            "4open.science preview. There is no repository, no release and no "
            "revision to pin, so a number measured against it could not be "
            "traced to the data that produced it. Adapting it would mean "
            "hashing whatever a preview link served on the day, which is the "
            "kind of untraceable figure this package exists to avoid."
        ),
    ),
    UnadaptedBenchmark(
        name="LOCA-bench",
        paper="arXiv:2602.07962",
        why_it_matters=(
            "VISTA reports 50.7 on it against 22.7 for ReAct, which is the "
            "comparison this project will be asked about"
        ),
        blocker=(
            "it is an interactive agent environment rather than a corpus: "
            "context grows through live tool calls against mock MCP servers "
            "for Calendar, Canvas and BigQuery, and running it needs provider "
            "credentials. The harness interface here hands a solver a question "
            "and a fixed set of documents, so adapting LOCA-bench means "
            "building an environment-driving solver, not a dataset adapter. "
            "That is a genuine piece of work and it does not belong in this "
            "package."
        ),
    ),
)


def names() -> tuple[str, ...]:
    return tuple(sorted(ADAPTERS))


def get(name: str) -> BenchmarkAdapter:
    """Construct an adapter by name, with its defaults."""
    try:
        factory = ADAPTERS[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; this package adapts {names()}. "
            f"Benchmarks considered and not adapted: "
            f"{tuple(b.name for b in NOT_ADAPTED)}, with reasons in "
            "rlm0.benchmarks.registry.NOT_ADAPTED."
        ) from None
    return factory()


def describe_catalogue() -> str:
    """Both lists, rendered, so the gaps are as visible as the coverage."""
    lines = ["adapted:"]
    for name in names():
        adapter = get(name)
        requirement = adapter.requirement(
            split="test" if name == "ruler-s-niah" else "validation"
        )
        lines.append(f"  {name:<16}{requirement.source} @ {requirement.revision[:12]}")
    lines.append("")
    lines.append("considered and not adapted:")
    for entry in NOT_ADAPTED:
        lines.append(f"  {entry.name} ({entry.paper})")
        lines.append(f"    matters because: {entry.why_it_matters}")
        lines.append(f"    blocked by: {entry.blocker}")
    return "\n".join(lines)
