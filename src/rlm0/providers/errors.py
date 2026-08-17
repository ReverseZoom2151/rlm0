"""Failures this layer raises, named so a caller can tell them apart.

Two kinds matter here. A missing SDK is a setup problem the user can fix in
one command, so it should read as an instruction rather than as a stack trace
through somebody else's import machinery. A response that does not carry usage
is something else entirely: it is the one case where the cheap move would be to
report zeros and carry on, which is exactly the failure this project exists to
remove, so it is an error rather than a default.
"""

from __future__ import annotations

__all__ = [
    "ProviderDependencyError",
    "ProviderResponseError",
]


class ProviderDependencyError(ImportError):
    """An optional provider SDK is not installed.

    Subclasses ImportError so that code written to catch import failures still
    catches it, while the message says what to install. The alternative, a bare
    ImportError from three frames inside an import statement, tells a user that
    something is broken but not that the fix is one pip command.
    """


class ProviderResponseError(RuntimeError):
    """The provider returned something this layer refuses to interpret.

    Raised when a response carries no usage object. Reporting zero tokens
    instead would make the call free in every downstream total, and a budget
    checked against a total that silently omits calls is worse than no budget,
    because it is believed.
    """
