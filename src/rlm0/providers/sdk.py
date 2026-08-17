"""Importing an optional SDK, and saying something useful when it is absent.

The SDKs are extras rather than dependencies, because the whole package should
install and its tests should pass on a machine with no API key and no network.
That makes the missing-SDK path a normal path rather than an exceptional one,
and a normal path deserves a message a user can act on.

The import is done through importlib and the result is typed as object, which
also means nothing in this package has a static dependency on either SDK. That
is what lets mypy run clean without the SDKs installed, and it is what lets the
tests substitute a plain object for a client without reimplementing a class
hierarchy they do not own.
"""

from __future__ import annotations

import importlib
from typing import Any

from rlm0.providers.errors import ProviderDependencyError

__all__ = ["load_sdk"]


def load_sdk(module_name: str, extra: str) -> Any:
    """Import an optional SDK, or raise with the command that installs it."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise ProviderDependencyError(
            f"the {module_name!r} package is required for this client but is "
            f"not installed. Install it with:\n"
            f"    pip install 'rlm0[{extra}]'\n"
            f"or, if you are not installing rlm0 from a package index:\n"
            f"    pip install {module_name}"
        ) from exc
