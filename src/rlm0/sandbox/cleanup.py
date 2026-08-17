"""Making sure nothing outlives the process that created it.

A leaked container per run is a real operational failure and not a tidiness
concern: the containers hold memory and pid budget, and a harness that sweeps
a benchmark leaks one per task until the host stops scheduling anything. The
ordinary exit path is `close()`, so this module exists for the paths that skip
it, which are an unhandled exception, a SIGINT from a person who changed their
mind, and a SIGTERM from whatever is supervising the run.

Signal handlers chain to whatever was installed before, because a library that
swallows the application's own SIGINT handling has traded one operational
surprise for another.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import signal
import threading
import weakref
from types import FrameType
from typing import Any, Protocol

__all__ = ["register", "unregister"]


class _Closeable(Protocol):
    def close(self) -> None: ...


_LOCK = threading.Lock()
_LIVE: weakref.WeakSet[Any] = weakref.WeakSet()
_PREVIOUS: dict[int, Any] = {}
_INSTALLED = False


def register(sandbox: _Closeable) -> None:
    """Track a sandbox until it closes itself or the process ends."""
    with _LOCK:
        _LIVE.add(sandbox)
        _install()


def unregister(sandbox: _Closeable) -> None:
    with _LOCK:
        _LIVE.discard(sandbox)


def _close_all() -> None:
    for sandbox in list(_LIVE):
        # Nothing useful is left to do at this point, and a raise here would
        # mask the signal or the exception that got us here.
        with contextlib.suppress(Exception):
            sandbox.close()


def _handle(signum: int, frame: FrameType | None) -> None:
    _close_all()
    previous = _PREVIOUS.get(signum)
    if callable(previous):
        previous(signum, frame)
        return
    if previous == signal.SIG_DFL:
        try:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        except (OSError, ValueError):  # pragma: no cover - platform dependent
            raise SystemExit(128 + signum) from None


def _install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True
    atexit.register(_close_all)
    for name in ("SIGTERM", "SIGINT"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            _PREVIOUS[int(signum)] = signal.getsignal(signum)
            signal.signal(signum, _handle)
        except (ValueError, OSError):
            # Only the main thread may install handlers, and some embedded
            # hosts forbid it entirely. atexit still covers the normal exit.
            _PREVIOUS.pop(int(signum), None)
