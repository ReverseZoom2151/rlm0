"""Reading fields off objects whose types this package deliberately does not know.

Both SDKs are optional imports, so nothing here can be typed against them
without making them mandatory. That leaves attribute access on values typed as
object, which mypy will not allow directly, so the accessors live in one place
and do the narrowing once.

Mappings are accepted alongside attribute objects because a response body is a
mapping before an SDK wraps it, and because a test double that is a plain dict
is a far better test double than one that reimplements a class hierarchy it
does not control. A double that has to mirror the SDK's object model tends to
mirror the author's belief about the SDK rather than the SDK.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["field", "int_field", "str_field"]


def field(obj: object, name: str, default: object = None) -> object:
    """One field, by attribute or by key, without asserting which shape it is."""
    if isinstance(obj, Mapping):
        from_mapping: object = obj.get(name, default)
        return from_mapping
    from_attribute: object = getattr(obj, name, default)
    return from_attribute


def int_field(obj: object, name: str, default: int = 0) -> int:
    """A non-negative integer field, or the default when it is absent or None.

    Deliberately strict about type rather than coercing. A usage field that
    arrives as a string is a signal that the response is not the shape this
    layer thinks it is, and quietly calling int() on it would hide that.
    """
    value = field(obj, name, None)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def str_field(obj: object, name: str, default: str = "") -> str:
    """A string field, or the default when it is absent, None or another type."""
    value = field(obj, name, None)
    if isinstance(value, str):
        return value
    return default
