"""Which request parameters a chat model accepts, stated per model rather than guessed.

Reasoning models changed the shape of the request in two ways that produce a
400 rather than a degraded answer: the output cap moved from `max_tokens` to
`max_completion_tokens`, and `temperature` stopped being accepted at all.

The tempting fix is `if "o1" in model or model.startswith("gpt-5")`. A surveyed
repo does exactly that. It is wrong in both directions and gets more wrong over
time: a future model whose name does not contain the magic substring is sent
the old shape and fails, and any model whose name happens to contain it is sent
the new shape and also fails. Neither failure is visible until a request is
made, and by then the classification is buried under a provider error message
about an unrelated field.

So this is a registry keyed by exact model name. That has a real cost: a model
nobody has registered gets the default chat shape, and if it is in fact a
reasoning model the call fails. That failure is loud, immediate, names the
parameter, and is fixed by one line in a table the caller can pass in. A silent
misclassification is not fixable, because nobody knows it happened.

Note that the entries here are claims about request shape, not about price.
They are in a different file from the price table for that reason: being wrong
about a parameter produces an error, and being wrong about a price produces a
number.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "CHAT_STYLE",
    "OPENAI_PARAMETER_STYLES",
    "REASONING_STYLE",
    "ChatParameterStyle",
    "style_for",
]


@dataclass(frozen=True, slots=True)
class ChatParameterStyle:
    """The request-shape differences that matter for one family of models."""

    max_tokens_field: str = "max_tokens"
    supports_temperature: bool = True
    system_role: str = "system"


CHAT_STYLE = ChatParameterStyle()
"""The original chat completions shape, and the default for unknown models."""

REASONING_STYLE = ChatParameterStyle(
    max_tokens_field="max_completion_tokens",
    supports_temperature=False,
    system_role="developer",
)
"""The reasoning-model shape: renamed output cap, no sampling temperature.

The system role is renamed too. It is included here because it is the same
class of breakage, discovered the same way, and because a caller who registers
a new reasoning model should get all three corrections from one entry rather
than two of the three.
"""

OPENAI_PARAMETER_STYLES: dict[str, ChatParameterStyle] = {
    "o1": REASONING_STYLE,
    "o1-mini": REASONING_STYLE,
    "o1-preview": REASONING_STYLE,
    "o3": REASONING_STYLE,
    "o3-mini": REASONING_STYLE,
    "o4-mini": REASONING_STYLE,
    "gpt-5": REASONING_STYLE,
    "gpt-5-mini": REASONING_STYLE,
    "gpt-5-nano": REASONING_STYLE,
    "gpt-4o": CHAT_STYLE,
    "gpt-4o-mini": CHAT_STYLE,
    "gpt-4.1": CHAT_STYLE,
    "gpt-4.1-mini": CHAT_STYLE,
}


def style_for(
    model: str,
    styles: Mapping[str, ChatParameterStyle] | None = None,
    default: ChatParameterStyle = CHAT_STYLE,
) -> ChatParameterStyle:
    """The registered style for a model, or the default when it is not registered."""
    table = OPENAI_PARAMETER_STYLES if styles is None else styles
    return table.get(model, default)
