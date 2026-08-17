"""Providers: the only place in this project that talks to a model API.

Everything above this package takes an `LMClient` and never learns which
service is behind it. That seam is not architectural taste. The survey behind
this project found provider selection hardcoded three layers down more than
once, and in one case an entire recursion path disabled for every remote
environment because the wiring lived in a conditional nobody revisited.

The three obligations this layer carries, in the order they matter:

1. Report the provider's own token counts, never an estimate. An estimate that
   is wrong by a constant factor is indistinguishable from a correct one until
   the bill arrives, and several surveyed implementations ship exactly that.
2. Use prefix caching, because a fan-out shares a prefix by construction, and
   report what the provider said about it rather than what we asked for. See
   `anthropic_client` for the placement rules and the list of things a caller
   must not do.
3. Return None for the cost of a model no price table covers. Never zero.

The SDKs are optional extras. Import failures are turned into an install
instruction, and nothing in this package has a static dependency on either one,
which is why the full test suite runs with neither installed and no network.
"""

from __future__ import annotations

from rlm0.providers.anthropic_client import AnthropicClient
from rlm0.providers.errors import ProviderDependencyError, ProviderResponseError
from rlm0.providers.fake import (
    FAKE_PRICES,
    FakeCall,
    FakeClient,
    FakeReply,
    ScriptExhaustedError,
)
from rlm0.providers.gemini_client import GeminiClient
from rlm0.providers.openai_client import OpenAIClient
from rlm0.providers.params import (
    CHAT_STYLE,
    OPENAI_PARAMETER_STYLES,
    REASONING_STYLE,
    ChatParameterStyle,
    style_for,
)
from rlm0.providers.pricing import (
    ANTHROPIC_PRICES,
    DEFAULT_PRICES,
    OPENAI_PRICES,
    ModelPrice,
    PriceTable,
)
from rlm0.providers.retry import (
    RETRYABLE_STATUS_CODES,
    RetryPolicy,
    call_with_retry,
    is_retryable,
    retry_after_seconds,
)

__all__ = [
    "ANTHROPIC_PRICES",
    "CHAT_STYLE",
    "DEFAULT_PRICES",
    "FAKE_PRICES",
    "OPENAI_PARAMETER_STYLES",
    "OPENAI_PRICES",
    "REASONING_STYLE",
    "RETRYABLE_STATUS_CODES",
    "AnthropicClient",
    "ChatParameterStyle",
    "FakeCall",
    "FakeClient",
    "FakeReply",
    "GeminiClient",
    "ModelPrice",
    "OpenAIClient",
    "PriceTable",
    "ProviderDependencyError",
    "ProviderResponseError",
    "RetryPolicy",
    "ScriptExhaustedError",
    "call_with_retry",
    "is_retryable",
    "retry_after_seconds",
    "style_for",
]
