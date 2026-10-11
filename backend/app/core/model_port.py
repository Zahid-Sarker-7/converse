"""The Model port — the heart of the "one engine, many providers" design.

Every LLM provider is an *adapter* that implements `ModelPort`. The rest of the app
(routes, agents, evals) depends only on this interface and never imports a vendor SDK
directly. Swapping Groq <-> Gemini <-> Claude is then a config change, not a code change.

This is the hexagonal / ports-and-adapters pattern: `ModelPort` is the port; the classes
in `providers/` are the adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from enum import StrEnum

from pydantic import BaseModel


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    """One turn in a conversation — deliberately provider-neutral. Each adapter maps
    this into whatever shape its SDK wants (OpenAI-style dicts, Gemini `contents`, …)."""

    role: Role
    content: str


class StreamChunk(BaseModel):
    """One incremental piece of a streamed response."""

    delta: str = ""  # the new text in this chunk
    done: bool = False  # True on the final chunk


class ModelPort(ABC):
    """Provider-agnostic interface. Implement one subclass per provider in `providers/`."""

    name: str = "base"
    ready: bool = False  # flip to True once the adapter is fully implemented
    default_model: str | None = None

    @abstractmethod
    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        """Stream a chat completion token-by-token.

        Implement this as an *async generator* (`async def` with `yield`). Yield a
        `StreamChunk(delta=...)` per token, and a final `StreamChunk(done=True)`.
        """
        raise NotImplementedError

    async def structured_output(
        self,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        """Return a validated instance of `schema` (schema-constrained JSON output).

        Optional capability — the default raises so callers get a clear signal. Override
        it in an adapter when a feature needs it (e.g. the Phase 1 Evaluator / LLM-as-judge:
        per-criterion scores + evidence as structured JSON).
        """
        raise NotImplementedError(f"{self.name} does not implement structured_output.")
