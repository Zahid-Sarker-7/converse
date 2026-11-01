"""EchoAdapter — the reference implementation of ModelPort.

It needs no API key and makes no network call: it streams the user's last message
back, token by token. Use it as the worked example for how a real adapter is shaped —
the Groq and Gemini adapters follow the exact same contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from ..model_port import ChatMessage, ModelPort, Role, StreamChunk


class EchoAdapter(ModelPort):
    name = "echo"
    ready = True
    default_model = "echo-1"

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        last_user = next((m.content for m in reversed(messages) if m.role == Role.user), "")
        reply = f"echo> {last_user}" if last_user else "echo> (say something)"
        for token in reply.split(" "):
            await asyncio.sleep(0.03)  # simulate token-by-token latency
            yield StreamChunk(delta=token + " ")
        yield StreamChunk(done=True)
