"""Gemini adapter — a free provider (Google AI Studio, Flash tier).

Uses the **Interactions API** (`client.interactions.create`), which is Google's GA,
recommended surface for new projects (the older `generate_content` is now "legacy").
System messages are hoisted into `system_instruction`; user/assistant turns map to the
`input` list (assistant -> "model_output"). Text arrives as `step.delta` events with
`delta.type == "text"` (reasoning arrives as `thought_signature` deltas, which we skip).

Docs: https://ai.google.dev/gemini-api/docs/text-generation (Interactions API) · SDK: https://googleapis.github.io/python-genai/
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel

from ..model_port import ChatMessage, ModelPort, Role, StreamChunk


class GeminiAdapter(ModelPort):
    name = "gemini"
    ready = True

    def __init__(self, api_key: str, default_model: str) -> None:
        from google import genai  # lazy import to keep app startup cheap

        self.api_key = api_key
        self.default_model = default_model
        self._client = genai.Client(api_key=api_key)

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        system_text = (
            "\n".join(m.content for m in messages if m.role == Role.system) or None
        )
        # Interactions `input`: a list of typed turns (plain-text only here; add
        # image/tool blocks when a later phase needs them).
        history = [
            {
                "type": "model_output" if m.role == Role.assistant else "user_input",
                "content": [{"type": "text", "text": m.content}],
            }
            for m in messages
            if m.role != Role.system
        ]
        stream = await self._client.aio.interactions.create(
            model=model or self.default_model,
            store=False,
            input=history,
            system_instruction=system_text,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            stream=True,
        )
        async for event in stream:
            if event.event_type == "error":
                # e.g. quota_exceeded — surface it instead of yielding an empty reply.
                err = event.error
                raise RuntimeError(
                    f"Gemini error [{getattr(err, 'code', '?')}]: {getattr(err, 'message', '')}"
                )
            if event.event_type == "step.delta" and event.delta.type == "text":
                yield StreamChunk(delta=event.delta.text)
        yield StreamChunk(done=True)

    async def structured_output(
        self,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        *,
        model: str | None = None,
    ):
        system_text = (
            "\n".join(m.content for m in messages if m.role == Role.system) or None
        )
        history = [
            {
                "type": "model_output" if m.role == Role.assistant else "user_input",
                "content": [{"type": "text", "text": m.content}],
            }
            for m in messages
            if m.role != Role.system
        ]
        output = await self._client.aio.interactions.create(
            model=model or self.default_model,
            store=False,
            input=history,
            system_instruction=system_text,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            }
        )
        return schema.model_validate_json(output.output_text)


