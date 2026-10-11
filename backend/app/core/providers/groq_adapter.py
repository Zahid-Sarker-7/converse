"""Groq adapter — the default free provider (OpenAI-compatible chat completions).

Docs: https://console.groq.com/docs/text-chat · Models: https://console.groq.com/docs/models
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel, ValidationError

from ..model_port import ChatMessage, ModelPort, StreamChunk


class GroqAdapter(ModelPort):
    name = "groq"
    ready = True

    def __init__(self, api_key: str, default_model: str) -> None:
        from groq import AsyncGroq  # lazy import to keep app startup cheap

        self.api_key = api_key
        self.default_model = default_model
        self._client = AsyncGroq(api_key=api_key)

    async def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> AsyncIterator[StreamChunk]:
        stream = await self._client.chat.completions.create(
            messages=[{"role": m.role.value, "content": m.content} for m in messages],
            model=model or self.default_model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield StreamChunk(delta=delta)
        yield StreamChunk(done=True)

    async def structured_output(
        self,
        messages: Sequence[ChatMessage],
        schema: type[BaseModel],
        *,
        model: str | None = None,
    ) -> BaseModel:
        """Schema-constrained JSON via OpenAI-compatible `response_format`.

        Groq's json_schema mode guides but doesn't hard-guarantee the shape, so we
        validate and, on failure, retry once feeding back the exact error (TDD D5).
        """
        payload = [{"role": m.role.value, "content": m.content} for m in messages]
        fmt = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema.model_json_schema(),
            },
        }
        last_err: ValidationError | None = None
        for _ in range(2):
            completion = await self._client.chat.completions.create(
                messages=payload,
                model=model or self.default_model,
                temperature=0,
                response_format=fmt,
            )
            content = completion.choices[0].message.content or ""
            try:
                return schema.model_validate_json(content)
            except ValidationError as err:
                last_err = err
                payload = payload + [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": (
                            f"That did not match the schema: {err}. "
                            "Return ONLY valid JSON matching the schema, nothing else."
                        ),
                    },
                ]
        raise RuntimeError(f"Groq structured_output failed validation: {last_err}")
