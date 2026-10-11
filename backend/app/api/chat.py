"""Streaming chat route.

Resolves a provider from the registry and streams its output to the client as
Server-Sent Events (SSE). Each event is `data: {json}\\n\\n` where json is one of:
    {"delta": "..."}   incremental text
    {"done": true}     end of stream
    {"error": "..."}   an error (e.g. an unimplemented adapter — your TODO)

The route itself is provider-agnostic: it only knows about `ModelPort`.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..logging import get_logger
from ..models.schemas import ChatRequest

router = APIRouter()
log = get_logger("chat")


@router.post("/chat")
async def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    settings = request.app.state.settings
    registry: dict = request.app.state.registry

    provider_name = body.provider or settings.default_provider
    port = registry.get(provider_name)
    if port is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Provider '{provider_name}' is not configured. "
                f"Available: {list(registry)}"
            ),
        )

    async def event_stream():
        try:
            async for chunk in port.stream_chat(
                body.messages,
                model=body.model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            ):
                if chunk.delta:
                    yield f"data: {json.dumps({'delta': chunk.delta})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except NotImplementedError as exc:
            msg = str(exc) or "This capability is not implemented for this provider."
            yield f"data: {json.dumps({'error': msg})}\n\n"
        except Exception as exc:  # provider/network error — log server-side, surface cleanly
            log.error(
                "chat_stream_error",
                provider=provider_name,
                error=f"{type(exc).__name__}: {exc}",
                exc_info=True,
            )
            yield f"data: {json.dumps({'error': f'{type(exc).__name__}: {exc}'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
