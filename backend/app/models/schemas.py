"""Request/response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..core.model_port import ChatMessage


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str | None = None  # None -> use DEFAULT_PROVIDER
    model: str | None = None  # None -> provider's default_model
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
