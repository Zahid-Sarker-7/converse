"""Provider adapters + the registry that wires them up at startup."""

from __future__ import annotations

from ..model_port import ModelPort
from .echo_adapter import EchoAdapter
from .gemini_adapter import GeminiAdapter
from .groq_adapter import GroqAdapter

__all__ = [
    "ModelPort",
    "EchoAdapter",
    "GroqAdapter",
    "GeminiAdapter",
    "build_registry",
]


def build_registry(settings) -> dict[str, ModelPort]:
    """Assemble the active providers.

    `echo` always runs (no keys, no network). `groq` and `gemini` are registered only
    when their API key is present — so once you implement an adapter and drop a key in
    `.env`, it shows up in /health and the UI automatically.
    """
    registry: dict[str, ModelPort] = {"echo": EchoAdapter()}

    if settings.groq_api_key:
        registry["groq"] = GroqAdapter(
            api_key=settings.groq_api_key, default_model=settings.groq_model
        )
    if settings.google_api_key:
        registry["gemini"] = GeminiAdapter(
            api_key=settings.google_api_key, default_model=settings.gemini_model
        )
    return registry
