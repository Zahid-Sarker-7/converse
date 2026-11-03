"""Voice adapters + the registries that wire them up at startup.

Mirrors `core/providers/__init__.py`: `echo` STT/TTS always run (no keys, no downloads);
the real adapters register only when their prerequisite (a key, or the installed package) is
present — so dropping a Groq key or `pip install`-ing Kokoro makes them appear automatically.
"""

from __future__ import annotations

from ..voice_port import STTPort, TTSPort
from .echo_voice import EchoSTT, EchoTTS

__all__ = [
    "STTPort",
    "TTSPort",
    "EchoSTT",
    "EchoTTS",
    "build_stt_registry",
    "build_tts_registry",
]


def build_stt_registry(settings) -> dict[str, STTPort]:
    registry: dict[str, STTPort] = {"echo": EchoSTT()}
    if settings.groq_api_key:
        from .groq_whisper_stt import GroqWhisperSTT  # lazy: needs the `groq` package

        registry["groq"] = GroqWhisperSTT(
            api_key=settings.groq_api_key, default_model=settings.whisper_model
        )
    return registry


def build_tts_registry(settings) -> dict[str, TTSPort]:
    registry: dict[str, TTSPort] = {"echo": EchoTTS()}
    # Kokoro is local (no key): register it only when `kokoro-onnx` + its model files exist, so a
    # fresh checkout without `make voice` still boots on echo.
    from . import kokoro_tts

    if kokoro_tts.available():
        registry["kokoro"] = kokoro_tts.KokoroTTS(default_voice=settings.kokoro_voice)
    return registry
