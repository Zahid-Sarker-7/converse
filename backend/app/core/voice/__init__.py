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
    # Kokoro is a local model (no key) — register it only if the package is importable so a
    # fresh checkout without the voice deps still boots on echo.
    try:
        import importlib.util

        if importlib.util.find_spec("kokoro") is not None:
            from .kokoro_tts import KokoroTTS

            registry["kokoro"] = KokoroTTS(default_voice=settings.kokoro_voice)
    except Exception:  # noqa: BLE001 — never let an optional adapter break startup
        pass
    return registry
