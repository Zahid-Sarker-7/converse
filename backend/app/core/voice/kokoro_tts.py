"""Kokoro TTS adapter — the default free, local text-to-speech.

Kokoro is an 82M-param, Apache-2.0 model (54 voices, 24 kHz). We run it through `kokoro-onnx`
(onnxruntime, no torch/spacy — the official `kokoro` package can't install on Python 3.14) with
the int8 weights `make voice` downloads to backend/models/. On CPU: ~0.5s to first audio, ~5x
realtime. The pipeline feeds us one sentence at a time, so the user hears sentence 1 while the
rest is still being generated.

Docs: https://github.com/thewh1teagle/kokoro-onnx · voices: hexgrad/Kokoro-82M VOICES.md
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from ..voice_port import TTS_SAMPLE_RATE, AudioChunk, TTSPort

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"  # backend/models
MODEL_PATH = MODELS_DIR / "kokoro-v1.0.int8.onnx"
VOICES_PATH = MODELS_DIR / "voices-v1.0.bin"


def available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("kokoro_onnx") is not None
        and MODEL_PATH.exists()
        and VOICES_PATH.exists()
    )


class KokoroTTS(TTSPort):
    name = "kokoro"
    ready = True

    # Voice ids: first letter = language (a = American English), second = gender, then a name.
    def __init__(self, default_voice: str = "af_heart") -> None:
        from kokoro_onnx import Kokoro  # lazy: optional dep

        self.default_voice = default_voice
        self._kokoro = Kokoro(str(MODEL_PATH), str(VOICES_PATH))

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        # ~0.5s of CPU per sentence: run it in a thread. On the event loop it would freeze mic
        # reading for that long and make barge-in impossible.
        audio, _sr = await asyncio.to_thread(
            self._kokoro.create, text, voice=voice or self.default_voice
        )
        yield AudioChunk(pcm=(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        yield AudioChunk(done=True, sample_rate=TTS_SAMPLE_RATE)
