"""Kokoro TTS adapter — the default free, local text-to-speech.

Kokoro is an 82M-param, Apache-2.0 model (cleanest commercial license among open TTS — the
actively-maintained Piper fork is GPL-3.0) with 54 preset voices at 24 kHz. It runs on CPU;
first-audio latency on CPU is ~1-3s, which we hide by having the *pipeline* feed us one
SENTENCE at a time so the user hears the first sentence while the rest is still generating.

Install (only when you start the voice TODOs):  pip install -r requirements-voice.txt
First run downloads the model weights (~file in HF cache).

Docs: https://github.com/hexgrad/kokoro
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import numpy as np

from ..voice_port import TTS_SAMPLE_RATE, AudioChunk, TTSPort


class KokoroTTS(TTSPort):
    name = "kokoro"
    ready = True

    # Kokoro voice ids: first letter = language (a=American English), rest = voice name.
    def __init__(self, default_voice: str = "af_heart", lang_code: str = "a") -> None:
        from kokoro import KPipeline  # lazy import — heavy, optional

        self.default_voice = default_voice
        self._pipeline = KPipeline(lang_code=lang_code)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        # KPipeline yields (graphemes, phonemes, audio) per internal segment. `audio` is a
        # float32 numpy array in [-1, 1] at 24 kHz; convert to PCM16 bytes for the wire.
        for _gs, _ps, audio in self._pipeline(text, voice=voice or self.default_voice):
            pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            yield AudioChunk(pcm=pcm16, sample_rate=TTS_SAMPLE_RATE)
        yield AudioChunk(done=True)
