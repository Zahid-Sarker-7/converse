"""Echo STT/TTS — the reference voice adapters, the twins of `EchoAdapter`.

No API key, no network, no model download: they let the whole voice pipeline (turn
detection, barge-in, metrics, the WebSocket) run and be tested at $0 with no keys — the
same role `EchoAdapter` plays for chat. Use them as the shape a real adapter must match.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ..voice_port import (
    TTS_SAMPLE_RATE,
    AudioChunk,
    STTPort,
    Transcript,
    TTSPort,
    Word,
)


class EchoSTT(STTPort):
    """Returns a fixed transcript regardless of audio — enough to exercise the pipeline.

    It fabricates plausible word timings (so the delivery-metrics TODO has something to
    chew on) proportional to the audio length it was handed.
    """

    name = "echo"
    ready = True

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
    ) -> Transcript:
        await asyncio.sleep(0)  # keep it a real coroutine/awaitable
        text = "echo one two three"
        tokens = text.split(" ")
        # Spread the fake words across the (approx) duration of the PCM16 audio we got.
        duration = max(len(audio) / (2 * sample_rate), 0.5)  # 2 bytes/sample
        step = duration / len(tokens)
        words = [
            Word(text=t, start=round(i * step, 3), end=round((i + 1) * step, 3))
            for i, t in enumerate(tokens)
        ]
        return Transcript(text=text, words=words, language=language or "en")


class EchoTTS(TTSPort):
    """Emits a short burst of silent PCM frames so playback/barge-in logic is exercisable
    without a TTS model. Swap in Kokoro for real audio."""

    name = "echo"
    ready = True

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        # ~0.6s of silence, delivered as 3 chunks so the pipeline sees a real stream.
        frame = b"\x00\x00" * (TTS_SAMPLE_RATE // 5)  # 0.2s of 16-bit silence
        for _ in range(3):
            await asyncio.sleep(0.05)
            yield AudioChunk(pcm=frame)
        yield AudioChunk(done=True)
