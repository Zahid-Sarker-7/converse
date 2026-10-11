"""Groq Whisper STT adapter — the default free speech-to-text.

Groq serves OpenAI-compatible Whisper endpoints on LPU hardware (~200x realtime). We use
`whisper-large-v3-turbo`: free tier (20 RPM / 2000 RPD / ~8h audio-day, no card) and it
returns **word-level timestamps** via `verbose_json` — which the delivery-metrics TODO needs.

Note: this endpoint is *batch* (send a whole utterance, get the whole transcript) — there is
no token streaming. That's fine: turn detection decides where an utterance ends, then we
transcribe it in one shot (an hour of audio transcribes in ~15s, so a few-second turn is
near-instant).

Docs: https://console.groq.com/docs/speech-to-text
Model: https://console.groq.com/docs/model/whisper-large-v3-turbo
"""

from __future__ import annotations

import io
import wave

from ..voice_port import STT_SAMPLE_RATE, STTPort, Transcript, Word


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Groq's transcription endpoint wants a real audio file, not raw PCM. Wrap the raw
    PCM16 mono bytes in a minimal in-memory WAV container (no disk, no ffmpeg)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


class GroqWhisperSTT(STTPort):
    name = "groq"
    ready = True

    def __init__(self, api_key: str, default_model: str = "whisper-large-v3-turbo") -> None:
        from groq import AsyncGroq  # lazy import — keeps startup cheap / optional dep

        self.api_key = api_key
        self.default_model = default_model
        self._client = AsyncGroq(api_key=api_key)

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int = STT_SAMPLE_RATE,
        language: str | None = None,
    ) -> Transcript:
        wav_bytes = _pcm_to_wav(audio, sample_rate)
        resp = await self._client.audio.transcriptions.create(
            file=("utterance.wav", wav_bytes, "audio/wav"),
            model=self.default_model,
            response_format="verbose_json",  # needed for timestamps
            timestamp_granularities=["word"],  # -> per-word start/end
            language=language,  # None lets Whisper auto-detect
            temperature=0,
        )
        # `resp.words` is a list of {word, start, end} when word granularity is requested.
        words = [
            Word(
                text=w["word"] if isinstance(w, dict) else w.word,
                start=w["start"] if isinstance(w, dict) else w.start,
                end=w["end"] if isinstance(w, dict) else w.end,
            )
            for w in (getattr(resp, "words", None) or [])
        ]
        return Transcript(
            text=resp.text.strip(),
            words=words,
            language=getattr(resp, "language", None) or language,
        )
