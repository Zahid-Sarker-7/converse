"""The Voice port — the Phase-2 twin of `model_port.py`.

Speech is two separable jobs, so there are two ports (each with interchangeable adapters):

    STTPort  — speech  -> text   (adapters: echo, Groq Whisper, …)
    TTSPort  — text    -> speech  (adapters: echo, Kokoro, …)

Everything above these ports (the voice pipeline, the WebSocket route) depends ONLY on
the interface, never on a vendor SDK — identical to how the orchestrator depends on
`ModelPort` and never imports `groq`/`google-genai` directly. Swapping Groq-Whisper for
Deepgram, or Kokoro for Cartesia, is then a config change, not a code change.

AUDIO CONVENTIONS (fix these once so adapters/pipeline agree)
  - Mic IN  : 16 kHz, mono, signed 16-bit little-endian PCM (what Silero VAD / Smart-Turn /
              Whisper all expect). Raw bytes, no container/header.
  - TTS OUT : 24 kHz, mono, signed 16-bit little-endian PCM (Kokoro's native rate).
  The frontend AudioWorklet resamples to/from the browser's device rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from enum import StrEnum

from pydantic import BaseModel, Field

# Canonical sample rates (see module docstring).
STT_SAMPLE_RATE = 16_000
TTS_SAMPLE_RATE = 24_000


class Word(BaseModel):
    """One transcribed word with timing — the raw material for delivery metrics
    (WPM, pauses, filler density all come from these start/end stamps)."""

    text: str
    start: float  # seconds from the start of the utterance
    end: float


class Transcript(BaseModel):
    """The result of transcribing one utterance (one learner turn)."""

    text: str
    words: list[Word] = Field(default_factory=list)  # empty if the adapter has no timestamps
    language: str | None = None


class AudioChunk(BaseModel):
    """One slice of synthesized speech, streamed out as it's generated."""

    pcm: bytes = b""  # raw PCM16 @ TTS_SAMPLE_RATE (see conventions)
    sample_rate: int = TTS_SAMPLE_RATE
    done: bool = False  # True on the final chunk of an utterance


class STTPort(ABC):
    """Provider-agnostic speech-to-text. One subclass per provider in `core/voice/`."""

    name: str = "base-stt"
    ready: bool = False

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int = STT_SAMPLE_RATE,
        language: str | None = None,
    ) -> Transcript:
        """Transcribe one complete utterance (raw PCM16 bytes) -> `Transcript`.

        This is deliberately *batch* (whole-utterance), not word-streaming: the free Groq
        Whisper endpoint returns the full transcript at once, and turn-detection already
        tells us where an utterance ends. True streaming STT (Deepgram) can implement the
        same port later by buffering internally.
        """
        raise NotImplementedError


class TTSPort(ABC):
    """Provider-agnostic text-to-speech. One subclass per provider in `core/voice/`."""

    name: str = "base-tts"
    ready: bool = False

    @abstractmethod
    def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
    ) -> AsyncIterator[AudioChunk]:
        """Stream synthesized speech for `text`, chunk by chunk.

        Implement as an async generator (`async def` + `yield`). Yield `AudioChunk(pcm=...)`
        as audio is produced and a final `AudioChunk(done=True)`. Streaming (rather than
        returning one blob) is what lets the pipeline start playback on the first chunk and
        lets barge-in cancel mid-utterance.
        """
        raise NotImplementedError


class VoiceProvider(StrEnum):
    """Names used in config + the registry (parallels the model provider names)."""

    echo = "echo"
    groq = "groq"
    kokoro = "kokoro"
