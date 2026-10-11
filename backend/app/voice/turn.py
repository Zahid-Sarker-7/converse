"""Turn detection — HANDS-ON TODO #1 (the first of your three Phase-2 learning pieces).

Turn detection answers two different questions with two different tools:
    stage 1  Silero VAD        "is anyone speaking right now?"   (fast, acoustic)
    stage 2  Smart-Turn v3     "did they actually FINISH?"        (semantic, after a pause)
A VAD-only system cuts people off: "I need to think about that for a moment…" <pause> gets
treated as end-of-turn. The stage-2 model is what makes the agent wait through that pause.

This file ships a crude ENERGY+SILENCE-TIMER fallback so the pipeline runs today on echo.
Your job is to replace `_fallback_is_speech` + the end-of-turn decision with the real models.

WHAT THE PIPELINE EXPECTS FROM YOU (the contract — don't change it):
  - `feed(frame_pcm16: bytes) -> TurnEvent | None`  called for each ~20-30ms mic frame.
       returns  TurnEvent("speech_started")  on the first speech frame of an utterance,
                TurnEvent("speech_ended")    when the turn is complete (this triggers STT),
                None                          otherwise.
  - `is_speaking` property — True while the user is mid-utterance (barge-in reads this).
  - `reset()` — clear buffers between utterances.
"""

from __future__ import annotations

from array import array  # stdlib — PCM16 RMS without numpy (audioop was removed in 3.13)
from dataclasses import dataclass

from ..core.voice_port import STT_SAMPLE_RATE


def _rms_pcm16(frame: bytes) -> float:
    """Root-mean-square amplitude of signed 16-bit little-endian PCM, stdlib only."""
    if len(frame) < 2:
        return 0.0
    samples = array("h")  # signed short
    samples.frombytes(frame[: len(frame) - (len(frame) % 2)])
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


@dataclass
class TurnEvent:
    kind: str  # "speech_started" | "speech_ended"


class TurnDetector:
    """Streaming turn detector. Feed it mic frames; it tells you when a turn starts/ends."""

    def __init__(
        self,
        sample_rate: int = STT_SAMPLE_RATE,
        silence_ms: int = 700,
        energy_threshold: int = 500,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_ms = silence_ms
        self.energy_threshold = energy_threshold
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._silence_run_ms = 0
        self._utterance = bytearray()  # accumulates the current turn's audio for STT

    @property
    def is_speaking(self) -> bool:
        return self._in_speech

    @property
    def utterance_audio(self) -> bytes:
        return bytes(self._utterance)

    def _fallback_is_speech(self, frame: bytes) -> bool:
        """Crude acoustic VAD: RMS energy over a threshold. REPLACE with Silero VAD."""
        return _rms_pcm16(frame) >= self.energy_threshold

    def feed(self, frame: bytes) -> TurnEvent | None:
        """### TODO: Turn detection (Silero VAD + Smart-Turn v3)  [track: C-project] [status: todo]

        - What:  replace the energy/timer heuristic below with a real two-stage detector:
                 Silero VAD per frame (speech?), and when VAD goes quiet, ask Smart-Turn v3
                 whether the utterance is semantically COMPLETE before emitting "speech_ended".
                 Keep this exact `feed()` signature + the `is_speaking`/`utterance_audio` API.
        - Why:   this is THE gate that shapes the whole conversation — get it wrong and the
                 agent either interrupts the learner or sits there awkwardly. It's also the
                 signal barge-in (TODO #2) depends on. Highest-leverage piece in Phase 2.
        - How:   1. Silero: load once (`from silero_vad import load_silero_vad, VADIterator`
                    or the ONNX build); run it on 16 kHz frames; threshold ~0.5-0.7.
                 2. On the speech->silence edge, slice the last ≤8s of `_utterance`, feed it
                    to Smart-Turn v3 (16 kHz mono PCM). If it says "complete" (or the silence
                    exceeds a hard `stop_secs` fallback), emit TurnEvent("speech_ended").
                 3. Emit TurnEvent("speech_started") on the first speech frame; keep
                    appending frames to `self._utterance` while in speech (already wired).
        - Working with: `TurnDetector` here · the pipeline in `voice/pipeline.py` (consumer) ·
                 frame size is whatever the WS sends (~20-30ms); don't assume a fixed size.
        - Docs & SDKs:
                 Silero VAD   https://github.com/snakers4/silero-vad   (MIT, 1.8MB, ~1ms/frame)
                 Smart-Turn   https://github.com/pipecat-ai/smart-turn (BSD-2, ~8M, ~65ms CPU)
                 LiveKit turn https://docs.livekit.io/agents/logic/turns/turn-detector/
                 concept      https://livekit.com/blog/turn-detection-voice-agents-vad-endpointing-model-based-detection
        - Acceptance: `make test` → unskip test_turn_waits_through_a_thinking_pause in
                 tests/test_voice.py and make it green: a 700ms gap mid-sentence must NOT end
                 the turn, but a real end-of-thought must. Manually: say "let me think…" and it
                 should wait.
        """
        frame_ms = (len(frame) / 2) / self.sample_rate * 1000  # 2 bytes/sample, mono
        speech = self._fallback_is_speech(frame)

        if speech:
            self._silence_run_ms = 0
            self._utterance.extend(frame)
            if not self._in_speech:
                self._in_speech = True
                return TurnEvent("speech_started")
            return None

        # silence frame
        if self._in_speech:
            self._utterance.extend(frame)  # keep trailing silence in the clip for STT context
            self._silence_run_ms += frame_ms
            if self._silence_run_ms >= self.silence_ms:
                self._in_speech = False
                return TurnEvent("speech_ended")
        return None
