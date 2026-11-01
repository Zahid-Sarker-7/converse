"""Turn detection — two models answer two different questions:

    stage 1  Silero VAD     "is this 32ms chunk speech?"        every chunk, ~1ms
    stage 2  Smart-Turn v3  "is the turn a complete thought?"  once silence starts, ~15ms

VAD alone cuts people off: "Let me think about that…" <1s pause> looks exactly like the end of a
turn to an acoustic model. Smart-Turn hears that the phrase is unfinished and we keep waiting.

Both run torch-free via onnxruntime. Models live in backend/models/ (`make voice` fetches them).

    chunk ─► Silero p(speech) ≥ vad_threshold?
       yes ─► in a turn already? no → emit speech_started.  Reset silence clock.
       no  ─► (only while in a turn) silence clock += 32ms
                 ≥ check_after_ms, not yet checked this pause → Smart-Turn(whole turn)
                       p(complete) ≥ turn_threshold → emit speech_ended
                 ≥ max_silence_ms → emit speech_ended   (fallback: never wait forever)

The contract the pipeline relies on: `feed(frame) -> TurnEvent | None`, `is_speaking`,
`utterance_audio`, `reset()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

from ..core.voice_port import STT_SAMPLE_RATE
from .whisper_features import compute_whisper_log_mel_features

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # backend/models
SILERO_PATH = MODELS_DIR / "silero_vad.onnx"
SMART_TURN_PATH = MODELS_DIR / "smart-turn-v3.2-cpu.onnx"

CHUNK_SAMPLES = 512  # Silero's fixed input at 16kHz (= 32ms)
CHUNK_BYTES = CHUNK_SAMPLES * 2  # PCM16
CHUNK_MS = CHUNK_SAMPLES / STT_SAMPLE_RATE * 1000
SILERO_CONTEXT = 64  # samples of the previous chunk Silero wants prepended
TURN_WINDOW = 8 * STT_SAMPLE_RATE  # Smart-Turn looks at the last 8s of the turn

# Silero's RNN state drifts during long non-speech and then under-reports the next speech
# (Pipecat's finding). Reset it after ~2s of audio, but only at a run of clearly-silent chunks so
# we never reset mid-word.
VAD_RESET_EVERY_MS = 2000
VAD_RESET_QUIET_PROB = 0.1
VAD_RESET_QUIET_CHUNKS = 5  # ~160ms of p < 0.1


@cache
def _silero():
    import onnxruntime as ort  # lazy: only needed once a voice session runs

    return ort.InferenceSession(str(SILERO_PATH))


@cache
def _smart_turn():
    import onnxruntime as ort

    return ort.InferenceSession(str(SMART_TURN_PATH))


def _to_float(pcm16: bytes) -> np.ndarray:
    return np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0


@dataclass
class TurnEvent:
    kind: str  # "speech_started" | "speech_ended"


class TurnDetector:
    """Streaming turn detector. Feed it mic frames of any size; it emits turn start/end."""

    def __init__(
        self,
        vad_threshold: float = 0.5,
        turn_threshold: float = 0.5,
        check_after_ms: int = 200,  # silence before asking Smart-Turn
        max_silence_ms: int = 3000,  # give up waiting even if Smart-Turn says "incomplete"
        max_turn_ms: int = 60_000,  # hard cap: steady noise must not grow a turn forever
    ) -> None:
        # Calibration knobs: tune per scenario (a language learner pauses longer than a sales rep).
        self.vad_threshold = vad_threshold
        self.turn_threshold = turn_threshold
        self.check_after_ms = check_after_ms
        self.max_silence_ms = max_silence_ms
        self.max_turn_ms = max_turn_ms
        self.reset()

    def reset(self) -> None:
        self._buf = bytearray()  # frame bytes not yet consumed as a whole 512-sample chunk
        self._reset_vad()
        self._in_turn = False
        self._silence_ms = 0.0
        self._checked_this_pause = False
        self._utterance = bytearray()  # the whole current turn, handed to STT
        self._speech_ms = 0.0  # speech-only time in this turn (pauses excluded) — barge-in guard

    @property
    def is_speaking(self) -> bool:
        return self._in_turn

    @property
    def speech_ms(self) -> float:
        return self._speech_ms

    @property
    def utterance_audio(self) -> bytes:
        return bytes(self._utterance)

    def feed(self, frame: bytes) -> TurnEvent | None:
        # Frames arrive in whatever size the client sends; Silero needs exactly 512 samples.
        self._buf.extend(frame)
        event = None
        while len(self._buf) >= CHUNK_BYTES:
            chunk = bytes(self._buf[:CHUNK_BYTES])
            del self._buf[:CHUNK_BYTES]
            # one frame (~43ms) holds 1-2 chunks, so at most one event per frame in
            # practice; if two fire, the later one wins.
            event = self._feed_chunk(chunk) or event
            if (
                self._in_turn
                and len(self._utterance) / 2 / STT_SAMPLE_RATE * 1000 >= self.max_turn_ms
            ):
                event = self._end()
        return event

    def _feed_chunk(self, chunk: bytes) -> TurnEvent | None:
        if self._speech_prob(chunk) >= self.vad_threshold:
            self._silence_ms = 0.0
            self._checked_this_pause = False  # speech resumed → re-check at the next pause
            self._speech_ms += CHUNK_MS
            self._utterance.extend(chunk)
            if not self._in_turn:
                self._in_turn = True
                return TurnEvent("speech_started")
            return None

        if not self._in_turn:
            return None  # silence between turns: ignore

        self._utterance.extend(chunk)  # keep in-turn pauses so STT/metrics see real timing
        self._silence_ms += CHUNK_MS
        if self._silence_ms >= self.max_silence_ms:
            return self._end()
        if self._silence_ms >= self.check_after_ms and not self._checked_this_pause:
            self._checked_this_pause = True
            if self._turn_complete_prob() >= self.turn_threshold:
                return self._end()
        return None

    def _end(self) -> TurnEvent:
        self._in_turn = False
        return TurnEvent("speech_ended")

    def _reset_vad(self) -> None:
        self._vad_state = np.zeros((2, 1, 128), dtype=np.float32)  # Silero RNN memory
        self._vad_context = np.zeros((1, SILERO_CONTEXT), dtype=np.float32)
        self._since_vad_reset_ms = 0.0
        self._quiet_chunks = 0

    def _maybe_reset_vad(self, prob: float) -> None:
        self._since_vad_reset_ms += CHUNK_MS
        self._quiet_chunks = self._quiet_chunks + 1 if prob < VAD_RESET_QUIET_PROB else 0
        if (
            self._since_vad_reset_ms >= VAD_RESET_EVERY_MS
            and self._quiet_chunks >= VAD_RESET_QUIET_CHUNKS
        ):
            self._reset_vad()

    def _speech_prob(self, chunk: bytes) -> float:
        """Silero: prepend the previous chunk's tail, carry the RNN state forward."""
        x = np.concatenate([self._vad_context, _to_float(chunk)[None, :]], axis=1)
        out, self._vad_state = _silero().run(
            None,
            {"input": x, "state": self._vad_state, "sr": np.array(STT_SAMPLE_RATE, np.int64)},
        )
        self._vad_context = x[:, -SILERO_CONTEXT:]
        prob = float(out[0][0])
        self._maybe_reset_vad(prob)
        return prob

    def _turn_complete_prob(self) -> float:
        """Smart-Turn on the last 8s of the turn, zero-padded at the FRONT (speech at the end)."""
        audio = _to_float(bytes(self._utterance))[-TURN_WINDOW:]
        audio = np.pad(audio, (TURN_WINDOW - len(audio), 0))
        feats = compute_whisper_log_mel_features(audio)[None, :, :]  # (1, 80 mel bins, 800 frames)
        return float(_smart_turn().run(None, {"input_features": feats})[0][0].item())
