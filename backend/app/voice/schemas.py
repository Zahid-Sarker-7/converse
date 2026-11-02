"""Voice-pipeline data contracts: the events sent over the WebSocket + delivery metrics.

`DeliveryMetrics` is kept SEPARATE from `agents/schemas.Scorecard` on purpose: the Scorecard
doubles as the LLM-judge's JSON schema, and bolting delivery fields onto it would pollute that
contract. The frontend renders the two side by side on one scorecard view.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class DeliveryMetrics(BaseModel):
    """How the learner *spoke* (not what they said) — derived from word-timestamped STT.

    Benchmarks (sales/interview corpora; recalibrate per scenario):
      talk_ratio ~0.43 · fillers < 8/min · WPM 140-160.
    """

    wpm: float = Field(0.0, description="words per minute across the learner's speaking time")
    filler_count: int = Field(0, description="um/uh/like/you-know/… occurrences")
    fillers_per_min: float = 0.0
    talk_ratio: float = Field(
        0.0, description="learner speaking time / total conversation time (0..1)"
    )
    longest_pause_s: float = Field(0.0, description="longest within-turn silence, seconds")
    speaking_time_s: float = 0.0


# ── WebSocket event envelope (server -> client) ────────────────────────────────
# Tiny tagged-union of JSON messages interleaved with binary audio frames.
#   {"type":"state", "state":"listening|thinking|speaking"}
#   {"type":"partial_transcript", "text": "..."}      (optional, if STT gives interim)
#   {"type":"transcript", "role":"user|assistant", "text":"..."}
#   {"type":"metrics", "metrics": {...DeliveryMetrics...}}
#   {"type":"error", "message":"..."}
#   {"type":"interrupt"}                              barge-in: client must flush queued playback
# Binary WS frames = raw PCM16 @ 24 kHz TTS audio to play.


class VoiceEvent(BaseModel):
    type: str
    state: str | None = None
    role: str | None = None
    text: str | None = None
    message: str | None = None
    metrics: DeliveryMetrics | None = None
