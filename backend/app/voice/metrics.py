"""Delivery metrics — HANDS-ON TODO #3 (the third Phase-2 learning piece).

"Delivery" = *how* the learner spoke, independent of what they said: pace (WPM), filler
density (um/uh/like), talk/listen balance, and pauses. This is a 2026 differentiator — most
coaches give one vague number; moment-level delivery feedback is what makes the scorecard feel
real. All of it falls out of the **word-timestamped** transcripts that Groq Whisper already
returns (`Transcript.words` — each has `start`/`end` in seconds).

The pipeline collects, per session, a list of the learner's `Transcript`s (their turns) and the
total wall-clock conversation duration, then calls `compute_delivery_metrics(...)`.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.voice_port import Transcript
from .schemas import DeliveryMetrics

# Starter filler lexicon — extend/localize (Banglish fillers differ!). Multi-word entries are
# matched as phrases, so order the detection accordingly.
FILLER_WORDS = {"um", "uh", "er", "ah", "like", "you know", "i mean", "sort of", "kind of"}


def compute_delivery_metrics(
    learner_turns: Sequence[Transcript],
    total_conversation_s: float,
) -> DeliveryMetrics:
    """### TODO: Delivery metrics from word timestamps  [track: C-project]  [status: todo]

    - What:  turn `learner_turns` (the learner's word-timestamped transcripts) + the total
             conversation duration into a filled-in `DeliveryMetrics`. Right now this returns
             an empty metrics object — implement each field.
    - Why:   it's the payoff of using a *word-timestamped* STT (Groq Whisper) over browser
             speech: real pace/pause/filler numbers, not guesses. Teaches you to reason about
             speech timing, and it's what lights up the voice scorecard.
    - How:   speaking_time_s = Σ (last word.end − first word.start) over each turn.
             wpm             = total_words / (speaking_time_s / 60).
             filler_count    = count tokens in FILLER_WORDS (lowercased, punctuation-stripped;
                               handle the multi-word phrases too).
             fillers_per_min = filler_count / (speaking_time_s / 60).
             talk_ratio      = speaking_time_s / total_conversation_s  (clamp 0..1).
             longest_pause_s = max gap between consecutive words' (end -> next.start) within a
                               turn (cross-turn gaps are listening time, not pauses — skip them).
             Guard every divide-by-zero (empty turns, zero duration) → return 0.0, never crash.
    - Working with: `Transcript`/`Word` (core/voice_port.py) · `DeliveryMetrics` (schemas.py) ·
             the pipeline passes you only the LEARNER's turns (role == user).
    - Docs & SDKs:  Groq word timestamps  https://console.groq.com/docs/speech-to-text
             benchmarks (recalibrate per scenario): talk ~0.43 · fillers <8/min · WPM 140-160.
    - Acceptance: `make test` → unskip test_delivery_metrics_from_timestamps and make it green
             (a hand-built transcript with known timings yields the expected WPM / filler /
             pause numbers). Then they render on the voice scorecard next to the rubric.
    """
    # TODO: replace this stub with the real computation described above.
    return DeliveryMetrics()
