"""Delivery metrics — pace, fillers, talk ratio and pauses from word-timestamped STT.

"Delivery" = *how* the learner spoke, independent of what they said: pace (WPM), filler
density (um/uh/like), talk/listen balance, and pauses. This is a 2026 differentiator — most
coaches give one vague number; moment-level delivery feedback is what makes the scorecard feel
real. All of it falls out of the **word-timestamped** transcripts that Groq Whisper already
returns (`Transcript.words` — each has `start`/`end` in seconds).

The pipeline collects, per session, a list of the learner's `Transcript`s (their turns) and the
total time the agent spoke, then calls `compute_delivery_metrics(...)`.
"""

from __future__ import annotations

import string
from collections.abc import Sequence

from ..core.voice_port import Transcript
from .schemas import DeliveryMetrics

# Starter filler lexicon — extend/localize (Banglish fillers differ!). Multi-word entries are
# matched as phrases, so order the detection accordingly.
# ponytail: "like"/"kind of" are counted even when used literally ("I like Python") —
# context-aware filler detection is the upgrade if the counts read as unfair.
FILLER_WORDS = {"um", "uh", "er", "ah", "like", "you know", "i mean", "sort of", "kind of"}


def compute_delivery_metrics(
    learner_turns: Sequence[Transcript],
    agent_speaking_s: float,
) -> DeliveryMetrics:
    """How the learner spoke, from word-timestamped STT.

    Timestamps restart at 0 for each turn (each turn is transcribed separately), so speaking time
    and pauses are computed per turn; gaps BETWEEN turns are agent time and are ignored.
    Benchmarks (recalibrate per scenario): talk ~0.43 · fillers <8/min · WPM 140-160.
    """
    speaking_time_s = 0.0
    total_words = 0
    filler_count = 0
    longest_pause_s = 0.0
    for each_turn in learner_turns:
        if each_turn.words:
            speaking_time_s += each_turn.words[-1].end - each_turn.words[0].start
            total_words += len(each_turn.words)
            # Whisper attaches punctuation/case ("Um,") → normalize; pairs catch "you know".
            tokens = [w.text.lower().strip(string.punctuation) for w in each_turn.words]
            filler_count += sum(t in FILLER_WORDS for t in tokens)
            filler_count += sum(
                f"{a} {b}" in FILLER_WORDS for a, b in zip(tokens, tokens[1:], strict=False)
            )
            # Compute longest pause within this turn
            for i in range(1, len(each_turn.words)):
                pause = each_turn.words[i].start - each_turn.words[i - 1].end
                if pause > longest_pause_s:
                    longest_pause_s = pause

    return DeliveryMetrics(
        wpm=(total_words / (speaking_time_s / 60)) if speaking_time_s > 0 else 0.0,
        filler_count=filler_count,
        fillers_per_min=(filler_count / (speaking_time_s / 60)) if speaking_time_s > 0 else 0.0,
        # Talk:listen = your speech vs the agent's. NOT vs the wall clock: typing, thinking and
        # silence would deflate it (a mixed text+voice session read 11% that way).
        talk_ratio=speaking_time_s / (speaking_time_s + agent_speaking_s)
        if speaking_time_s + agent_speaking_s > 0
        else 0.0,
        spoken_turns=sum(1 for t in learner_turns if t.words),
        longest_pause_s=longest_pause_s,
        speaking_time_s=speaking_time_s,
    )
