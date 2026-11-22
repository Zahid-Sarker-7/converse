"""DeliveryTracker — collects delivery-metric inputs from the Pipecat frame stream.

The maths is `metrics.compute_delivery_metrics`; its inputs come from frames (the hand-built
engine, tag `v0.2-hand-built-voice`, kept this bookkeeping inside its own pipeline).

Where it sits (bot.py): transport.input → STT → VoiceEvents → **DeliveryTracker** → user aggregator.
So it sees transcripts going DOWN, and the VAD + bot-speaking frames that the aggregator and the
output transport broadcast back UP.
"""

from __future__ import annotations

import time

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..logging import get_logger
from .metrics import compute_delivery_metrics
from .schemas import Transcript, VoiceEvent

log = get_logger("voice")


class DeliveryTracker(FrameProcessor):
    """Pure observer: every frame passes through unchanged, in the direction it came.

      learner turns  ← TranscriptionFrame.result (word-timestamped; typed turns never produce one)
      agent speech   ← BotStarted → BotStoppedSpeaking (an interruption ends it early: barge-in
                       is counted correctly for free)
      longest pause  ← VAD quiet (stopped → next started) WITHIN one user turn: the end of a turn
                       clears it, so endpointing waits and agent latency never count (a live run
                       read 5.28s that way); Whisper swallows silence into word timings, so pauses
                       come from the VAD
    On EndFrame the metrics event goes out first, so it reaches the browser before the close.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.learner_turns: list[Transcript] = []
        self.agent_speaking_s = 0.0
        self.longest_pause_s = 0.0
        self._bot_since: float | None = None
        self._quiet_since: float | None = None  # None = not holding the floor (or never spoke)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and isinstance(frame.result, Transcript):
            self.learner_turns.append(frame.result)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            # VAD reports after waiting stop_secs of silence: the quiet began that much earlier.
            self._quiet_since = frame.timestamp - frame.stop_secs
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            if self._quiet_since is not None:
                pause = frame.timestamp - frame.start_secs - self._quiet_since
                self.longest_pause_s = max(self.longest_pause_s, pause)
            self._quiet_since = None
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._quiet_since = None  # the turn is over: what follows is the agent's floor
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._bot_since = time.time()
            self._quiet_since = None  # the agent took the floor: what follows is listening
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._bot_since is not None:
            self.agent_speaking_s += time.time() - self._bot_since
            self._bot_since = None
        elif isinstance(frame, EndFrame):
            from .bot import event_frame  # bot imports this module

            metrics = compute_delivery_metrics(self.learner_turns, self.agent_speaking_s)
            metrics.longest_pause_s = round(self.longest_pause_s, 2)
            log.info("session_metrics", turns=len(self.learner_turns), **metrics.model_dump())
            await self.push_frame(event_frame(VoiceEvent(type="metrics", metrics=metrics)))
        await self.push_frame(frame, direction)
