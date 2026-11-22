"""DeliveryTracker — collects delivery-metric inputs from the Pipecat frame stream.

The maths is unchanged (`metrics.compute_delivery_metrics`, your Phase-2 TODO #3); only its
*inputs* now come from frames instead of the hand-built pipeline's own bookkeeping
(`pipeline.learner_turns`, `_agent_speaking_s`, `TurnDetector.longest_pause_ms`).

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
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..core.voice_port import Transcript
from .metrics import compute_delivery_metrics
from .schemas import VoiceEvent


class DeliveryTracker(FrameProcessor):
    """Pure observer: every frame passes through unchanged, in the direction it came.

      learner turns  ← TranscriptionFrame.result (word-timestamped; typed turns never produce one)
      agent speech   ← BotStarted → BotStoppedSpeaking (an interruption ends it early: barge-in
                       is counted correctly for free)
      longest pause  ← VAD quiet (stopped → next started) while the learner holds the floor;
                       Whisper swallows silence into word timings, so pauses come from the VAD
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
            await self.push_frame(event_frame(VoiceEvent(type="metrics", metrics=metrics)))
        await self.push_frame(frame, direction)
