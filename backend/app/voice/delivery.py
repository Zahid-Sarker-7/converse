"""DeliveryTracker — collects delivery-metric inputs from the Pipecat frame stream.

The maths is unchanged (`metrics.compute_delivery_metrics`, your Phase-2 TODO #3); only its
*inputs* now come from frames instead of the hand-built pipeline's own bookkeeping
(`pipeline.learner_turns`, `_agent_speaking_s`, `TurnDetector.longest_pause_ms`).

Where it sits (bot.py): transport.input → STT → VoiceEvents → **DeliveryTracker** → user aggregator.
So it sees transcripts going DOWN, and the VAD + bot-speaking frames that the aggregator and the
output transport broadcast back UP.
"""

from __future__ import annotations

from pipecat.frames.frames import Frame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class DeliveryTracker(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """### TODO #2: Delivery metrics from the frame stream  [track: C-project]  [status: todo]

        - What:  observe frames (pass EVERY frame on unchanged) and collect the three inputs of
                 `compute_delivery_metrics`; on `EndFrame`, send the metrics event BEFORE passing
                 the EndFrame on, so it reaches the browser before the socket closes.
                   learner turns  ← `TranscriptionFrame.result` (a `Transcript` with word timings;
                                    bot.py's STT puts it there; typed turns never produce one)
                   agent speech s ← time between `BotStartedSpeakingFrame` and
                                    `BotStoppedSpeakingFrame` (an interruption ends it early, so
                                    barge-in is counted correctly for free)
                   longest pause  ← VAD: quiet from `VADUserStoppedSpeakingFrame` until the next
                                    `VADUserStartedSpeakingFrame`, but only while the learner holds
                                    the floor (the agent speaking resets it — that's listening)
        - Why:   the hand-built pipeline had to keep this bookkeeping itself, threaded through its
                 own code. Here it's a pure observer you can drop anywhere, which teaches the frame
                 stream: which frames exist, which direction they travel, and what that position
                 in the pipeline can see. (Also: Whisper swallows silence into word timings — a
                 1.5s pause showed as 0.00s — which is why pauses come from the VAD, not words.)
        - How:   VAD frames carry `timestamp` (when VAD *decided*) + `stop_secs`/`start_secs` (how
                 long it waited to decide), so the real quiet started at `timestamp - stop_secs` and
                 ended at `timestamp - start_secs`. Bot frames have no timestamp: use `time.time()`
                 (call it as `time.time()` so the test can fake the clock). Then
                 `metrics = compute_delivery_metrics(turns, agent_s)`, set
                 `metrics.longest_pause_s = round(pause, 2)`, and push
                 `bot.event_frame(VoiceEvent(type="metrics", metrics=metrics))` (import `bot` inside
                 the method). Keep `await super().process_frame(...)` first.
        - Working with: `metrics.compute_delivery_metrics` · `core/voice_port.Transcript` ·
                 `schemas.VoiceEvent` · frames in `pipecat.frames.frames`.
        - Docs & SDKs:  https://docs.pipecat.ai/pipecat/learn/pipeline
                 https://docs.pipecat.ai/pipecat/learn/speech-input  (VAD + turn frames)
        - Acceptance: delete the `@TODO_TRACKER` line in `tests/test_voice_pipecat.py` → green
                 (two turns + a 1.0s pause + 2s of agent speech → the expected metrics, sent before
                 the EndFrame). Then un-gate the pipecat half of `tests/test_voice_ws.py`.
        """
        await super().process_frame(frame, direction)
        # TODO #2: collect + emit as described above. Until then no metrics are sent.
        await self.push_frame(frame, direction)
