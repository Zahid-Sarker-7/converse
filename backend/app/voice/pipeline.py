"""The voice pipeline — glues turn detection -> STT -> the EXISTING LangGraph loop -> TTS,
and hosts the barge-in seam (TODO #2).

Design (why it looks like this):
  - The WS route feeds mic frames in via `on_audio_frame()` and consumes everything the
    pipeline wants to send (state changes, transcripts, metrics, and raw TTS audio) from the
    `out` queue. One queue keeps ordering simple; binary audio and JSON events are tagged.
  - `_respond()` runs as its OWN task so that while the agent is SPEAKING we're still reading
    the mic — that concurrency is precisely what makes barge-in possible.
  - The LangGraph loop is called exactly as the HTTP route calls it: build_graph(...).ainvoke
    with mode="chat". Voice is just a new transport in front of the Phase-1 brain.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator

from ..agents.graph import build_graph
from ..agents.schemas import ScenarioConfig
from ..core.model_port import ChatMessage, ModelPort, Role
from ..core.voice_port import STTPort, Transcript, TTSPort
from ..logging import get_logger
from .metrics import compute_delivery_metrics
from .schemas import VoiceEvent
from .turn import TurnDetector

log = get_logger("voice")

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Chunk a reply into sentences so TTS can start on sentence 1 while the rest synthesizes
    (and so barge-in cancels at a natural boundary). Keeps it dependency-free."""
    parts = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class VoicePipeline:
    def __init__(
        self,
        *,
        scenario: ScenarioConfig,
        chat_port: ModelPort,
        stt: STTPort,
        tts: TTSPort,
        tts_voice: str | None = None,
    ) -> None:
        self.scenario = scenario
        self.chat_port = chat_port
        self.stt = stt
        self.tts = tts
        self.tts_voice = tts_voice
        self.graph = build_graph(chat_port=chat_port, judge_port=chat_port)

        self.turn = TurnDetector()
        self.messages: list[ChatMessage] = []
        self.learner_turns: list[Transcript] = []  # for delivery metrics (TODO #3)

        self.out: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self._respond_task: asyncio.Task | None = None
        self._tts_cancel = asyncio.Event()
        self._loop_start = asyncio.get_event_loop().time()

    # ── inbound ────────────────────────────────────────────────────────────────
    async def on_audio_frame(self, frame: bytes) -> None:
        """Called per mic frame by the WS route."""
        event = self.turn.feed(frame)
        if event is None:
            return
        if event.kind == "speech_started":
            await self._on_speech_started()
        elif event.kind == "speech_ended":
            await self._on_speech_ended()

    async def _on_speech_started(self) -> None:
        # The user started talking. If the agent is mid-reply, that's a barge-in.
        if self._respond_task is not None and not self._respond_task.done():
            await self._barge_in()
        await self._emit(VoiceEvent(type="state", state="listening"))

    async def _on_speech_ended(self) -> None:
        utterance = self.turn.utterance_audio
        self.turn.reset()
        # Kick off the response as its own task so the mic keeps being read.
        self._tts_cancel.clear()
        self._respond_task = asyncio.create_task(self._respond(utterance))

    # ── barge-in — HANDS-ON TODO #2 ─────────────────────────────────────────────
    async def _barge_in(self) -> None:
        """### TODO: Barge-in handling  [track: C-project]  [status: todo]

        - What:  when the learner starts speaking while the agent is talking, STOP the agent's
                 speech **fast** and cleanly. Three requirements make it feel natural:
                   (a) stop TTS in <60 ms  (cancel the `_respond` task + signal `_tts_cancel`);
                   (b) DRAIN already-queued audio from `self.out` so no stale speech plays after
                       the cut (pop the ("audio", ...) items that are still waiting);
                   (c) a ~200 ms MIN-DURATION guard so a cough/"mm-hm" backchannel doesn't
                       trigger a false barge-in (only barge in if the new speech sustains).
        - Why:  barge-in is the difference between a walkie-talkie and a conversation. Doing it
                well (fast + drained + guarded) is the craft of real-time voice. Depends on your
                turn detector (TODO #1) for the "user is really speaking" signal.
        - How:  1. cancel `self._respond_task` and `await` it (swallow CancelledError).
                2. set `self._tts_cancel` so any in-flight synth loop breaks on its next check.
                3. drain audio frames from `self.out` (keep non-audio events if you like).
                4. implement the 200ms guard — e.g. only call this once the turn detector has
                   reported sustained speech, not on the very first frame.
                5. emit VoiceEvent(type="state", state="listening").
        - Working with: `_respond()` (checks `_tts_cancel` between chunks) · `TurnDetector`
                 (TODO #1) · `self.out` queue · target from the latency budget: TTS stop <60ms.
        - Docs & SDKs:  barge-in/turn-taking  https://futureagi.com/blog/voice-ai-barge-in-turn-taking-2026/
                 latency budget (stop <60ms, drain)  ventures/roadmap/project-plan.md §voice budget
        - Acceptance: `make test` → unskip test_barge_in_stops_and_drains and make it green:
                 starting speech mid-reply cancels the response task and leaves no audio queued.
                 Manually: talk over the agent — it should go quiet almost instantly.
        """
        # Naive placeholder so the pipeline is runnable before you implement the real thing:
        # it cancels, but does NOT drain the queue or apply the min-duration guard.
        if self._respond_task and not self._respond_task.done():
            self._tts_cancel.set()
            self._respond_task.cancel()
            try:
                await self._respond_task
            except asyncio.CancelledError:
                pass

    # ── the turn: STT -> graph -> TTS ───────────────────────────────────────────
    async def _respond(self, utterance_audio: bytes) -> None:
        try:
            await self._emit(VoiceEvent(type="state", state="thinking"))

            # 1. STT (batch, whole utterance).
            transcript = await self.stt.transcribe(utterance_audio)
            if not transcript.text:
                await self._emit(VoiceEvent(type="state", state="listening"))
                return
            self.learner_turns.append(transcript)
            self.messages.append(ChatMessage(role=Role.user, content=transcript.text))
            await self._emit(VoiceEvent(type="transcript", role="user", text=transcript.text))

            # 2. The existing Phase-1 brain — untouched.
            result = await self.graph.ainvoke(
                {"scenario": self.scenario, "messages": self.messages, "mode": "chat"}
            )
            self.messages = result["messages"]
            reply = next(
                (m.content for m in reversed(self.messages) if m.role == Role.assistant), ""
            )
            await self._emit(VoiceEvent(type="transcript", role="assistant", text=reply))

            # 3. TTS, sentence by sentence (fast first-audio + natural barge-in boundaries).
            await self._emit(VoiceEvent(type="state", state="speaking"))
            for sentence in _split_sentences(reply):
                if self._tts_cancel.is_set():
                    break
                async for chunk in self.tts.synthesize(sentence, voice=self.tts_voice):
                    if self._tts_cancel.is_set():
                        break
                    if chunk.pcm:
                        await self.out.put(("audio", chunk.pcm))
            await self._emit(VoiceEvent(type="state", state="listening"))
        except asyncio.CancelledError:
            raise  # barge-in cancelled us; let it propagate
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the socket
            log.error("voice_respond_error", error=f"{type(exc).__name__}: {exc}", exc_info=True)
            await self._emit(VoiceEvent(type="error", message=f"{type(exc).__name__}: {exc}"))

    # ── session end: delivery metrics (TODO #3 fills these in) ──────────────────
    async def finish(self) -> None:
        total_s = asyncio.get_event_loop().time() - self._loop_start
        metrics = compute_delivery_metrics(self.learner_turns, total_s)
        await self._emit(VoiceEvent(type="metrics", metrics=metrics))

    # ── outbound helper ─────────────────────────────────────────────────────────
    async def _emit(self, event: VoiceEvent) -> None:
        await self.out.put(("event", event))

    async def drain(self) -> AsyncIterator[tuple[str, object]]:
        """Yield queued ('event'|'audio', payload) items until the queue is empty."""
        while not self.out.empty():
            yield await self.out.get()
