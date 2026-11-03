"""The voice pipeline — glues turn detection -> STT -> the EXISTING LangGraph loop -> TTS,
plus barge-in (sustained user speech cancels the agent mid-reply).

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
import contextlib
import re

from ..agents.graph import build_graph
from ..agents.schemas import ScenarioConfig
from ..core.model_port import ChatMessage, ModelPort, Role
from ..core.voice_port import TTS_SAMPLE_RATE, STTPort, Transcript, TTSPort
from ..logging import get_logger
from .metrics import compute_delivery_metrics
from .schemas import VoiceEvent
from .turn import TurnDetector

log = get_logger("voice")

# Less detected speech than this in a whole turn = a click or noise, not a turn (dropped pre-STT).
MIN_TURN_SPEECH_MS = 250
# Whisper invents these on non-speech audio (verified: silence and keyboard noise → "Thank you.").
# Groq returns no_speech_prob=0, so the usual confidence filter is unavailable; drop them when the
# turn was short. ponytail: a genuinely spoken brief "thank you" is dropped too (logged).
WHISPER_PHANTOMS = {
    "thank you",
    "thanks",
    "thanks for watching",
    "thank you for watching",
    "you",
    "bye",
    "",
}
PHANTOM_MAX_SPEECH_MS = 1200
# Second opinion on "is the turn over?", from the words: a transcript ending mid-thought ("we had
# to...") means the acoustic detector ended the turn early (often the 3s silence fallback during a
# thinking pause). Hold before answering; resumed speech barges into the hold and merges.
UNFINISHED_HOLD_S = 3.0
_DANGLING = {
    "and",
    "but",
    "so",
    "or",
    "because",
    "to",
    "the",
    "a",
    "an",
    "of",
    "with",
    "for",
    "that",
    "which",
    "um",
    "uh",
    "like",
    "is",
    "was",
    "we",
    "i",
    "my",
    "our",
}


def _looks_unfinished(text: str) -> bool:
    t = text.strip()
    if t.endswith(("...", "…", ",", "-", "—")):
        return True
    words = t.rstrip(".!?").split()
    return bool(words) and not t.endswith(("?", "!")) and words[-1].lower().strip(",") in _DANGLING


BARGE_IN_MIN_SPEECH_MS = 200  # shorter speech while the agent talks = backchannel, not a barge-in

VOICE_STYLE = (
    "You are speaking aloud on a live voice call: reply in one to three short, natural spoken "
    "sentences. Never use markdown, bullet points, headings, numbered lists, or emoji."
)

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
        history: list[ChatMessage] | None = None,  # prior text turns: one conversation, any mode
    ) -> None:
        # Spoken replies must be short and plain: TTS reads markdown/lists aloud verbatim.
        self.scenario = scenario.model_copy(
            update={"persona": f"{scenario.persona}. {VOICE_STYLE}"}
        )
        self.chat_port = chat_port
        self.stt = stt
        self.tts = tts
        self.tts_voice = tts_voice
        self.graph = build_graph(chat_port=chat_port, judge_port=chat_port)

        self.turn = TurnDetector()
        self.messages: list[ChatMessage] = list(history or [])
        self.learner_turns: list[Transcript] = []  # word-timestamped turns → delivery metrics

        self.out: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        self._respond_task: asyncio.Task | None = None
        self._untranscribed: tuple[bytes, float] | None = None
        self._tts_cancel = asyncio.Event()
        self._agent_speaking_s = 0.0  # TTS audio actually sent (barge-in subtracts the unsent part)

    # ── inbound ────────────────────────────────────────────────────────────────
    def _responding(self) -> bool:
        return self._respond_task is not None and not self._respond_task.done()

    async def on_audio_frame(self, frame: bytes) -> None:
        """Called per mic frame by the WS route."""
        event = self.turn.feed(frame)
        if event:
            log.info(
                "turn_event", kind=event.kind, reason=event.reason, speech_ms=self.turn.speech_ms
            )

        # Barge-in guard: only SUSTAINED speech interrupts the agent, not a cough or "mm-hm".
        if self._responding() and self.turn.speech_ms >= BARGE_IN_MIN_SPEECH_MS:
            await self._barge_in()

        if event is None:
            return
        if event.kind == "speech_started" and not self._responding():
            await self._emit(VoiceEvent(type="state", state="listening"))
        elif event.kind == "speech_ended":
            await self._on_speech_ended()

    async def _on_speech_ended(self) -> None:
        utterance, speech_ms = self.turn.utterance_audio, self.turn.speech_ms
        self.turn.reset()
        if speech_ms < MIN_TURN_SPEECH_MS:
            log.info("noise_ignored", speech_ms=speech_ms)
            return
        if self._responding():
            # A turn ended while the agent is still busy, so it never reached the guard:
            # a backchannel/noise. Drop it; the agent keeps talking.
            log.info("blip_ignored", bytes=len(utterance))
            return
        self._tts_cancel.clear()
        # Until STT returns, this audio exists nowhere else: if the learner resumes and barges in
        # now, _barge_in carries it into the next turn instead of losing it (it was losing whole
        # opening segments: "We delivered it faster… than the committed time" → "than the …").
        self._untranscribed = (utterance, speech_ms)
        # Its own task, so the mic keeps being read while the agent thinks/speaks.
        self._respond_task = asyncio.create_task(self._respond(utterance, speech_ms=speech_ms))

    # ── barge-in ────────────────────────────────────────────────────────────────
    async def _barge_in(self) -> None:
        """Stop the agent: cancel generation, drop unsent audio, tell the client to flush."""
        t0 = asyncio.get_running_loop().time()
        self._tts_cancel.set()  # breaks the TTS loops at their next check
        self._respond_task.cancel()  # raises CancelledError at _respond's current await
        with contextlib.suppress(asyncio.CancelledError):
            await self._respond_task
        if self._untranscribed:  # cancelled mid-STT: same thought, so it leads the new turn
            self.turn.prepend(*self._untranscribed)
            self._untranscribed = None
        dropped = self._drop_queued_audio()
        # Audio already sent is sitting in the browser's playback schedule: flush it there too.
        await self._emit(VoiceEvent(type="interrupt"))
        await self._emit(VoiceEvent(type="state", state="listening"))
        stop_ms = (asyncio.get_running_loop().time() - t0) * 1000
        log.info("barge_in", dropped_audio=dropped, stop_ms=round(stop_ms, 1))

    def _drop_queued_audio(self) -> int:
        """Remove queued audio but keep events, in order. No await inside → atomic vs the pump."""
        kept, dropped = [], 0
        while not self.out.empty():
            item = self.out.get_nowait()
            if item[0] == "audio":
                dropped += 1
                self._agent_speaking_s -= len(item[1]) / 2 / TTS_SAMPLE_RATE  # never played
            else:
                kept.append(item)
        for item in kept:
            self.out.put_nowait(item)
        return dropped

    # ── the turn: STT -> graph -> TTS ───────────────────────────────────────────
    async def on_text(self, text: str) -> None:
        """A typed message mid-session: same conversation, no STT. Interrupts the agent if needed.
        The client already shows what it typed, so no user transcript is echoed back."""
        if self._responding():
            await self._barge_in()
        self._tts_cancel.clear()
        self._respond_task = asyncio.create_task(self._respond(typed=text))

    def _add_user_line(self, text: str) -> None:
        # The agent hasn't answered the previous user line (its reply was interrupted): the learner
        # is still on the same answer, so join it rather than leaving fragments for the LLM.
        if self.messages and self.messages[-1].role == Role.user:
            prev = self.messages.pop()
            text = f"{prev.content} {text}"
        self.messages.append(ChatMessage(role=Role.user, content=text))

    async def _respond(
        self, utterance_audio: bytes = b"", typed: str | None = None, speech_ms: float = 0.0
    ) -> None:
        try:
            await self._emit(VoiceEvent(type="state", state="thinking"))

            if typed is not None:
                self._add_user_line(typed)
            else:
                # 1. STT (batch, whole utterance).
                transcript = await self.stt.transcribe(utterance_audio)
                self._untranscribed = None  # no await since → atomic with the line above
                log.info("stt_done", words=len(transcript.words))
                log.debug("stt_text", text=transcript.text)  # learner speech = PII: debug only
                phantom = transcript.text.lower().strip(" .!?,") in WHISPER_PHANTOMS
                if not transcript.text or (phantom and speech_ms < PHANTOM_MAX_SPEECH_MS):
                    log.info("phantom_ignored", speech_ms=speech_ms)
                    await self._emit(VoiceEvent(type="state", state="listening"))
                    return
                self.learner_turns.append(transcript)
                self._add_user_line(transcript.text)
                await self._emit(VoiceEvent(type="transcript", role="user", text=transcript.text))
                if _looks_unfinished(transcript.text):
                    # Still "responding", so speech now is a barge-in that cancels this hold, and
                    # the next turn merges into this unanswered line.
                    log.info("held_unfinished", hold_s=UNFINISHED_HOLD_S)
                    await self._emit(VoiceEvent(type="state", state="listening"))
                    await asyncio.sleep(UNFINISHED_HOLD_S)
                    await self._emit(VoiceEvent(type="state", state="thinking"))

            # 2. The existing Phase-1 brain — untouched.
            result = await self.graph.ainvoke(
                {"scenario": self.scenario, "messages": self.messages, "mode": "chat"}
            )
            self.messages = result["messages"]
            reply = next(
                (m.content for m in reversed(self.messages) if m.role == Role.assistant), ""
            )
            log.info("llm_reply", chars=len(reply))
            self.turn.agent_took_turn()
            await self._emit(VoiceEvent(type="transcript", role="assistant", text=reply))

            # 3. TTS, sentence by sentence (fast first-audio + natural barge-in boundaries).
            await self._emit(VoiceEvent(type="state", state="speaking"))
            sentences = _split_sentences(reply)
            audio_chunks = 0
            for sentence in sentences:
                if self._tts_cancel.is_set():
                    break
                async for chunk in self.tts.synthesize(sentence, voice=self.tts_voice):
                    if self._tts_cancel.is_set():
                        break
                    if chunk.pcm:
                        audio_chunks += 1
                        self._agent_speaking_s += len(chunk.pcm) / 2 / TTS_SAMPLE_RATE
                        await self.out.put(("audio", chunk.pcm))
            log.info(
                "tts_done",
                sentences=len(sentences),
                audio_chunks=audio_chunks,
                cancelled=self._tts_cancel.is_set(),
            )
            await self._emit(VoiceEvent(type="state", state="listening"))
        except asyncio.CancelledError:
            raise  # barge-in cancelled us; let it propagate
        except Exception as exc:  # noqa: BLE001 — surface, don't crash the socket
            log.error("voice_respond_error", error=f"{type(exc).__name__}: {exc}", exc_info=True)
            await self._emit(VoiceEvent(type="error", message=f"{type(exc).__name__}: {exc}"))

    # ── session end: delivery metrics ──────────────────────────────────────────────
    async def finish(self) -> None:
        metrics = compute_delivery_metrics(self.learner_turns, self._agent_speaking_s)
        # Pauses come from the VAD, not word timestamps (Whisper absorbs silence into words).
        metrics.longest_pause_s = round(self.turn.longest_pause_ms / 1000, 2)
        log.info("session_metrics", turns=len(self.learner_turns), **metrics.model_dump())
        await self._emit(VoiceEvent(type="metrics", metrics=metrics))

    # ── outbound helper ─────────────────────────────────────────────────────────
    async def _emit(self, event: VoiceEvent) -> None:
        await self.out.put(("event", event))
