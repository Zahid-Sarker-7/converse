"""Voice on Pipecat — the Phase-2 pipeline rebuilt on a framework (Decision 2026-11-01).

Same wire protocol, same brain, same metrics as the hand-built engine (`pipeline.py` + `turn.py`,
tag `v0.2-hand-built-voice`). What changed is who owns the plumbing:

  WS ─► transport.input ─► STT ─► VoiceEvents ─► DeliveryTracker ─► user aggregator
        (ConverseSerializer)                                       (Silero VAD + Smart-Turn v3)
     ─► ConverseBrain (LangGraph, unchanged) ─► TTS ─► transport.output ─► assistant aggregator
                                                       (ConverseSerializer) ─► WS

Barge-in is now the framework's job: user speech → `InterruptionFrame` → every processor drops
its queue and cancels the frame it is working on (the brain's graph call, or TTS), the output
transport discards unsent audio, and the serializer tells the browser to flush playback. That one
mechanism replaces `_respond_task`, `_barge_in`, `_drop_queued_audio` and the carry-over hack.
"""

from __future__ import annotations

import json
import string

from fastapi import WebSocket
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndWorkerFrame,
    Frame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMMessagesAppendFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    UserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner

from ..agents.schemas import ScenarioConfig
from ..core.model_port import ChatMessage, ModelPort
from ..core.voice.groq_whisper_stt import WHISPER_PROMPT
from ..core.voice_port import STT_SAMPLE_RATE, TTS_SAMPLE_RATE, Transcript, Word
from ..logging import get_logger
from .brain import ConverseBrain
from .delivery import DeliveryTracker
from .pipeline import PHANTOM_MAX_SPEECH_MS, WHISPER_PHANTOMS
from .schemas import VoiceEvent

log = get_logger("voice")


def event_frame(event: VoiceEvent) -> OutputTransportMessageUrgentFrame:
    """A `VoiceEvent` on its way to the browser. Urgent = a system frame: it isn't queued behind
    audio, and an interruption can't drop it."""
    return OutputTransportMessageUrgentFrame(message=event.model_dump(mode="json"))


# ── transport boundary ─────────────────────────────────────────────────────────────────────────


class ConverseSerializer(FrameSerializer):
    """The browser's wire protocol (see `api/voice.py`) <-> Pipecat frames, so the frontend is
    untouched. The handshake (first text frame) is read by the route before Pipecat starts."""

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if self.should_ignore_frame(frame):
            return None
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio  # PCM16 @ 24 kHz, the transport already resampled
        if isinstance(frame, InterruptionFrame):
            return VoiceEvent(type="interrupt").model_dump_json()
        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)
        return None  # everything else stays inside the pipeline

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=STT_SAMPLE_RATE, num_channels=1)
        try:
            msg = json.loads(data)
        except ValueError:
            return None  # malformed control frames are ignored, not fatal
        if not isinstance(msg, dict):
            return None
        if msg.get("type") == "text" and (text := str(msg.get("text", "")).strip()):
            # A typed turn: straight into the conversation, no STT, and the brain answers it.
            # ponytail: typing while the agent talks doesn't interrupt it (the hand-built one did);
            # push an InterruptionFrame first if that matters.
            return LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": text}], run_llm=True
            )
        if msg.get("type") == "end":
            return InputTransportMessageFrame(message=msg)  # VoiceEvents ends the session
        return None


# ── STT / TTS services ─────────────────────────────────────────────────────────────────────────


def _phantom(transcript: Transcript) -> bool:
    """Whisper writes "Thank you." on non-speech (Groq's no_speech_prob is always 0, so there's no
    confidence to filter on): drop those when the speech was short."""
    spoken_ms = (
        (transcript.words[-1].end - transcript.words[0].start) * 1000 if transcript.words else 0
    )
    phrase = transcript.text.lower().strip(string.punctuation + " ")
    return phrase in WHISPER_PHANTOMS and spoken_ms < PHANTOM_MAX_SPEECH_MS


def _transcription(transcript: Transcript, user_id: str) -> TranscriptionFrame | None:
    """`result` carries our `Transcript` (with word timestamps) to the DeliveryTracker."""
    if not transcript.text or _phantom(transcript):
        log.info("phantom_ignored", words=len(transcript.words))
        return None
    return TranscriptionFrame(transcript.text, user_id, time_now_iso8601(), result=transcript)


class GroqWordsSTT(SegmentedSTTService):
    """Groq Whisper with word timestamps + our filler/vocab prompt. Pipecat's `GroqSTTService`
    can't ask for `timestamp_granularities`, and delivery metrics are built from word timings."""

    def __init__(self, *, api_key: str, model: str, **kwargs) -> None:
        super().__init__(settings=STTSettings(model=model, language=None), **kwargs)
        from groq import AsyncGroq

        self._client = AsyncGroq(api_key=api_key)
        self._model = model

    async def run_stt(self, audio: bytes):  # audio = one VAD segment, as a WAV file
        resp = await self._client.audio.transcriptions.create(
            file=("utterance.wav", audio, "audio/wav"),
            model=self._model,
            response_format="verbose_json",
            timestamp_granularities=["word"],
            prompt=WHISPER_PROMPT,
            temperature=0,
        )
        words = [
            Word(text=w["word"], start=w["start"], end=w["end"])
            for w in (getattr(resp, "words", None) or [])
        ]
        if frame := _transcription(Transcript(text=resp.text.strip(), words=words), self._user_id):
            yield frame


class EchoSTT(SegmentedSTTService):
    """$0 / keyless: a fixed transcript with word timings spread over the segment."""

    def __init__(self, **kwargs) -> None:
        super().__init__(settings=STTSettings(model=None, language=None), **kwargs)

    async def run_stt(self, audio: bytes):
        tokens = "echo one two three".split()
        step = max(len(audio) / 2 / STT_SAMPLE_RATE, 0.5) / len(tokens)
        words = [Word(text=t, start=i * step, end=(i + 1) * step) for i, t in enumerate(tokens)]
        if frame := _transcription(Transcript(text=" ".join(tokens), words=words), self._user_id):
            yield frame


class EchoTTS(TTSService):
    """$0 / no model: ~0.6s of silence per sentence, in 3 chunks (a real stream to interrupt)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            settings=TTSSettings(model=None, voice=None, language=None),
            **kwargs,
        )

    async def run_tts(self, text: str, context_id: str):
        for _ in range(3):
            yield TTSAudioRawFrame(
                b"\x00\x00" * (TTS_SAMPLE_RATE // 5), TTS_SAMPLE_RATE, 1, context_id=context_id
            )


def make_stt(name: str, settings) -> SegmentedSTTService:
    if name == "groq":
        return GroqWordsSTT(api_key=settings.groq_api_key, model=settings.whisper_model)
    return EchoSTT()


def make_tts(name: str, settings) -> TTSService:
    if name == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        from ..core.voice.kokoro_tts import MODEL_PATH, VOICES_PATH

        # ponytail: loads the ONNX model per session (~1s); share one Kokoro if sessions pile up.
        return KokoroTTSService(
            model_path=str(MODEL_PATH),
            voices_path=str(VOICES_PATH),
            settings=KokoroTTSService.Settings(voice=settings.kokoro_voice),
        )
    return EchoTTS()


# ── session state → browser ────────────────────────────────────────────────────────────────────


class VoiceEvents(FrameProcessor):
    """Turns pipeline facts into the browser's `state` / user `transcript` events.

    Sits right after STT so it sees transcripts going down AND the turn/bot-speaking frames the
    aggregator and output transport broadcast back up. (The brain sends the assistant transcript.)
    """

    _STATES = {
        UserStartedSpeakingFrame: "listening",  # "thinking" comes from the brain (typed turns too)
        BotStartedSpeakingFrame: "speaking",
        BotStoppedSpeakingFrame: "listening",
    }

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, InputTransportMessageFrame) and frame.message.get("type") == "end":
            # End = stop talking now, then end gracefully. A bare EndWorkerFrame would flush the
            # agent's unplayed speech first (paced in real time), and the browser stops waiting
            # for metrics after 5s. Interrupt drops it; the EndFrame then flows down in order, so
            # the DeliveryTracker's metrics still reach the browser before the socket closes.
            if direction == FrameDirection.DOWNSTREAM:  # the input transport broadcasts both ways
                await self.broadcast_interruption()
                await self.push_frame(EndWorkerFrame(reason="client ended the session"))
            return
        if (state := self._STATES.get(type(frame))) is not None:
            await self.push_frame(event_frame(VoiceEvent(type="state", state=state)))
        elif isinstance(frame, TranscriptionFrame):
            await self.push_frame(
                event_frame(VoiceEvent(type="transcript", role="user", text=frame.text))
            )
        await self.push_frame(frame, direction)


# ── assembly ───────────────────────────────────────────────────────────────────────────────────


def user_turn_params() -> LLMUserAggregatorParams:
    """The turn-taking calibration that was hand-tuned in `turn.py`, as Pipecat knobs.

    VAD start 0.2s = the 200ms barge-in guard (a cough or "mm-hm" doesn't interrupt).
    VAD stop 0.5s  = Smart-Turn is asked after 0.5s of silence (0.2s cut people off, live).
    Smart-Turn stop 3.0s = give up waiting even if it says "incomplete".
    ponytail: no 60s turn cap (steady noise grows a turn forever); add a watchdog if seen live.
    """
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(confidence=0.5, start_secs=0.2, stop_secs=0.5)
        ),
        user_turn_strategies=UserTurnStrategies(
            stop=[
                TurnAnalyzerUserTurnStopStrategy(
                    turn_analyzer=LocalSmartTurnAnalyzerV3(params=SmartTurnParams(stop_secs=3.0))
                )
            ]
        ),
    )


async def run_session(
    ws: WebSocket,
    *,
    settings,
    scenario: ScenarioConfig,
    history: list[ChatMessage],
    chat_port: ModelPort,
    stt_name: str,
    tts_name: str,
) -> None:
    """One voice session on Pipecat. The route has already accepted `ws` and read the handshake."""
    transport = FastAPIWebsocketTransport(
        websocket=ws,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            serializer=ConverseSerializer(),
            audio_out_end_silence_secs=0,  # no trailing silence after the metrics
        ),
    )
    context = LLMContext(messages=[{"role": m.role.value, "content": m.content} for m in history])
    aggregators = LLMContextAggregatorPair(context, user_params=user_turn_params())
    tracker = DeliveryTracker()

    pipeline = Pipeline(
        [
            transport.input(),
            make_stt(stt_name, settings),
            VoiceEvents(),
            tracker,
            aggregators.user(),
            ConverseBrain(scenario=scenario, chat_port=chat_port),
            make_tts(tts_name, settings),
            transport.output(),
            aggregators.assistant(),
        ]
    )
    worker = PipelineWorker(pipeline, params=PipelineParams(), enable_rtvi=False)

    @transport.event_handler("on_client_disconnected")
    async def _disconnected(_transport, _ws) -> None:
        await worker.cancel()  # tab closed / network drop: no metrics, just stop

    await ws.send_text(VoiceEvent(type="state", state="listening").model_dump_json())
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
