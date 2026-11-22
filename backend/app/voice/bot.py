"""Voice on Pipecat — the Phase-2 pipeline on a framework (Decision 2026-11-01, TDD D9).

Same wire protocol, brain and metrics as the hand-built engine it replaced (tag
`v0.2-hand-built-voice`: `pipeline.py` + `turn.py`). What changed is who owns the plumbing:

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

import importlib.util
import json
import string
from pathlib import Path

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
from ..logging import get_logger
from .brain import ConverseBrain
from .delivery import DeliveryTracker
from .schemas import STT_SAMPLE_RATE, TTS_SAMPLE_RATE, Transcript, VoiceEvent, Word

log = get_logger("voice")

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"  # backend/models (`make voice`)
KOKORO_MODEL = MODELS_DIR / "kokoro-v1.0.int8.onnx"
KOKORO_VOICES = MODELS_DIR / "voices-v1.0.bin"

# Whisper is trained to write clean prose, so it silently drops "um/uh" — which zeroed the filler
# metric and hid hesitations from the judge. A prompt is style context, not an instruction: one
# written WITH fillers biases verbatim output (OpenAI docs: prompting for "filler words"), and a
# spelling list fixes terms it mangled ("RAC"→RAG, "Popsub"→Pub/Sub). Limit: 224 tokens.
WHISPER_PROMPT = (
    "Umm, so, uh, I was like, hmm... you know, I mean, we, um, used Pub/Sub, RAG, MCP, "
    "Kubernetes, PostgreSQL, Redis, Kafka, gRPC, LLM, API, and, uh, CI/CD."
)

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
            return InputTransportMessageFrame(message={"type": "text", "text": text})
        if msg.get("type") == "end":
            return InputTransportMessageFrame(message=msg)
        return None  # VoiceEvents handles both control messages (they need an interruption first)


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


def voice_providers(settings) -> dict[str, list[str]]:
    """Which STT/TTS names a session may ask for. Echo always works (no keys, no downloads); the
    real ones appear when their prerequisite exists, so a fresh checkout still boots."""
    kokoro = (
        importlib.util.find_spec("kokoro_onnx") and KOKORO_MODEL.exists() and KOKORO_VOICES.exists()
    )
    return {
        "stt": ["echo"] + (["groq"] if settings.groq_api_key else []),
        "tts": ["echo"] + (["kokoro"] if kokoro else []),
    }


def make_stt(name: str, settings) -> SegmentedSTTService:
    if name == "groq":
        return GroqWordsSTT(api_key=settings.groq_api_key, model=settings.whisper_model)
    return EchoSTT()


def make_tts(name: str, settings) -> TTSService:
    if name == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        # ponytail: loads the model per session (0.3s, measured); share one if sessions pile up.
        return KokoroTTSService(
            model_path=str(KOKORO_MODEL),
            voices_path=str(KOKORO_VOICES),
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
        if isinstance(frame, InputTransportMessageFrame):
            # The input transport broadcasts these both ways: act once, on the downstream copy.
            if direction == FrameDirection.DOWNSTREAM:
                await self._control(frame.message)
            return
        if (state := self._STATES.get(type(frame))) is not None:
            await self.push_frame(event_frame(VoiceEvent(type="state", state=state)))
        elif isinstance(frame, TranscriptionFrame):
            await self.push_frame(
                event_frame(VoiceEvent(type="transcript", role="user", text=frame.text))
            )
        await self.push_frame(frame, direction)

    async def _control(self, msg: dict) -> None:
        if msg.get("type") == "text":
            # A typed turn interrupts the agent (like speaking would), then goes straight into the
            # conversation: no STT, and run_llm makes the aggregator hand the brain the context.
            await self.broadcast_interruption()
            await self.push_frame(
                LLMMessagesAppendFrame(
                    messages=[{"role": "user", "content": msg["text"]}], run_llm=True
                )
            )
        elif msg.get("type") == "end":
            # Stop talking now, then end gracefully. A bare EndWorkerFrame would flush the agent's
            # unplayed speech first (paced in real time), and the browser stops waiting for metrics
            # after 5s. The EndFrame then flows down in order, so the DeliveryTracker's metrics
            # still reach the browser before the socket closes.
            await self.broadcast_interruption()
            await self.push_frame(EndWorkerFrame(reason="client ended the session"))


# ── assembly ───────────────────────────────────────────────────────────────────────────────────


def user_turn_params(vad_stop_secs: float = 0.3) -> LLMUserAggregatorParams:
    """The turn-taking calibration that was hand-tuned in `turn.py`, as Pipecat knobs.

    VAD start 0.2s = the 200ms barge-in guard (a cough or "mm-hm" doesn't interrupt).
    VAD stop 0.3s  = Smart-Turn is asked after 0.3s of silence. It was trained on ~0.2s of trailing
                     silence: at 0.5s (the hand-built endpoint) it called a mid-thought pause
                     "complete" and split the answer (measured, see the real-model test).
    Smart-Turn stop 3.0s = give up waiting even if it says "incomplete".
    ponytail: no 60s turn cap (steady noise grows a turn forever); add a watchdog if seen live.
    """
    return LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(
            params=VADParams(confidence=0.5, start_secs=0.2, stop_secs=vad_stop_secs)
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

    @worker.event_handler("on_pipeline_error")
    async def _error(_worker, frame) -> None:
        # STT/TTS failures (quota, network) become ErrorFrames; without this the turn just goes
        # silent. The brain reports its own errors (it knows the turn is lost).
        log.error("voice_pipeline_error", error=str(frame.error))
        await worker.queue_frame(event_frame(VoiceEvent(type="error", message=str(frame.error))))

    await ws.send_text(VoiceEvent(type="state", state="listening").model_dump_json())
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
