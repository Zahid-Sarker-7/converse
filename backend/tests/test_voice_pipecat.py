"""Voice on Pipecat (app/voice/bot.py, brain.py, delivery.py): the pieces in isolation.

Processors run through `pipecat.tests.utils.run_test` (a real mini-pipeline) or a direct
`process_frame` call; all keyless, via the echo LLM. The WebSocket contract is in test_voice_ws.py.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import wave

import pytest
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    InputAudioRawFrame,
    InputTransportMessageFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMMessagesAppendFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    OutputTransportMessageUrgentFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.tests.utils import SleepFrame, run_test

from app.agents.schemas import SAMPLE_SCENARIO
from app.core.providers.echo_adapter import EchoAdapter
from app.voice import bot
from app.voice.brain import ConverseBrain, to_chat_messages
from app.voice.delivery import DeliveryTracker
from app.voice.schemas import Transcript, Word


def _events(frames) -> list[dict]:
    return [f.message for f in frames if isinstance(f, OutputTransportMessageUrgentFrame)]


def _context(*lines: tuple[str, str]) -> LLMContextFrame:
    return LLMContextFrame(LLMContext(messages=[{"role": r, "content": c} for r, c in lines]))


def _words(text: str, start: float = 0.0, step: float = 0.3) -> Transcript:
    return Transcript(
        text=text,
        words=[
            Word(text=w, start=start + i * step, end=start + i * step + 0.25)
            for i, w in enumerate(text.split())
        ],
    )


# ── plumbing ──────────────────────────────────────────────────────────────────


def test_serializer_speaks_the_unchanged_wire_protocol():
    s = bot.ConverseSerializer()

    async def run():
        mic = await s.deserialize(b"\x01\x00" * 10)
        typed = await s.deserialize('{"type": "text", "text": "  hi  "}')
        end = await s.deserialize('{"type": "end"}')
        junk = [
            await s.deserialize(x) for x in ("not json", "[1]", '{"type": "text", "text": " "}')
        ]
        audio = await s.serialize(OutputAudioRawFrame(b"\x02\x00", 24_000, 1))
        flush = await s.serialize(InterruptionFrame())
        event = await s.serialize(bot.event_frame(bot.VoiceEvent(type="state", state="speaking")))
        return mic, typed, end, junk, audio, flush, event

    mic, typed, end, junk, audio, flush, event = asyncio.run(run())
    assert isinstance(mic, InputAudioRawFrame) and mic.sample_rate == 16_000
    assert isinstance(typed, InputTransportMessageFrame) and typed.message == {
        "type": "text",
        "text": "hi",
    }
    assert isinstance(end, InputTransportMessageFrame) and end.message["type"] == "end"
    assert junk == [None, None, None]  # malformed control frames are ignored, not fatal
    assert audio == b"\x02\x00"
    assert '"interrupt"' in flush and '"speaking"' in event


def test_whisper_phantoms_are_dropped_but_real_short_answers_kept():
    assert bot._transcription(_words("Thank you."), "u") is None  # Whisper on keyboard noise
    assert bot._transcription(Transcript(text=""), "u") is None
    long_thanks = Transcript(text="Thank you.", words=[Word(text="Thank", start=0, end=1.5)])
    assert bot._transcription(long_thanks, "u") is not None  # 1.5s of real speech: keep it
    frame = bot._transcription(_words("Yes, we did."), "u")
    assert isinstance(frame.result, Transcript)  # word timings ride along to the tracker


def test_turn_taking_keeps_the_hand_tuned_calibration():
    p = bot.user_turn_params()
    vad = p.vad_analyzer.params
    assert (vad.start_secs, vad.stop_secs, vad.confidence) == (0.2, 0.3, 0.5)
    (stop,) = p.user_turn_strategies.stop
    assert type(stop._turn_analyzer).__name__ == "LocalSmartTurnAnalyzerV3"
    assert stop._turn_analyzer.params.stop_secs == 3.0


@pytest.mark.skipif(not shutil.which("say"), reason="needs macOS `say` for real speech")
def test_real_models_keep_a_thinking_pause_inside_one_turn(tmp_path):
    """REAL Silero + Smart-Turn with our calibration: an unfinished clause + 1s pause must NOT end
    the turn; the real end must. (With VAD stop 0.5s this split into two turns.) Runs in real time:
    the VAD and Smart-Turn timers are wall-clock."""
    from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair

    wav = tmp_path / "turn.wav"
    phrase = "The main reason we did it was because [[slnc 1000]] the old database could not scale."
    subprocess.run(["say", "-o", str(wav), "--data-format=LEI16@16000", phrase], check=True)
    pcm = wave.open(str(wav)).readframes(10**9) + b"\0\0" * 16_000 * 2  # 2s trailing silence
    frames = []
    for i in range(0, len(pcm), 640):  # 20ms mic frames, paced like a real mic
        frames += [InputAudioRawFrame(pcm[i : i + 640], 16_000, 1), SleepFrame(0.02)]

    user = LLMContextAggregatorPair(LLMContext(), user_params=bot.user_turn_params()).user()
    down, _ = asyncio.run(run_test(Pipeline([bot.EchoSTT(), user]), frames_to_send=frames))
    turns = [
        type(f) for f in down if isinstance(f, (UserStartedSpeakingFrame, UserStoppedSpeakingFrame))
    ]
    assert turns == [UserStartedSpeakingFrame, UserStoppedSpeakingFrame]


def test_context_becomes_graph_messages_with_unanswered_lines_joined():
    msgs = to_chat_messages(
        [
            {"role": "system", "content": "ignored"},
            {"role": "assistant", "content": "Tell me about a hard project."},
            {"role": "user", "content": "And for the stakeholders, we had to..."},
            {"role": "user", "content": "write a weekly status report."},
            {"role": "user", "content": [{"type": "text", "text": "non-text parts are skipped"}]},
        ]
    )
    assert [m.role for m in msgs] == ["assistant", "user"]
    assert msgs[1].content == "And for the stakeholders, we had to... write a weekly status report."


# ── ConverseBrain ────────────────────────────────────────────────────────────────────


def _brain() -> ConverseBrain:
    return ConverseBrain(scenario=SAMPLE_SCENARIO, chat_port=EchoAdapter())


def test_brain_answers_a_user_turn_with_llm_response_frames():
    down, _ = asyncio.run(
        run_test(
            _brain(), frames_to_send=[_context(("assistant", "Hi?"), ("user", "Hello there."))]
        )
    )
    replies = [f.text for f in down if isinstance(f, LLMTextFrame)]
    assert replies == ["echo> Hello there."]  # the LangGraph loop answered, via the echo LLM
    events = _events(down)
    assert {"type": "transcript", "role": "assistant", "text": replies[0]} in [
        {k: v for k, v in e.items() if v is not None} for e in events
    ]
    assert any(e.get("state") == "thinking" for e in events)


def test_brain_ignores_speculative_and_answered_turns():
    speculative = _context(("user", "Hello."))
    speculative.speculation = True
    answered = _context(("user", "Hi."), ("assistant", "Hey."))
    real = _context(("user", "Real turn."))
    down, _ = asyncio.run(run_test(_brain(), frames_to_send=[speculative, answered, real]))
    assert [f.text for f in down if isinstance(f, LLMTextFrame)] == ["echo> Real turn."]


def test_unfinished_turn_waits_and_an_interruption_cancels_the_wait(monkeypatch):
    """'we had to...' → hold; the learner resumes (interruption) → no reply to the fragment; the
    joined turn gets ONE reply."""
    from app.voice import brain as brain_mod

    monkeypatch.setattr(brain_mod, "UNFINISHED_HOLD_S", 0.5)
    first = ("user", "And for the stakeholders, we had to...")
    down, _ = asyncio.run(
        run_test(
            _brain(),
            frames_to_send=[
                _context(first),
                SleepFrame(0.1),
                InterruptionFrame(),
                _context(first, ("user", "write a weekly status report.")),
            ],
        )
    )
    replies = [f.text for f in down if isinstance(f, LLMTextFrame)]
    assert replies == ["echo> And for the stakeholders, we had to... write a weekly status report."]


# ── DeliveryTracker ──────────────────────────────────────────────────────────────────


def test_tracker_turns_frames_into_delivery_metrics(monkeypatch):
    """Two spoken turns, a 1.0s pause inside the learner's floor, 2s of agent speech → metrics
    sent before the EndFrame. (A pause across the agent's turn is listening, not a pause.)"""
    from app.voice import delivery as delivery_mod

    clock = iter([100.0, 102.0])  # BotStarted at t=100, BotStopped at t=102 → 2s of agent speech
    monkeypatch.setattr(delivery_mod.time, "time", lambda: next(clock))
    turn1, turn2 = _words("um I led the migration"), _words("it was hard")
    up = FrameDirection.UPSTREAM  # VAD + bot-speaking frames arrive from downstream processors
    frames = [
        (VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=10.2), up),
        (VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=11.5), up),  # quiet since 11.0
        (VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=12.2), up),  # spoke at 12.0 → 1.0s
        (TranscriptionFrame(turn1.text, "u", "t", result=turn1), FrameDirection.DOWNSTREAM),
        (VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=13.5), up),
        (UserStoppedSpeakingFrame(), up),  # turn over: the wait for the agent is not a pause
        (BotStartedSpeakingFrame(), up),
        (BotStoppedSpeakingFrame(), up),
        (VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=30.2), up),  # 17s later: listening
        (VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=31.5), up),
        (UserStoppedSpeakingFrame(), up),  # ended before the agent spoke (thinking / slow TTS)…
        (VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=36.2), up),  # …so 5s isn't a pause
        (TranscriptionFrame(turn2.text, "u", "t", result=turn2), FrameDirection.DOWNSTREAM),
    ]

    async def run():
        tracker = DeliveryTracker()
        sent: list = []

        async def capture(frame, direction=FrameDirection.DOWNSTREAM):
            sent.append(frame)

        tracker.push_frame = capture  # isolate the processor: no pipeline needed
        for frame, direction in frames:
            await tracker.process_frame(frame, direction)
        await tracker.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)
        return sent

    sent = asyncio.run(run())
    end_at = next(i for i, f in enumerate(sent) if isinstance(f, EndFrame))
    (metrics,) = [e["metrics"] for e in _events(sent[:end_at]) if e["type"] == "metrics"]
    assert metrics["spoken_turns"] == 2
    assert metrics["filler_count"] == 1
    assert metrics["longest_pause_s"] == 1.0
    speaking = metrics["speaking_time_s"]  # 1.45s + 0.85s of word timings
    assert round(metrics["talk_ratio"], 3) == round(speaking / (speaking + 2.0), 3)


def test_brain_reports_a_failing_llm_and_keeps_the_call_alive():
    class DownLLM(EchoAdapter):
        async def stream_chat(self, messages, **kw):
            raise RuntimeError("quota exceeded")
            yield  # pragma: no cover — makes this an async generator

    brain = ConverseBrain(scenario=SAMPLE_SCENARIO, chat_port=DownLLM())
    down, _ = asyncio.run(run_test(brain, frames_to_send=[_context(("user", "Hello."))]))
    assert not [f for f in down if isinstance(f, LLMTextFrame)]
    errors = [e for e in _events(down) if e["type"] == "error"]
    assert errors and "quota exceeded" in errors[0]["message"]
    assert _events(down)[-1]["state"] == "listening"  # the learner can try again


def test_typed_turn_interrupts_the_agent_then_joins_the_conversation():
    down, up = asyncio.run(
        run_test(
            bot.VoiceEvents(),
            frames_to_send=[InputTransportMessageFrame(message={"type": "text", "text": "Hi"})],
        )
    )
    assert any(isinstance(f, InterruptionFrame) for f in up)  # stops the agent mid-sentence
    (append,) = [f for f in down if isinstance(f, LLMMessagesAppendFrame)]
    assert append.messages == [{"role": "user", "content": "Hi"}] and append.run_llm
