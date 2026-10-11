"""Phase 2 voice — your acceptance harness (TDD style, mirrors test_graph.py).

The first two tests run today (they prove the scaffold + echo pipeline work). The last three
are SKIPPED and tied to the three hands-on TODOs. When you reach a TODO:
  1. delete its `@pytest.mark.skip(...)` line,
  2. run `make test`,
  3. implement until it goes green.
`make test` stays green meanwhile (skips count as passing).
"""

from __future__ import annotations

import asyncio

import pytest

from app.agents.schemas import SAMPLE_SCENARIO
from app.core.providers.echo_adapter import EchoAdapter
from app.core.voice.echo_voice import EchoSTT, EchoTTS
from app.core.voice_port import STT_SAMPLE_RATE, Transcript, Word
from app.voice.pipeline import VoicePipeline
from app.voice.turn import TurnDetector

# A loud 100ms PCM16 frame (speech) and a silent one, for driving the turn detector.
_LOUD = b"\x00\x40" * (STT_SAMPLE_RATE // 10)  # amplitude ~16384
_QUIET = b"\x00\x00" * (STT_SAMPLE_RATE // 10)


def _pipeline() -> VoicePipeline:
    return VoicePipeline(
        scenario=SAMPLE_SCENARIO, chat_port=EchoAdapter(), stt=EchoSTT(), tts=EchoTTS()
    )


def test_turn_detector_emits_start_then_end():
    """Fallback detector: loud frame starts a turn; enough silence ends it."""
    td = TurnDetector(silence_ms=200)
    assert td.feed(_LOUD).kind == "speech_started"
    assert td.feed(_LOUD) is None
    # feed silence until it crosses the 200ms threshold (100ms frames)
    assert td.feed(_QUIET) is None
    assert td.feed(_QUIET).kind == "speech_ended"
    assert td.utterance_audio  # the utterance was accumulated for STT


def test_echo_pipeline_runs_stt_graph_tts():
    """One full turn over the echo stack produces transcripts + audio, no keys/network."""

    async def run():
        p = _pipeline()
        # speak (2 frames), then go silent past the 700ms threshold (8×100ms) so a turn completes
        await p.on_audio_frame(_LOUD)
        await p.on_audio_frame(_LOUD)
        for _ in range(8):
            await p.on_audio_frame(_QUIET)
        if p._respond_task:
            await p._respond_task
        kinds = []
        audio = 0
        while not p.out.empty():
            kind, payload = await p.out.get()
            if kind == "audio":
                audio += 1
            else:
                kinds.append(payload.type)
        return kinds, audio

    kinds, audio = asyncio.run(run())
    assert "transcript" in kinds  # user + assistant transcripts emitted
    assert audio > 0  # TTS produced audio frames


@pytest.mark.skip(reason="TODO #1: implement Silero VAD + Smart-Turn v3 in voice/turn.py")
def test_turn_waits_through_a_thinking_pause():
    """A 700ms gap mid-sentence must NOT end the turn (the whole point of Smart-Turn v3).

    The fallback energy-timer ends on ANY silence >silence_ms, so this fails until you add the
    semantic 'did they finish?' stage. Drive it with real audio clips of a mid-thought pause.
    """
    raise AssertionError("implement test with a 'let me think…' clip once TODO #1 lands")


@pytest.mark.skip(reason="TODO #2: implement barge-in (stop <60ms + drain + guard) in pipeline.py")
def test_barge_in_stops_and_drains():
    """Starting to speak mid-reply cancels the response task AND leaves no stale audio queued."""

    async def run():
        p = _pipeline()
        # trigger a response, then barge in while it's speaking
        await p.on_audio_frame(_LOUD)
        await p.on_audio_frame(_QUIET)
        await p.on_audio_frame(_QUIET)
        await asyncio.sleep(0.05)
        await p.on_audio_frame(_LOUD)  # barge-in
        # after a correct barge-in: task cancelled, and no ("audio", ...) left in the queue
        leftover_audio = sum(1 for _ in range(p.out.qsize()) if p.out.get_nowait()[0] == "audio")
        return p._respond_task.cancelled(), leftover_audio

    cancelled, leftover_audio = asyncio.run(run())
    assert cancelled and leftover_audio == 0


@pytest.mark.skip(reason="TODO #3: implement compute_delivery_metrics in voice/metrics.py")
def test_delivery_metrics_from_timestamps():
    """Known word timings -> expected WPM / filler / pause numbers."""
    from app.voice.metrics import compute_delivery_metrics

    # 6 words over 3.0s of speaking time = 120 WPM; one filler ("um"); a 1.0s internal pause.
    turn = Transcript(
        text="um I really enjoyed the project",
        words=[
            Word(text="um", start=0.0, end=0.3),
            Word(text="I", start=0.3, end=0.5),
            Word(text="really", start=1.5, end=1.9),  # 1.0s pause before this
            Word(text="enjoyed", start=1.9, end=2.3),
            Word(text="the", start=2.3, end=2.5),
            Word(text="project", start=2.5, end=3.0),
        ],
    )
    m = compute_delivery_metrics([turn], total_conversation_s=6.0)
    assert round(m.wpm) == 120
    assert m.filler_count == 1
    assert round(m.longest_pause_s, 1) == 1.0
    assert round(m.talk_ratio, 2) == 0.5
