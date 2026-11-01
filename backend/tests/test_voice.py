"""Phase 2 voice — acceptance harness (TDD style, mirrors test_graph.py).

Turn-detection logic tests stub the two model calls (deterministic, no model files needed).
`test_turn_waits_through_a_thinking_pause` runs the REAL Silero + Smart-Turn models on real
speech (macOS `say`); it skips if the models or `say` are missing. Remaining TODOs stay
skip-gated: delete the `@pytest.mark.skip` line, run `make test`, make it green.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import wave

import pytest

from app.agents.schemas import SAMPLE_SCENARIO
from app.core.providers.echo_adapter import EchoAdapter
from app.core.voice.echo_voice import EchoSTT, EchoTTS
from app.core.voice_port import STT_SAMPLE_RATE, Transcript, Word
from app.voice import turn as turn_mod
from app.voice.pipeline import VoicePipeline
from app.voice.turn import TurnDetector

# One-chunk frames (512 samples = 32ms) keep the arithmetic exact: 200ms of silence = the 7th
# quiet frame (6×32=192 < 200 ≤ 224). With the stubbed VAD, non-zero bytes = "speech".
_LOUD = b"\x00\x40" * turn_mod.CHUNK_SAMPLES
_QUIET = b"\x00\x00" * turn_mod.CHUNK_SAMPLES


def _stub(td: TurnDetector, complete: float = 1.0) -> TurnDetector:
    """Replace the model calls: speech = any non-zero sample; Smart-Turn returns `complete`."""
    td._speech_prob = lambda chunk: 1.0 if any(chunk) else 0.0
    td._turn_complete_prob = lambda: complete
    return td


def _pipeline() -> VoicePipeline:
    p = VoicePipeline(
        scenario=SAMPLE_SCENARIO, chat_port=EchoAdapter(), stt=EchoSTT(), tts=EchoTTS()
    )
    _stub(p.turn)
    return p


def _events(td: TurnDetector, frames: list[bytes]) -> list[tuple[int, str]]:
    return [(i, e.kind) for i, f in enumerate(frames) if (e := td.feed(f))]


def test_turn_ends_when_smart_turn_says_complete():
    td = _stub(TurnDetector(check_after_ms=200), complete=0.9)
    events = _events(td, [_LOUD, _LOUD] + [_QUIET] * 8)
    assert events == [(0, "speech_started"), (8, "speech_ended")]  # 7th quiet frame = 224ms
    assert td.utterance_audio


def test_turn_waits_when_incomplete_then_rechecks_after_more_speech():
    td = _stub(TurnDetector(check_after_ms=200, max_silence_ms=3000), complete=0.1)
    # "incomplete" → a ~1s pause does NOT end the turn
    assert _events(td, [_LOUD] + [_QUIET] * 30) == [(0, "speech_started")]
    # learner continues (no second speech_started — same turn), then finishes
    td._turn_complete_prob = lambda: 0.9
    assert _events(td, [_LOUD] + [_QUIET] * 8) == [(7, "speech_ended")]


def test_turn_max_silence_fallback():
    td = _stub(TurnDetector(check_after_ms=200, max_silence_ms=500), complete=0.1)
    assert _events(td, [_LOUD] + [_QUIET] * 20)[-1] == (16, "speech_ended")  # 16×32=512ms


def test_odd_frame_sizes_are_buffered():
    td = _stub(TurnDetector())
    loud = _LOUD * 2
    assert td.feed(loud[:700]) is None  # < one 512-sample chunk: buffered, no decision yet
    assert td.feed(loud[700:1400]).kind == "speech_started"


def test_vad_state_resets_after_2s_only_at_clear_silence():
    td = TurnDetector()
    td._vad_state += 1.0  # pretend the RNN has drifted
    for _ in range(70):  # ~2.2s of chunks, but never 5 quiet in a row → no reset
        td._maybe_reset_vad(0.05)
        td._maybe_reset_vad(0.9)
    assert td._vad_state.any()
    for _ in range(5):  # a run of clearly-silent chunks after the 2s mark → reset
        td._maybe_reset_vad(0.05)
    assert not td._vad_state.any() and td._since_vad_reset_ms == 0


def test_echo_pipeline_runs_stt_graph_tts():
    """One full turn over the echo stack produces transcripts + audio, no keys/network."""

    async def run():
        p = _pipeline()
        for f in [_LOUD, _LOUD] + [_QUIET] * 8:
            await p.on_audio_frame(f)
        if p._respond_task:
            await p._respond_task
        kinds, audio = [], 0
        while not p.out.empty():
            kind, payload = await p.out.get()
            if kind == "audio":
                audio += 1
            else:
                kinds.append(payload.type)
        return kinds, audio

    kinds, audio = asyncio.run(run())
    assert "transcript" in kinds
    assert audio > 0


@pytest.mark.skipif(
    not (
        turn_mod.SILERO_PATH.exists() and turn_mod.SMART_TURN_PATH.exists() and shutil.which("say")
    ),
    reason="needs `make voice` models + macOS `say`",
)
def test_turn_waits_through_a_thinking_pause(tmp_path):
    """REAL models: an unfinished clause + 1s pause must NOT end the turn; the real end must.

    Synthetic `say` audio reads fragments with sentence-final pitch, so Smart-Turn (a prosody
    model) is fooled by e.g. "Let me think about that"; this phrase is unfinished in both words
    and audio. Validate the rest with your own voice.
    """
    wav = tmp_path / "turn.wav"
    subprocess.run(
        [
            "say",
            "-o",
            str(wav),
            "--data-format=LEI16@16000",
            "I led the migration and the main reason was [[slnc 1000]]"
            " the old database could not scale.",
        ],
        check=True,
    )
    pcm = wave.open(str(wav)).readframes(10**9) + _QUIET * 50  # 1.6s trailing silence
    frames = [pcm[i : i + 1364] for i in range(0, len(pcm), 1364)]  # browser-ish frame size

    td = TurnDetector()
    events = _events(td, frames)
    kinds = [k for _, k in events]
    assert kinds == ["speech_started", "speech_ended"]  # the mid-thought pause did NOT end it
    turn_seconds = len(td.utterance_audio) / 2 / STT_SAMPLE_RATE
    assert turn_seconds > 4.0  # the turn spans BOTH phrases, not just "let me think about that"


async def _until_speaking(p: VoicePipeline) -> None:
    """Start a reply and wait until TTS audio is queued (the agent is mid-speech)."""
    for f in [_LOUD] + [_QUIET] * 8:  # a complete turn → _respond starts
        await p.on_audio_frame(f)
    while not any(kind == "audio" for kind, _ in list(p.out._queue)):
        await asyncio.sleep(0.01)


def test_barge_in_stops_and_drains():
    """≥200ms of speech mid-reply: reply cancelled, queued audio dropped, interrupt sent."""

    async def run():
        p = _pipeline()
        await _until_speaking(p)
        for f in [_LOUD] * 7:  # 7 × 32ms = 224ms of speech ≥ guard
            await p.on_audio_frame(f)
        items = [p.out.get_nowait() for _ in range(p.out.qsize())]
        kinds = [k for k, _ in items]
        events = [e.type for k, e in items if k == "event"]
        return p._respond_task.cancelled(), kinds.count("audio"), events

    cancelled, leftover_audio, events = asyncio.run(run())
    assert cancelled and leftover_audio == 0
    assert "interrupt" in events


def test_short_blip_does_not_barge_in():
    """A cough (32ms of speech) mid-reply is ignored: the reply finishes, nothing is cancelled."""

    async def run():
        p = _pipeline()
        await _until_speaking(p)
        task = p._respond_task
        for f in [_LOUD] + [_QUIET] * 8:  # blip, then silence → its turn ends while agent talks
            await p.on_audio_frame(f)
        await task
        return task.cancelled(), p._respond_task is task

    cancelled, same_task = asyncio.run(run())
    assert not cancelled and same_task  # no barge-in, and the blip didn't start a new reply


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


def test_delivery_metrics_handles_whisper_punctuation_and_phrases():
    """Real Whisper words carry punctuation/case; multi-word fillers span two words."""
    from app.voice.metrics import compute_delivery_metrics

    texts = ["Um,", "it", "was,", "like,", "hard,", "you", "know."]
    words = [Word(text=t, start=i * 0.3, end=i * 0.3 + 0.25) for i, t in enumerate(texts)]
    m = compute_delivery_metrics([Transcript(text=" ".join(texts), words=words)], 10.0)
    assert m.filler_count == 3  # um, like, you know
    assert compute_delivery_metrics([], 0).wpm == 0.0  # empty session: no crash
