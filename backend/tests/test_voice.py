"""Voice logic that doesn't need a pipeline: delivery-metric maths and the unfinished-turn check.

Pipeline behaviour (turns, barge-in, typed turns, metrics from frames) is in test_voice_pipecat.py;
the WebSocket contract in test_voice_ws.py.
"""

from __future__ import annotations

from app.voice.schemas import Transcript, Word


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
    m = compute_delivery_metrics([turn], agent_speaking_s=3.0)  # 3s you / (3+3)s = 0.5
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


def test_talk_ratio_is_against_agent_speech_not_the_wall_clock():
    """Typing or silence must not deflate talk:listen (it read 11% in a mixed session)."""
    from app.voice.metrics import compute_delivery_metrics

    turn = Transcript(
        text="hi there", words=[Word(text="hi", start=0, end=1), Word(text="there", start=1, end=2)]
    )
    m = compute_delivery_metrics([turn], agent_speaking_s=6.0)  # 2s you vs 6s agent
    assert round(m.talk_ratio, 2) == 0.25 and m.spoken_turns == 1


def test_unfinished_text_detection():
    from app.voice.brain import _looks_unfinished

    for t in [
        "For the communication with stakeholders, we had to...",
        "The main reason was",
        "We used a queue and",
        "So, um,",
    ]:
        assert _looks_unfinished(t), t
    for t in ["We shipped a week earlier.", "I don't remember.", "Is that right?", "Thanks."]:
        assert not _looks_unfinished(t), t
