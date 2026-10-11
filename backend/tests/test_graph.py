"""Phase 1 graph — your acceptance harness (TDD style).

Each test is SKIPPED and tied to a TODO in `app/agents/graph.py`. When you reach that TODO:
  1. delete its `@pytest.mark.skip(...)` line,
  2. run `make test`,
  3. implement until it goes green.

`make test` stays green meanwhile (skips count as passing).
"""

from __future__ import annotations

import asyncio

from app.agents.schemas import SAMPLE_SCENARIO
from app.core.model_port import ChatMessage, Role
from app.core.providers.echo_adapter import EchoAdapter


def test_director_picks_curveball_by_turn():
    from app.agents.graph import scenario_director

    # No assistant turns yet → first curveball.
    assert (
        scenario_director({"scenario": SAMPLE_SCENARIO, "messages": []})["director_note"]
        == SAMPLE_SCENARIO.curveballs[0]
    )
    # One assistant turn done → second curveball.
    msgs = [
        ChatMessage(role=Role.user, content="hi"),
        ChatMessage(role=Role.assistant, content="Tell me about a conflict."),
    ]
    assert (
        scenario_director({"scenario": SAMPLE_SCENARIO, "messages": msgs})["director_note"]
        == SAMPLE_SCENARIO.curveballs[1]
    )
    # Past the end of the list → no note (don't crash).
    many = [ChatMessage(role=Role.assistant, content="x")] * 99
    assert (
        scenario_director({"scenario": SAMPLE_SCENARIO, "messages": many})["director_note"]
        is None
    )


def test_route_mode():
    from app.agents.graph import route_mode

    assert route_mode({"mode": "chat"}) == "chat"
    assert route_mode({"mode": "score"}) == "score"
    assert route_mode({}) == "chat"  # default


def test_chat_path_appends_a_roleplayer_message():
    from app.agents.graph import build_graph

    port = EchoAdapter()  # no keys/network; echoes the last user message as the "reply"
    graph = build_graph(chat_port=port, judge_port=port)
    result = asyncio.run(
        graph.ainvoke(
            {
                "scenario": SAMPLE_SCENARIO,
                "messages": [ChatMessage(role=Role.user, content="hello")],
                "mode": "chat",
            }
        )
    )
    assistant_turns = [m for m in result["messages"] if m.role == Role.assistant]
    assert len(assistant_turns) == 1  # role_player appended exactly one reply


def test_score_path_produces_a_scorecard():
    from app.agents.graph import build_graph
    from app.agents.schemas import CriterionScore, Scorecard
    from app.core.model_port import ModelPort

    # A fake judge so we can test the graph's score path without keys — it just proves the
    # Evaluator node returns whatever structured_output gives it under "scorecard".
    class FakeJudge(ModelPort):
        name = "fake-judge"

        async def stream_chat(self, messages, **kw):  # unused on the score path
            yield  # pragma: no cover

        async def structured_output(self, messages, schema, *, model=None):
            return Scorecard(
                criteria=[
                    CriterionScore(criterion="clarity", score=4, evidence="…", rationale="…")
                ],
                overall=4,
                summary="ok",
                focus_next="be specific",
            )

    port = FakeJudge()
    graph = build_graph(chat_port=port, judge_port=port)
    result = asyncio.run(
        graph.ainvoke(
            {
                "scenario": SAMPLE_SCENARIO,
                "messages": [ChatMessage(role=Role.user, content="my answer")],
                "mode": "score",
            }
        )
    )
    assert isinstance(result["scorecard"], Scorecard)
