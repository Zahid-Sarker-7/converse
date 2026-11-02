"""Role-play routes — the Phase 1 multi-agent loop over HTTP.

  POST /api/roleplay/turn   → one learner turn: Director + Role-Player reply (mode="chat")
  POST /api/roleplay/score  → score the transcript: Evaluator → Scorecard   (mode="score")
  GET  /api/roleplay/sample → a ready-to-use ScenarioConfig to try the loop

Both POSTs run the SAME compiled graph, differing only by `state["mode"]`. The route is
provider-agnostic: it resolves a `ModelPort` from the registry and hands it to the graph.
These return plain JSON (not SSE) — token streaming of the role-player turn is a Phase 2 concern.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..agents.graph import build_graph
from ..agents.schemas import SAMPLE_SCENARIO, ScenarioConfig, Scorecard
from ..core.model_port import ChatMessage, Role

router = APIRouter(prefix="/roleplay")


class TurnRequest(BaseModel):
    scenario: ScenarioConfig
    messages: list[ChatMessage]  # transcript so far, including the latest user turn
    provider: str | None = None  # None -> DEFAULT_PROVIDER


class TurnResponse(BaseModel):
    reply: str
    director_note: str | None = None
    turn_count: int


class ScoreRequest(BaseModel):
    scenario: ScenarioConfig
    messages: list[ChatMessage]  # the full transcript to grade
    provider: str | None = None  # a structured-output-capable provider (Gemini recommended)


def _resolve_port(request: Request, provider: str | None):
    settings = request.app.state.settings
    registry: dict = request.app.state.registry
    name = provider or settings.default_provider
    port = registry.get(name)
    if port is None:
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{name}' is not configured. Available: {list(registry)}",
        )
    return port


@router.get("/sample")
async def sample() -> ScenarioConfig:
    return SAMPLE_SCENARIO


async def _run(port, state: dict) -> dict:
    """Build + run the graph, turning a not-yet-built graph / TODO node into a clean 501.

    `build_graph` and the agent nodes raise `NotImplementedError` until you implement the
    LangGraph TODOs in `agents/graph.py`; this maps that to a 501 so the API stays usable.
    """
    try:
        graph = build_graph(chat_port=port, judge_port=port)
        return await graph.ainvoke(state)
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc


@router.post("/turn")
async def turn(body: TurnRequest, request: Request) -> TurnResponse:
    port = _resolve_port(request, body.provider)
    result = await _run(
        port, {"scenario": body.scenario, "messages": body.messages, "mode": "chat"}
    )
    messages: list[ChatMessage] = result["messages"]
    reply = next((m.content for m in reversed(messages) if m.role == Role.assistant), "")
    turn_count = sum(1 for m in messages if m.role == Role.assistant)
    return TurnResponse(
        reply=reply, director_note=result.get("director_note"), turn_count=turn_count
    )


@router.post("/score")
async def score(body: ScoreRequest, request: Request) -> Scorecard:
    port = _resolve_port(request, body.provider)
    result = await _run(
        port, {"scenario": body.scenario, "messages": body.messages, "mode": "score"}
    )
    card = result.get("scorecard")
    if card is None:  # Evaluator not implemented yet, or returned nothing
        raise HTTPException(status_code=501, detail="Evaluator produced no scorecard.")
    return card
