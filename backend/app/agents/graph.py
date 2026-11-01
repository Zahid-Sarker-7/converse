"""The LangGraph graph that orchestrates the three agents — YOUR hands-on build.

This file is intentionally a SKELETON. The LangGraph pieces (state, nodes, edges, compile)
are the whole point of Phase 1, so you implement them — follow the TODO ladder below and
find each API in the docs (links per TODO). `agents/schemas.py` (the data models) and
`api/agents.py` (the HTTP wiring that calls `build_graph`) are already done for you.

WHAT WE'RE ACHIEVING WITH LANGGRAPH
-----------------------------------
A role-play loop as a small state machine. One graph, two entry paths chosen by a "mode":

    START ──route(mode)──┬─ "chat"  ─→ scenario_director ─→ role_player ─┐
                         │                                               ├─→ END
                         └─ "score" ───────────────→ evaluator ─────────┘

  - chat  turn: Director frames the scene + picks a curveball; Role-Player replies in character.
  - score turn: Evaluator reads the full transcript and returns a rubric `Scorecard`.

The engine only ever calls `build_graph(...)` then `graph.ainvoke(state)`. Everything else
is internal to this file.

THE MENTAL MODEL (hold this the whole way)
  - State : a TypedDict "clipboard" that flows through the graph. A node returns only the
            keys it changed; LangGraph merges them. A key can have a *reducer* (e.g.
            operator.add) so returning {"messages":[m]} APPENDS instead of overwrites.
  - Node  : a plain function `(state) -> dict`. No base class. This is why your ModelPort
            adapters plug straight in — a node just calls `port.stream_chat(...)`.
  - Edge  : "after A go to B"; a conditional edge picks the next node at runtime.
  - compile(): turns the builder into a runnable graph you `await graph.ainvoke(state)`.

TODO LADDER (do them in order — each builds on the last)
  T1  Define the graph state          -> ConverseState (TypedDict + reducer)
  T2  Write the Scenario-Director node -> first node: read state, return an update
  T3  Write the mode router            -> route_mode(): pick the path
  T4  Build & compile the graph        -> build_graph(): nodes + edges + compile
  T5  Write the Role-Player node       -> a node that calls the chat port (async)
  T6  Write the Evaluator node         -> LLM-judge + structured_output on an adapter

DOCS (start here every step)
  Overview     https://docs.langchain.com/oss/python/langgraph/overview
  Quickstart   https://docs.langchain.com/oss/python/langgraph/quickstart
  Graph API    https://docs.langchain.com/oss/python/langgraph/use-graph-api
  Persistence  https://docs.langchain.com/oss/python/langgraph/persistence   (Phase 5)
  StateGraph   https://reference.langchain.com/python/langgraph/graph/state/StateGraph

Verify tests as you go: `make test` (the checks in tests/test_graph.py are skip-gated —
unskip each one when you reach its TODO, then make it pass — a red→green loop).
"""

from __future__ import annotations

# You'll add imports as you go. Expect to need (find the exact lines in the docs):
import operator
from typing import Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from ..core.model_port import ChatMessage, ModelPort, Role
from .schemas import ScenarioConfig, Scorecard

# ──────────────────────────────────────────────────────────────────────────────
# WORKED DATA EXAMPLE — roles matter: the MODEL plays the interviewer.
#   role="assistant" = the Role-Player / persona (the AI interviewer) — model OUTPUT
#   role="user"      = the learner (the human being interviewed)      — model INPUT
#
# One chat turn = the learner just answered; role_player appends the interviewer's next line.
#
#   messages coming IN (interviewer already opened, learner already replied):
#     assistant: "Tell me about a conflict you handled."          # interviewer, turn 1
#     user:      "On a migration project we fell behind; I         # learner's answer
#                 proposed cutting scope and aligned the team."
#
#   scenario_director: assistant-turns so far = 1  ->  curveballs[1]
#   role_player returns {"messages": [assistant: "Good — what would you do differently?"]}
#   operator.add reducer APPENDS it  ->  transcript now has 3 messages (history preserved).
#
# Curveball indexing (curveballs[N] fires on the N-th interviewer/assistant turn, 0-based):
#     0 assistant turns yet  -> curveballs[0]   (fires on the OPENING interviewer line)
#     1 assistant turn done  -> curveballs[1]
#     2 assistant turns done -> curveballs[2]
#     past the end of list   -> None            (no curveball this turn; don't crash)
#   Design choice for T2: if you'd rather the OPENER be a plain question and curveballs start
#   only AFTER the learner has spoken, offset by one (e.g. key off the learner/user-turn count).
#
# Evaluator (T6) grades the LEARNER, so pull `evidence` quotes from the user (learner) turns.
# ──────────────────────────────────────────────────────────────────────────────


# ── T1: Define the graph state ────────────────────────────────────────────────
# ### TODO: ConverseState   [track: C-project]  [status: done]
# - What:  a TypedDict the graph threads through. Fields you need:
#            scenario (ScenarioConfig) · messages (list[ChatMessage]) ·
#            mode ("chat"|"score") · director_note (str|None) · scorecard (Scorecard|None)
# - Why:   this is LangGraph's "state" — plus your first reducer. `messages` must ACCUMULATE
#          across turns, so annotate it with a reducer:
#          `Annotated[list[ChatMessage], operator.add]`. Everything else is last-write-wins.
# - How:   1. read Graph API → "Define state" and "Reducers".
#          2. define `class ConverseState(TypedDict, total=False): ...` with the fields above.
#          3. annotate `messages` with the operator.add reducer.
# - Docs:  https://docs.langchain.com/oss/python/langgraph/use-graph-api  (State, Reducers)
# - Accept: `from app.agents.graph import ConverseState` imports; messages field is annotated.


class ConverseState(TypedDict, total=False):
    scenario: ScenarioConfig
    messages: Annotated[list[ChatMessage], operator.add]
    mode: Literal["chat", "score"]
    director_note: str | None
    scorecard: Scorecard | None


# ── T2: Scenario-Director node ────────────────────────────────────────────────
def scenario_director(state: ConverseState):  # noqa: ANN001
    """### TODO: Scenario-Director node   [track: C-project]  [status: done]
    - What:  a node that picks the next curveball for this turn and returns
             `{"director_note": <str or None>}`.
    - Why:   your FIRST node — learn the node contract: receive `state`, return a partial
             dict update. No LLM needed (rule-based is fine for Phase 1).
    - How:   1. read Graph API → "Nodes".
             2. count how many role-player turns happened:
                  turn = number of messages in state with role == assistant.
             3. return the turn-th curveball from `state["scenario"].curveballs`
                (or None if you've run past the end — don't crash).
    - Working with: your `ScenarioConfig` (schemas.py) + `Role` enum (core/model_port.py).
    - Docs:  https://docs.langchain.com/oss/python/langgraph/use-graph-api  (Nodes)
    - Accept: returns curveballs[0] on an empty transcript; curveballs[1] after one assistant
              turn; None when the transcript is longer than the curveball list.
    """
    turn = sum(1 for m in state.get("messages", []) if m.role == Role.assistant)
    curveballs = state["scenario"].curveballs
    note = curveballs[turn] if turn < len(curveballs) else None
    return {"director_note": note}


# ── T3: Mode router ───────────────────────────────────────────────────────────
def route_mode(state):  # noqa: ANN001
    """### TODO: route_mode (conditional edge)   [track: C-project]  [status: done]
    - What:  a function `(state) -> "chat" | "score"` that decides which path START takes.
    - Why:   this is how LangGraph branches. `add_conditional_edges` (T4) calls it and maps
             its return value to a node.
    - How:   1. read Graph API → "Conditional edges".
             2. return `state.get("mode", "chat")`.
    - Docs:  https://docs.langchain.com/oss/python/langgraph/use-graph-api  (Conditional edges)
    - Accept: returns "chat"/"score" for those modes; defaults to "chat" when mode is absent.
    """
    return state.get("mode", "chat")


def build_graph(chat_port: ModelPort, judge_port: ModelPort):
    """Wire and compile the graph. Define the two model-using nodes INSIDE this function so
    they close over `chat_port` / `judge_port` (nodes take only `state`, so a closure is the
    simplest way to hand them a model).

    ### TODO: T4 — build & compile the graph   [track: C-project]  [status: done]
    - How:   1. read Quickstart + Graph API → "Build the graph".
             2. `builder = StateGraph(ConverseState)`
             3. add three nodes: "scenario_director", "role_player" (T5), "evaluator" (T6).
             4. conditional ENTRY: `builder.add_conditional_edges(START, route_mode,
                {"chat": "scenario_director", "score": "evaluator"})`.
             5. edges: scenario_director → role_player → END; evaluator → END.
             6. `return builder.compile()`.
    - Docs:  https://docs.langchain.com/oss/python/langgraph/use-graph-api
             https://reference.langchain.com/python/langgraph/graph/state/StateGraph
    - Accept: `build_graph(echo, echo)` returns a compiled graph; a chat-mode `ainvoke`
              reaches role_player, a score-mode `ainvoke` reaches evaluator.

    ### TODO: T5 — Role-Player node (define below, inside build_graph)
    - What:  `async def role_player(state) -> dict` that appends the persona's in-character reply.
    - How:   1. build a system ChatMessage from `state["scenario"]` (persona/objective/difficulty)
                + `state["director_note"]` if set.
             2. `msgs = [system, *state["messages"]]`.
             3. collect the stream from the closed-over `chat_port`:
                  `reply = "".join([c.delta async for c in chat_port.stream_chat(msgs)])`
             4. `return {"messages": [ChatMessage(role=Role.assistant, content=reply)]}`
                (the operator.add reducer appends it).
    - Docs:  ModelPort: core/model_port.py (same interface as /api/chat).
    - Accept: a coherent, in-character 10-turn convo; obeys curveballs; no "As an AI".

    ### TODO: T6 — Evaluator node (define below) + structured_output on an adapter
    - What:  `async def evaluator(state) -> dict` returning `{"scorecard": <Scorecard>}`.
    - How A) implement `structured_output` on ONE adapter (`judge_port`):
             • Gemini (recommended): `self._client.aio.models.generate_content(model=...,
               contents=..., config={"response_mime_type":"application/json",
               "response_schema": schema})` then `schema.model_validate_json(resp.text)`.
               This is a SEPARATE one-shot call — it does NOT touch your Interactions
               streaming path. (Verified present on google-genai 2.20.0.)
             • Groq: `chat.completions.create(..., response_format={"type":"json_schema",
               "json_schema":{"name":"Scorecard","strict":True,"schema":
               schema.model_json_schema()}})`. ⚠ gpt-oss-120b can ignore it → validate+retry.
      How B) in this node: build a judging prompt (rubric + transcript), call
             `card = await judge_port.structured_output(msgs, Scorecard)`, return the card.
    - Bias guards: tell the judge to ignore answer length, score each criterion independently,
      and quote a real moment as evidence (no hallucinated praise).
    - Docs:  Groq https://console.groq.com/docs/structured-outputs
             Gemini https://ai.google.dev/gemini-api/docs/structured-output
             LLM-judge pitfalls https://langfuse.com/docs/evaluation/evaluation-methods/llm-as-a-judge
    - Accept: schema-valid `Scorecard` on a sample transcript, each criterion carrying a real
              quote; reruns stable (±1).
    """

    async def role_player(state: ConverseState) -> dict:
        # build system prompt
        # send the messages with system prompt to the LLM
        scenario = state["scenario"]
        note = state.get("director_note")
        system_prompt = (
            "You are a role-playing agent. "
            f"Your persona - {scenario.persona}. "
            f"Your objective - {scenario.objective}. "
            f"The difficulty level of your persona - {scenario.difficulty}, "
            "this determines how you should conduct the conversation. "
            f"Director note - {note}, if a director note is available, "
            "pivot the next turn of the conversation based on the note; otherwise "
            "follow a natural flow based on the previous conversation, or end the "
            "conversation if there is no more to discuss."
        )
        messages = [ChatMessage(role=Role.system, content=system_prompt)] + state["messages"]
        reply = "".join([chunk.delta async for chunk in chat_port.stream_chat(messages=messages)])
        return {"messages": [ChatMessage(role=Role.assistant, content=reply)]}

    async def evaluator(state: ConverseState) -> dict:
        scenario = state["scenario"]
        system_prompt = (
            "You are an impartial judge grading a role-play. Read the full conversation "
            "between the interviewer (assistant) and the candidate (user), then grade "
            "ONLY the candidate. "
            f"Score each of these criteria from 1 to 5: {scenario.rubric}. "
            "Judge each criterion independently and ignore answer length. "
            "For every criterion, quote a VERBATIM line from the candidate's (user) turns "
            "as evidence — if you cannot quote it, score low — and give a one-line rationale."
        )
        messages = [ChatMessage(role=Role.system, content=system_prompt), *state["messages"]]
        scorecard = await judge_port.structured_output(messages=messages, schema=Scorecard)
        return {"scorecard": scorecard}

    graph_builder = StateGraph(ConverseState)
    graph_builder.add_node("scenario_director", scenario_director)
    graph_builder.add_node("role_player", role_player)
    graph_builder.add_node("evaluator", evaluator)
    graph_builder.add_conditional_edges(
        START, route_mode, {"chat": "scenario_director", "score": "evaluator"}
    )
    graph_builder.add_edge("scenario_director", "role_player")
    graph_builder.add_edge("role_player", END)
    graph_builder.add_edge("evaluator", END)
    return graph_builder.compile()
