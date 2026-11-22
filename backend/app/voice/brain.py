"""ConverseBrain — the Phase-1 LangGraph loop as a Pipecat processor (it replaces an LLM service).

A normal Pipecat bot has `LLMService` here. Ours is the role-play graph (director → role-player),
exactly as `/api/roleplay/turn` calls it. Pipecat owns the conversation (`LLMContext`): the user
aggregator appends each finished user turn and sends us an `LLMContextFrame`; we answer with the
standard LLM response frames, which the TTS speaks and the assistant aggregator writes back.

    LLMContextFrame ─► ConverseBrain ─► LLMFullResponseStartFrame
                                        LLMTextFrame(reply)        ─► TTS ─► …
                                        LLMFullResponseEndFrame
"""

from __future__ import annotations

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..agents.graph import build_graph
from ..agents.schemas import ScenarioConfig
from ..core.model_port import ChatMessage, ModelPort, Role
from .pipeline import VOICE_STYLE


def to_chat_messages(context_messages: list[dict]) -> list[ChatMessage]:
    """Pipecat's context → the graph's `ChatMessage`s, joining consecutive user lines.

    Consecutive user lines happen when the agent never answered the first one (the learner kept
    talking and interrupted it, or resumed during the unfinished-hold): it's one answer, so join it
    rather than hand the LLM fragments. Non-text and system/developer messages are skipped.
    """
    out: list[ChatMessage] = []
    for m in context_messages:
        role, content = m.get("role"), m.get("content")
        if role not in (Role.user, Role.assistant) or not isinstance(content, str) or not content:
            continue
        if role == Role.user and out and out[-1].role == Role.user:
            out[-1] = ChatMessage(role=Role.user, content=f"{out[-1].content} {content}")
        else:
            out.append(ChatMessage(role=Role(role), content=content))
    return out


class ConverseBrain(FrameProcessor):
    def __init__(self, *, scenario: ScenarioConfig, chat_port: ModelPort, **kwargs) -> None:
        super().__init__(**kwargs)
        # Spoken replies must be short and plain: TTS reads markdown/lists aloud verbatim.
        self.scenario = scenario.model_copy(
            update={"persona": f"{scenario.persona}. {VOICE_STYLE}"}
        )
        self.graph = build_graph(chat_port=chat_port, judge_port=chat_port)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """### TODO #1: The brain as a frame processor  [track: C-project]  [status: todo]

        - What:  answer each finished user turn with the role-play graph. On an `LLMContextFrame`:
                 (1) skip it if `frame.speculation` (a provisional turn must never reach the user);
                 (2) `to_chat_messages(frame.context.get_messages())`; do nothing unless the last
                     message is the user's;
                 (3) send a "thinking" `state` event; if `_looks_unfinished(last line)`: send
                     "listening", `await asyncio.sleep(UNFINISHED_HOLD_S)`, then "thinking" again;
                 (4) `await self.graph.ainvoke({"scenario": self.scenario, "messages": msgs,
                     "mode": "chat"})` — exactly what `/api/roleplay/turn` does; reply = the
                     last message;
                 (5) send the assistant `transcript` event, then push the three LLM response
                     frames: `LLMFullResponseStartFrame`, `LLMTextFrame(reply)`,
                     `LLMFullResponseEndFrame`.
                 Every OTHER frame: pass it on unchanged, in the direction it came (as below).
        - Why:   this is the core Pipecat idea: everything is a frame flowing through processors,
                 and a processor is `process_frame` + `push_frame`. Swapping Pipecat's `LLMService`
                 for your LangGraph brain is the "framework, but our brain" decision in one class.
                 Watch what you DON'T write: no task, no cancel, no barge-in code. When the learner
                 talks over the agent, Pipecat sends an `InterruptionFrame` and cancels whatever
                 `process_frame` is awaiting (your graph call, or the hold's sleep). Compare with
                 `_respond_task` + `_barge_in` in `pipeline.py` — that's what the framework bought.
        - How:   events are `bot.event_frame(VoiceEvent(...))` (import inside the method: `bot`
                 imports this module). Push frames with `await self.push_frame(frame)`. Keep
                 `await super().process_frame(frame, direction)` as the first line — it's how
                 Pipecat's own Start/End/Interruption handling runs.
        - Working with: `to_chat_messages` + `__init__` above (already done) · `VOICE_STYLE`,
                 `_looks_unfinished`, `UNFINISHED_HOLD_S` (pipeline.py, live-tuned) · events the
                 browser expects: `schemas.VoiceEvent`.
        - Docs & SDKs:  https://docs.pipecat.ai/pipecat/fundamentals/custom-frame-processor
                 https://docs.pipecat.ai/pipecat/fundamentals/interruptions
                 https://docs.pipecat.ai/pipecat/learn/context-management
                 reference: `pipecat/services/llm_service.py` (how a real LLM service does it).
        - Acceptance: delete the `@TODO_BRAIN` lines in `tests/test_voice_pipecat.py` → green
                 (reply via the echo LLM; speculative/answered turns ignored; an interruption
                 cancels the unfinished-hold and the joined turn gets ONE reply).
        """
        await super().process_frame(frame, direction)
        # TODO #1: handle LLMContextFrame as described above. Until then the agent stays silent.
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
