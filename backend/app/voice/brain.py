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

import asyncio

from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from ..agents.graph import build_graph
from ..agents.schemas import ScenarioConfig
from ..core.model_port import ChatMessage, ModelPort, Role
from ..logging import get_logger
from .pipeline import UNFINISHED_HOLD_S, VOICE_STYLE, _looks_unfinished
from .schemas import VoiceEvent

log = get_logger("voice")


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
        """Answer each finished user turn; pass every other frame through untouched.

        No task, no cancel, no barge-in code: when the learner talks over the agent, Pipecat sends
        an `InterruptionFrame` and cancels whatever this method is awaiting (the graph call or the
        unfinished-hold sleep). Compare `_respond_task` + `_barge_in` in the hand-built engine.
        """
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
        elif not frame.speculation:  # a provisional turn must never reach the user
            await self._respond(frame.context.get_messages())

    async def _respond(self, context_messages: list[dict]) -> None:
        from .bot import event_frame  # bot imports this module

        messages = to_chat_messages(context_messages)
        if not messages or messages[-1].role != Role.user:
            return  # already answered (e.g. a context frame after the agent's own turn)
        await self.push_frame(event_frame(VoiceEvent(type="state", state="thinking")))
        if _looks_unfinished(messages[-1].content):
            # "...we had to" + silence: wait. If the learner resumes, the interruption cancels this
            # sleep, and their next turn lands as a consecutive user line that we join.
            log.info("held_unfinished", hold_s=UNFINISHED_HOLD_S)
            await self.push_frame(event_frame(VoiceEvent(type="state", state="listening")))
            await asyncio.sleep(UNFINISHED_HOLD_S)
            await self.push_frame(event_frame(VoiceEvent(type="state", state="thinking")))

        try:
            result = await self.graph.ainvoke(
                {"scenario": self.scenario, "messages": messages, "mode": "chat"}
            )
        except Exception as exc:  # noqa: BLE001 — provider down/quota: tell the learner, keep the call
            log.error("voice_brain_error", error=f"{type(exc).__name__}: {exc}", exc_info=True)
            await self.push_frame(
                event_frame(VoiceEvent(type="error", message=f"{type(exc).__name__}: {exc}"))
            )
            await self.push_frame(event_frame(VoiceEvent(type="state", state="listening")))
            return
        reply = result["messages"][-1].content
        log.info("llm_reply", chars=len(reply))
        await self.push_frame(
            event_frame(VoiceEvent(type="transcript", role="assistant", text=reply))
        )
        await self.push_frame(LLMFullResponseStartFrame())
        await self.push_frame(LLMTextFrame(reply))
        await self.push_frame(LLMFullResponseEndFrame())
