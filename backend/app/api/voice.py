"""Voice routes — the Phase-2 pipeline over a WebSocket.

  WS  /api/voice/session   bidirectional audio: mic frames up, TTS audio + events down
  GET /api/voice/providers readiness of the STT/TTS adapters (for the UI / health)

Why WebSocket (not SSE like /api/chat): voice is two-way and interruptible — the client streams
mic audio UP while the server streams speech DOWN, and barge-in needs both at once. SSE is
one-way, so it can't carry the mic.

Wire protocol:
  client -> server : first a TEXT frame = JSON session config (see `SessionConfig`);
                     thereafter BINARY frames = raw PCM16 @ 16 kHz mic audio,
                     and TEXT frame {"type":"end"} to end the session (-> delivery metrics).
  server -> client : TEXT frames = JSON `VoiceEvent`; BINARY frames = PCM16 @ 24 kHz to play.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from ..agents.schemas import SAMPLE_SCENARIO, ScenarioConfig
from ..logging import get_logger
from ..voice.pipeline import VoicePipeline
from ..voice.schemas import VoiceEvent

router = APIRouter(prefix="/voice")
log = get_logger("voice")


@router.get("/providers")
async def providers(request: Request) -> dict:
    stt_registry: dict = request.app.state.stt_registry
    tts_registry: dict = request.app.state.tts_registry
    return {
        "stt": [{"name": n, "ready": p.ready} for n, p in stt_registry.items()],
        "tts": [{"name": n, "ready": p.ready} for n, p in tts_registry.items()],
    }


def _resolve(registry: dict, name: str | None, default: str):
    return registry.get(name or default) or registry[default]


@router.websocket("/session")
async def session(ws: WebSocket) -> None:
    await ws.accept()
    app = ws.app
    settings = app.state.settings

    # 1. Handshake: the first text frame configures the session.
    try:
        cfg = await ws.receive_json()
    except Exception:  # noqa: BLE001
        await ws.close(code=1003)
        return

    scenario = (
        ScenarioConfig.model_validate(cfg["scenario"]) if cfg.get("scenario") else SAMPLE_SCENARIO
    )
    chat_port = _resolve(app.state.registry, cfg.get("provider"), settings.default_provider)
    stt = _resolve(app.state.stt_registry, cfg.get("stt"), settings.voice_stt_provider)
    tts = _resolve(app.state.tts_registry, cfg.get("tts"), settings.voice_tts_provider)

    pipeline = VoicePipeline(
        scenario=scenario, chat_port=chat_port, stt=stt, tts=tts, tts_voice=settings.kokoro_voice
    )

    # 2. Pump: continuously flush the pipeline's out-queue to the socket.
    async def pump() -> None:
        while True:
            kind, payload = await pipeline.out.get()
            if kind == "audio":
                await ws.send_bytes(payload)  # type: ignore[arg-type]
            else:  # "event"
                await ws.send_text(VoiceEvent.model_validate(payload).model_dump_json())

    pump_task = asyncio.create_task(pump())
    await pipeline._emit(VoiceEvent(type="state", state="listening"))

    # 3. Receive loop: mic frames in, control messages handled.
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if (data := msg.get("bytes")) is not None:
                await pipeline.on_audio_frame(data)
            elif (text := msg.get("text")) is not None:
                import json

                if json.loads(text).get("type") == "end":
                    await pipeline.finish()
                    await asyncio.sleep(0.1)  # let the metrics event flush
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.error("voice_ws_error", error=f"{type(exc).__name__}: {exc}", exc_info=True)
    finally:
        pump_task.cancel()
        if pipeline._respond_task:
            pipeline._respond_task.cancel()
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass
