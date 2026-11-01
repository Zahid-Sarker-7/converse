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
import contextlib
import json
import re

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from ..agents.schemas import SAMPLE_SCENARIO, ScenarioConfig
from ..logging import get_logger
from ..voice.pipeline import VoicePipeline
from ..voice.schemas import VoiceEvent

router = APIRouter(prefix="/voice")
_LOCALHOST = r"http://(localhost|127\.0\.0\.1):\d+"  # same rule as the CORS middleware
log = get_logger("voice")


@router.get("/providers")
async def providers(request: Request) -> dict:
    stt_registry: dict = request.app.state.stt_registry
    tts_registry: dict = request.app.state.tts_registry
    return {
        "stt": [{"name": n, "ready": p.ready} for n, p in stt_registry.items()],
        "tts": [{"name": n, "ready": p.ready} for n, p in tts_registry.items()],
    }


class _HandshakeError(ValueError):
    pass


def _resolve(registry: dict, name: str | None, default: str):
    port = registry.get(name or default)
    if port is None:
        raise _HandshakeError(f"'{name}' is not configured. Available: {list(registry)}")
    return port


def _origin_allowed(ws: WebSocket, settings) -> bool:
    """CORS doesn't apply to WebSockets: without this, any site could open a session from a
    visitor's browser and spend our provider quota (cross-site WebSocket hijacking). Non-browser
    clients send no Origin and are allowed (they can't ride a victim's browser)."""
    origin = ws.headers.get("origin")
    if origin is None or origin in settings.cors_origins:
        return True
    return settings.cors_allow_localhost and bool(re.fullmatch(_LOCALHOST, origin))


@router.websocket("/session")
async def session(ws: WebSocket) -> None:
    app = ws.app
    settings = app.state.settings
    if not _origin_allowed(ws, settings):
        await ws.close(code=1008)  # policy violation
        return
    await ws.accept()

    # 1. Handshake: the first text frame configures the session.
    try:
        cfg = await ws.receive_json()
        if not isinstance(cfg, dict):
            raise _HandshakeError("config must be a JSON object")
        scenario = (
            ScenarioConfig.model_validate(cfg["scenario"])
            if cfg.get("scenario")
            else SAMPLE_SCENARIO
        )
        chat_port = _resolve(app.state.registry, cfg.get("provider"), settings.default_provider)
        stt = _resolve(app.state.stt_registry, cfg.get("stt"), settings.voice_stt_provider)
        tts = _resolve(app.state.tts_registry, cfg.get("tts"), settings.voice_tts_provider)
    except WebSocketDisconnect:
        return
    except (ValueError, KeyError) as exc:  # bad JSON, invalid scenario, unknown provider
        await ws.send_text(
            VoiceEvent(type="error", message=f"bad handshake: {exc}").model_dump_json()
        )
        await ws.close(code=1008)
        return

    pipeline = VoicePipeline(
        scenario=scenario, chat_port=chat_port, stt=stt, tts=tts, tts_voice=settings.kokoro_voice
    )

    # 2. Pump: the ONLY writer to the socket. A ("close", None) item ends it after everything
    #    queued before it (e.g. the final metrics) has been sent.
    async def pump() -> None:
        while True:
            kind, payload = await pipeline.out.get()
            if kind == "close":
                return
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
            elif (text := msg.get("text")) is not None and _is_end(text):
                await pipeline.finish()
                await pipeline.out.put(("close", None))
                await asyncio.wait_for(pump_task, timeout=5)  # metrics delivered, then close
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.error("voice_ws_error", error=f"{type(exc).__name__}: {exc}", exc_info=True)
    finally:
        pump_task.cancel()
        if pipeline._respond_task:
            pipeline._respond_task.cancel()
        with contextlib.suppress(Exception):
            await ws.close()


def _is_end(text: str) -> bool:
    """Control frames are JSON; anything malformed is ignored rather than ending the session."""
    try:
        msg = json.loads(text)
    except ValueError:
        return False
    return isinstance(msg, dict) and msg.get("type") == "end"
