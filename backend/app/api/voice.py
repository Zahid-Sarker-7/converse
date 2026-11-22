"""Voice routes — the Phase-2 pipeline over a WebSocket.

  WS  /api/voice/session   bidirectional audio: mic frames up, TTS audio + events down
  GET /api/voice/providers which STT/TTS a session can use (for the UI / health)

The route owns the security + handshake; the session itself runs on Pipecat (`voice/bot.py`).

Why WebSocket (not SSE like /api/chat): voice is two-way and interruptible — the client streams
mic audio UP while the server streams speech DOWN, and barge-in needs both at once. SSE is
one-way, so it can't carry the mic.

Wire protocol:
  client -> server : first a TEXT frame = JSON config {scenario?, messages? (prior chat history),
                     provider?, stt?, tts?}; then BINARY frames = raw PCM16 @ 16 kHz mic audio,
                     TEXT {"type":"text","text":...} = a typed turn (no STT), and
                     TEXT {"type":"end"} to end the session (-> delivery metrics).
  server -> client : TEXT frames = JSON `VoiceEvent`; BINARY frames = PCM16 @ 24 kHz to play.
"""

from __future__ import annotations

import contextlib
import re

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect

from ..agents.schemas import SAMPLE_SCENARIO, ScenarioConfig
from ..core.model_port import ChatMessage
from ..logging import get_logger
from ..voice.bot import run_session
from ..voice.schemas import VoiceEvent

router = APIRouter(prefix="/voice")
_LOCALHOST = r"http://(localhost|127\.0\.0\.1):\d+"  # same rule as the CORS middleware
log = get_logger("voice")


@router.get("/providers")
async def providers(request: Request) -> dict:
    available: dict[str, list[str]] = request.app.state.voice_providers
    return {kind: [{"name": n, "ready": True} for n in names] for kind, names in available.items()}


class _HandshakeError(ValueError):
    pass


def _pick(available: list[str], name: str | None, default: str) -> str:
    chosen = name or default
    if chosen not in available:
        raise _HandshakeError(f"'{chosen}' is not configured. Available: {available}")
    return chosen


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

    # Handshake: the first text frame configures the session (before Pipecat reads the socket).
    available = app.state.voice_providers
    try:
        cfg = await ws.receive_json()
        if not isinstance(cfg, dict):
            raise _HandshakeError("config must be a JSON object")
        history = [ChatMessage.model_validate(m) for m in cfg.get("messages") or []]
        scenario = (
            ScenarioConfig.model_validate(cfg["scenario"])
            if cfg.get("scenario")
            else SAMPLE_SCENARIO
        )
        chat_port = app.state.registry.get(cfg.get("provider") or settings.default_provider)
        if chat_port is None:
            raise _HandshakeError(f"'{cfg.get('provider')}' is not configured")
        stt = _pick(available["stt"], cfg.get("stt"), settings.voice_stt_provider)
        tts = _pick(available["tts"], cfg.get("tts"), settings.voice_tts_provider)
    except WebSocketDisconnect:
        return
    except (ValueError, KeyError) as exc:  # bad JSON, invalid scenario, unknown provider
        await ws.send_text(
            VoiceEvent(type="error", message=f"bad handshake: {exc}").model_dump_json()
        )
        await ws.close(code=1008)
        return

    try:
        await run_session(
            ws,
            settings=settings,
            scenario=scenario,
            history=history,
            chat_port=chat_port,
            stt_name=stt,
            tts_name=tts,
        )
    except Exception as exc:  # noqa: BLE001 — one broken session must not take the worker down
        log.error("voice_session_error", error=f"{type(exc).__name__}: {exc}", exc_info=True)
    finally:
        with contextlib.suppress(Exception):
            await ws.close()
