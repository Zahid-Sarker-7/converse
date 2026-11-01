"""The voice WebSocket route: origin policy, handshake validation, clean session end."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.voice import turn as turn_mod
from app.voice.turn import TurnDetector

ECHO = {"provider": "echo", "stt": "echo", "tts": "echo"}


def _events(ws) -> list[dict]:
    out = []
    while True:
        try:
            msg = ws.receive()
        except WebSocketDisconnect:
            return out
        if msg["type"] == "websocket.close":
            return out
        if msg.get("text"):
            out.append(json.loads(msg["text"]))


def test_foreign_browser_origin_is_rejected():
    with TestClient(app) as c, pytest.raises(WebSocketDisconnect) as exc:
        with c.websocket_connect("/api/voice/session", headers={"origin": "https://evil.example"}):
            pass
    assert exc.value.code == 1008


def test_bad_handshake_gets_an_error_then_close():
    with TestClient(app) as c, c.websocket_connect("/api/voice/session") as ws:
        ws.send_text(json.dumps({**ECHO, "stt": "nope"}))
        events = _events(ws)
    assert events[0]["type"] == "error" and "nope" in events[0]["message"]


def test_end_delivers_metrics_before_close():
    headers = {"origin": "http://localhost:3000"}
    with TestClient(app) as c, c.websocket_connect("/api/voice/session", headers=headers) as ws:
        ws.send_text(json.dumps(ECHO))
        ws.send_text("not json")  # malformed control frame: ignored, session survives
        ws.send_text(json.dumps({"type": "end"}))
        types = [e["type"] for e in _events(ws)]
    assert types[-1] == "metrics"


def test_turn_is_capped_at_max_length():
    td = TurnDetector(max_turn_ms=320)
    td._speech_prob = lambda chunk: 1.0  # endless "speech" (e.g. steady noise)
    loud = b"\x00\x40" * turn_mod.CHUNK_SAMPLES
    kinds = [e.kind for _ in range(12) if (e := td.feed(loud))]
    assert kinds[:2] == ["speech_started", "speech_ended"]  # forced end at 10 chunks = 320ms
