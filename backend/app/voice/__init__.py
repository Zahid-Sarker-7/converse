"""Phase-2 voice pipeline: turn detection -> STT -> (existing LangGraph loop) -> TTS,
with barge-in and delivery metrics. The HTTP/WebSocket surface lives in `api/voice.py`."""
