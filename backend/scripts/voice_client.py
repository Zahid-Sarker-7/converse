"""End-to-end voice test client — a scripted "browser" for WS /api/voice/session.

Streams a 16 kHz mono PCM16 WAV up in real time (like a mic), prints every server event,
saves the agent's speech to a WAV you can play, then sends "end" and prints delivery metrics.

    make backend                                   # terminal 1 (watch the step logs)
    say -o /tmp/me.wav --data-format=LEI16@16000 "Um, so I led the migration. [[slnc 1500]] \
        It was, like, really hard, you know."
    .venv/bin/python scripts/voice_client.py /tmp/me.wav --stt groq     # terminal 2
    afplay /tmp/agent.wav                                                 # hear the reply
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave

from websockets.asyncio.client import connect

FRAME_BYTES = 1364  # ~43 ms @ 16 kHz, what the browser sends
FRAME_S = FRAME_BYTES / 2 / 16_000


async def main(args: argparse.Namespace) -> None:
    with wave.open(args.wav) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16_000, 1, 2), (
            "need 16 kHz mono PCM16 — use: say -o x.wav --data-format=LEI16@16000 '...'"
        )
        pcm = w.readframes(w.getnframes())

    agent_audio = bytearray()
    async with connect(args.url) as ws:
        await ws.send(json.dumps({"provider": args.provider, "stt": args.stt, "tts": args.tts}))

        async def receive() -> None:
            async for msg in ws:
                if isinstance(msg, bytes):
                    agent_audio.extend(msg)
                    continue
                e = json.loads(msg)
                body = {k: v for k, v in e.items() if v is not None and k != "type"}
                print(f"  ◄ {e['type']:12} {body}")

        rx = asyncio.create_task(receive())
        # A real mic never stops sending: after the clip, keep streaming silence so the turn
        # detector's clocks keep advancing while the last reply (STT → LLM → TTS) runs.
        pcm += b"\0\0" * int(16_000 * args.wait)
        for i in range(0, len(pcm), FRAME_BYTES):  # real-time pacing, like a mic
            await ws.send(pcm[i : i + FRAME_BYTES])
            await asyncio.sleep(FRAME_S)
        await ws.send(json.dumps({"type": "end"}))
        await asyncio.sleep(0.5)
        rx.cancel()

    with wave.open(args.out, "wb") as w:  # agent speech is PCM16 @ 24 kHz
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24_000)
        w.writeframes(agent_audio)
    print(f"\nagent audio: {len(agent_audio) / 48_000:.1f}s → {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--url", default="ws://localhost:8000/api/voice/session")
    ap.add_argument("--provider", default=None, help="chat LLM (default: DEFAULT_PROVIDER)")
    ap.add_argument("--stt", default=None, help="echo | groq")
    ap.add_argument("--tts", default=None, help="echo | kokoro")
    ap.add_argument("--wait", type=float, default=6.0, help="silence (s) streamed after the clip")
    ap.add_argument("--out", default="/tmp/agent.wav")
    asyncio.run(main(ap.parse_args()))
