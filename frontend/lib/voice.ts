/**
 * Voice session client — the browser half of the Phase-2 pipeline.
 *
 * Opens the WebSocket at /api/voice/session, captures the mic, downsamples to the backend's
 * 16 kHz PCM16, streams it up, and plays the 24 kHz PCM16 speech that comes back. It mirrors
 * the backend wire protocol in `api/voice.py`:
 *   up   : first a JSON config frame, then binary PCM16@16k frames, then {"type":"end"}
 *   down : JSON VoiceEvent frames (text) + binary PCM16@24k audio frames
 *
 * This is a working scaffold. The interesting backend craft (turn detection, barge-in,
 * delivery metrics) lives server-side as your three TODOs; this client just moves audio.
 *
 * NOTE: mic capture here uses a ScriptProcessor for brevity. For production, port the capture
 * to an AudioWorklet (off the main thread) — see
 * https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet
 */

import { API_URL } from "./api";

const WS_URL = API_URL.replace(/^http/, "ws") + "/api/voice/session";
const MIC_RATE = 16_000; // what the backend STT/VAD expect
const TTS_RATE = 24_000; // what Kokoro / the backend sends down

export interface VoiceEvent {
  type: "state" | "transcript" | "partial_transcript" | "metrics" | "error" | "interrupt";
  state?: "listening" | "thinking" | "speaking";
  role?: "user" | "assistant";
  text?: string;
  message?: string;
  metrics?: DeliveryMetrics;
}

export interface DeliveryMetrics {
  wpm: number;
  filler_count: number;
  fillers_per_min: number;
  talk_ratio: number;
  longest_pause_s: number;
  speaking_time_s: number;
}

export interface VoiceSessionConfig {
  scenario?: unknown; // a ScenarioConfig; omit to use the backend sample
  provider?: string | null; // chat LLM provider
  stt?: string | null; // "echo" | "groq"
  tts?: string | null; // "echo" | "kokoro"
  onEvent: (e: VoiceEvent) => void;
}

/** Float32 [-1,1] -> Int16 PCM, with naive linear resampling from `inRate` to `outRate`. */
function toPcm16(input: Float32Array, inRate: number, outRate: number): ArrayBuffer {
  const ratio = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Int16Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const s = Math.max(-1, Math.min(1, input[Math.floor(i * ratio)]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out.buffer;
}

export class VoiceSession {
  private ws?: WebSocket;
  private ctx?: AudioContext;
  private stream?: MediaStream;
  private node?: ScriptProcessorNode;
  private playhead = 0; // scheduled-audio cursor for gap-free playback
  private scheduled = new Set<AudioBufferSourceNode>(); // queued speech, so barge-in can stop it

  constructor(private cfg: VoiceSessionConfig) {}

  async start(): Promise<void> {
    this.ws = new WebSocket(WS_URL);
    this.ws.binaryType = "arraybuffer";

    this.ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const e = JSON.parse(ev.data) as VoiceEvent;
        if (e.type === "interrupt") this.flushPlayback(); // barge-in: silence already-sent audio
        this.cfg.onEvent(e);
      } else {
        this.playPcm(new Int16Array(ev.data as ArrayBuffer));
      }
    };

    await new Promise<void>((resolve, reject) => {
      this.ws!.onopen = () => resolve();
      this.ws!.onerror = () => reject(new Error("voice websocket failed"));
    });

    // Handshake: send the session config first.
    this.ws.send(
      JSON.stringify({
        scenario: this.cfg.scenario,
        provider: this.cfg.provider,
        stt: this.cfg.stt,
        tts: this.cfg.tts,
      }),
    );

    // Mic capture -> PCM16@16k -> WS.
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    this.ctx = new AudioContext();
    this.playhead = this.ctx.currentTime;
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(2048, 1, 1);
    this.node.onaudioprocess = (e) => {
      if (this.ws?.readyState !== WebSocket.OPEN) return;
      const pcm = toPcm16(e.inputBuffer.getChannelData(0), this.ctx!.sampleRate, MIC_RATE);
      this.ws.send(pcm);
    };
    src.connect(this.node);
    this.node.connect(this.ctx.destination);
  }

  /** Queue a chunk of 24 kHz PCM16 speech for gap-free playback. */
  private playPcm(pcm: Int16Array): void {
    if (!this.ctx) return;
    const buf = this.ctx.createBuffer(1, pcm.length, TTS_RATE);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 0x8000;
    const node = this.ctx.createBufferSource();
    node.buffer = buf;
    node.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    this.playhead = Math.max(this.playhead, now);
    node.start(this.playhead);
    this.playhead += buf.duration;
    this.scheduled.add(node);
    node.onended = () => this.scheduled.delete(node);
  }

  /** Stop everything queued for playback (the server already dropped what it hadn't sent). */
  private flushPlayback(): void {
    this.scheduled.forEach((n) => n.stop());
    this.scheduled.clear();
    this.playhead = this.ctx?.currentTime ?? 0;
  }

  /** End the turn/session cleanly (-> backend computes delivery metrics). */
  async stop(): Promise<void> {
    try {
      this.ws?.send(JSON.stringify({ type: "end" }));
    } catch {
      /* already closing */
    }
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    await this.ctx?.close();
    this.ws?.close();
  }
}
