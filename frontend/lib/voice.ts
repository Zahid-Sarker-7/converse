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
 * ponytail: mic capture uses ScriptProcessor (deprecated, runs on the main thread, but supported
 * everywhere). Port to an AudioWorklet if the UI thread gets busy enough to drop audio:
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
  spoken_turns: number;
}

export interface VoiceSessionConfig {
  scenario?: unknown; // a ScenarioConfig; omit to use the backend sample
  messages?: { role: string; content: string }[]; // prior chat history: text + voice are one conversation
  provider?: string | null; // chat LLM provider
  stt?: string | null; // "echo" | "groq"
  tts?: string | null; // "echo" | "kokoro"
  onEvent: (e: VoiceEvent) => void;
  onClose?: () => void; // the session ended (normal end, server error, or network drop)
}

export interface VoiceProviders {
  stt: { name: string; ready: boolean }[];
  tts: { name: string; ready: boolean }[];
}

export async function getVoiceProviders(): Promise<VoiceProviders> {
  const res = await fetch(`${API_URL}/api/voice/providers`, { cache: "no-store" });
  if (!res.ok) throw new Error(`voice providers ${res.status}`);
  return res.json();
}

/** Best ready adapter, in preference order (real before echo). */
export function pickProvider(list: { name: string; ready: boolean }[], prefs: string[]): string {
  return prefs.find((p) => list.some((x) => x.name === p && x.ready)) ?? "echo";
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
  private closed: Promise<void> = Promise.resolve();
  private mutedUntil = 0; // performance.now() deadline; see muteFor()

  constructor(private cfg: VoiceSessionConfig) {}

  async start(): Promise<void> {
    // Mic first: if permission is denied we fail before opening a server session.
    // Echo cancellation keeps the agent's own voice (from speakers) out of the mic, which would
    // otherwise trigger barge-in on itself. Headphones are still the most reliable.
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

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
      this.ws!.onerror = () => {
        this.stream?.getTracks().forEach((t) => t.stop());
        reject(new Error("Voice connection failed — is the backend running?"));
      };
    });
    this.closed = new Promise<void>((resolve) => {
      this.ws!.onclose = () => {
        this.teardownAudio();
        this.cfg.onClose?.();
        resolve();
      };
    });

    // Handshake: send the session config first.
    this.ws.send(
      JSON.stringify({
        scenario: this.cfg.scenario,
        messages: this.cfg.messages ?? [],
        provider: this.cfg.provider,
        stt: this.cfg.stt,
        tts: this.cfg.tts,
      }),
    );

    // Mic capture -> PCM16@16k -> WS.
    this.ctx = new AudioContext();
    this.playhead = this.ctx.currentTime;
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(2048, 1, 1);
    this.node.onaudioprocess = (e) => {
      if (this.ws?.readyState !== WebSocket.OPEN) return;
      const pcm = toPcm16(e.inputBuffer.getChannelData(0), this.ctx!.sampleRate, MIC_RATE);
      // While muted, send silence of the same length: the server's turn clocks keep advancing.
      this.ws.send(performance.now() < this.mutedUntil ? new ArrayBuffer(pcm.byteLength) : pcm);
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

  /** Silence the mic for `ms` — call on each keystroke: keyboard clicks reaching the mic were
   *  becoming phantom turns (Whisper transcribes non-speech as "Thank you."). */
  muteFor(ms: number): void {
    this.mutedUntil = performance.now() + ms;
  }

  /** A typed turn mid-session: the agent answers it (spoken) with the full conversation. */
  sendText(text: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify({ type: "text", text }));
  }

  /** End the session: the server replies with delivery metrics, then closes the socket. */
  async stop(): Promise<void> {
    this.node?.disconnect(); // stop sending mic audio
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "end" }));
      // Wait for the server's close (metrics arrive first); don't hang forever if it never comes.
      await Promise.race([this.closed, new Promise((r) => setTimeout(r, 5000))]);
    }
    this.ws?.close();
    this.teardownAudio();
  }

  private teardownAudio(): void {
    this.flushPlayback();
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    if (this.ctx && this.ctx.state !== "closed") void this.ctx.close();
  }
}
