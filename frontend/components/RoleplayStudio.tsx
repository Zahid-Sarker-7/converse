"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { Mic, MicOff, RefreshCw, Sparkles, Trophy, X, Zap } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  getSampleScenario,
  roleplayScore,
  roleplayTurn,
  type ChatMessage,
  type Health,
  type Scorecard as ScorecardData,
  type ScenarioConfig,
} from "@/lib/api";
import {
  getVoiceProviders,
  pickProvider,
  VoiceSession,
  type DeliveryMetrics,
  type VoiceEvent,
} from "@/lib/voice";
import { Scorecard } from "./Scorecard";
import {
  Banner,
  Button,
  Card,
  Chip,
  Field,
  Input,
  Select,
  Skeleton,
  Textarea,
} from "./ui";

type Turn = ChatMessage & { note?: string | null };
type VoiceState = "off" | "connecting" | "listening" | "thinking" | "speaking" | "ending";

const VOICE_LABEL: Record<VoiceState, string> = {
  off: "",
  connecting: "Connecting…",
  listening: "Listening — speak any time",
  thinking: "Thinking…",
  speaking: "Speaking — talk over it to interrupt",
  ending: "Wrapping up…",
};

function Stat({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="rounded-xl border border-border bg-white/5 px-3 py-2">
      <div className="text-xs text-muted">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      <div className="text-[11px] text-muted">{sub}</div>
    </div>
  );
}

/** How the learner spoke (voice sessions only) — benchmarks from interview/sales corpora. */
function DeliveryCard({ m }: { m: DeliveryMetrics }) {
  return (
    <Card className="p-4">
      <div className="mb-3 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">Delivery</h3>
        <span className="text-xs text-muted">
          {m.spoken_turns} spoken turn{m.spoken_turns === 1 ? "" : "s"} in the last voice session ·
          typed turns not counted
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Pace" value={`${Math.round(m.wpm)} wpm`} sub="target 140–160" />
        <Stat
          label="Fillers per minute"
          value={m.fillers_per_min.toFixed(1)}
          sub={`${m.filler_count} um/uh/like in total · target < 8`}
        />
        <Stat
          label="You vs AI talk"
          value={`${Math.round(m.talk_ratio * 100)}%`}
          sub="target ~43%"
        />
        <Stat
          label="Longest pause"
          value={`${m.longest_pause_s.toFixed(1)}s`}
          sub="target < 2s"
        />
      </div>
    </Card>
  );
}

const BLANK: ScenarioConfig = {
  title: "",
  persona: "",
  objective: "",
  difficulty: "medium",
  curveballs: [],
  rubric: [],
};

function TypingBubble({
  content,
  onDone,
}: {
  content: string;
  onDone: () => void;
}) {
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(reduce ? content : "");
  const done = useRef(false);

  useEffect(() => {
    if (reduce) {
      if (!done.current) {
        done.current = true;
        onDone();
      }
      return;
    }
    let i = 0;
    const id = setInterval(() => {
      i += 2;
      setShown(content.slice(0, i));
      if (i >= content.length) {
        clearInterval(id);
        if (!done.current) {
          done.current = true;
          onDone();
        }
      }
    }, 16);
    return () => clearInterval(id);
    // run once on mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="max-w-[85%] self-start rounded-2xl rounded-tl-sm border border-border bg-white/5 px-4 py-2.5 text-sm leading-relaxed">
      <span className={reduce ? "" : "cursor-blink"}>{shown}</span>
    </div>
  );
}

export function RoleplayStudio({
  health,
  provider,
}: {
  health: Health | null;
  provider: string;
}) {
  const [scenario, setScenario] = useState<ScenarioConfig | null>(null);
  const [curveballsText, setCurveballsText] = useState("");
  const [rubricText, setRubricText] = useState("");
  const [loadingSample, setLoadingSample] = useState(true);
  const [sampleError, setSampleError] = useState<string | null>(null);

  const [turns, setTurns] = useState<Turn[]>([]);
  const [typing, setTyping] = useState<{ content: string; note?: string | null } | null>(
    null,
  );
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [turnError, setTurnError] = useState<string | null>(null);

  const [scoring, setScoring] = useState(false);
  const [scoreError, setScoreError] = useState<string | null>(null);
  const [card, setCard] = useState<ScorecardData | null>(null);

  const [voiceState, setVoiceState] = useState<VoiceState>("off");
  const [voiceInfo, setVoiceInfo] = useState("");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<DeliveryMetrics | null>(null);
  const voiceRef = useRef<VoiceSession | null>(null);
  const voiceOn = voiceState !== "off";

  const endRef = useRef<HTMLDivElement>(null);
  const offline = !health;

  // Leaving the page mid-session must release the mic and the server session.
  useEffect(() => () => void voiceRef.current?.stop(), []);

  // Mirrors the server: a user line the agent never answered (its reply was interrupted) is the
  // same answer continuing, so it joins the previous user bubble instead of fragmenting.
  function addTurn(role: Turn["role"], content: string) {
    setTurns((prev) => {
      const last = prev[prev.length - 1];
      if (role === "user" && last?.role === "user")
        return [...prev.slice(0, -1), { ...last, content: `${last.content} ${content}` }];
      return [...prev, { role, content }];
    });
  }

  function onVoiceEvent(e: VoiceEvent) {
    if (e.type === "state" && e.state) setVoiceState(e.state);
    else if (e.type === "transcript" && e.role && e.text) addTurn(e.role, e.text); else if (e.type === "metrics" && e.metrics) setMetrics(e.metrics);
    else if (e.type === "error") setVoiceError(e.message ?? "voice error");
  }

  async function startVoice() {
    if (voiceOn || offline) return;
    setVoiceError(null);
    setMetrics(null);
    setVoiceState("connecting");
    // A reply mid-typewriter isn't in `turns` yet (state updates are async): include it explicitly.
    const history = typing ? [...turns, { role: "assistant" as const, content: typing.content }] : turns;
    commitTyping();
    try {
      const vp = await getVoiceProviders();
      const stt = pickProvider(vp.stt, ["groq"]);
      const tts = pickProvider(vp.tts, ["kokoro"]);
      setVoiceInfo(`${stt} → ${provider || "default"} → ${tts}`);
      const session = new VoiceSession({
        scenario: buildScenario(),
        // Text and voice are one conversation: the voice session continues this history.
        messages: history.map((t) => ({ role: t.role, content: t.content })),
        provider: provider || null,
        stt,
        tts,
        onEvent: onVoiceEvent,
        onClose: () => {
          voiceRef.current = null;
          setVoiceState("off");
        },
      });
      voiceRef.current = session;
      await session.start();
    } catch (e) {
      voiceRef.current = null;
      setVoiceState("off");
      setVoiceError(e instanceof Error ? e.message : String(e));
    }
  }

  async function stopVoice() {
    setVoiceState("ending");
    await voiceRef.current?.stop();
    voiceRef.current = null;
    setVoiceState("off");
  }

  function applyScenario(s: ScenarioConfig) {
    setScenario(s);
    setCurveballsText(s.curveballs.join("\n"));
    setRubricText(s.rubric.join("\n"));
  }

  async function loadSample() {
    setLoadingSample(true);
    setSampleError(null);
    try {
      applyScenario(await getSampleScenario());
    } catch (e) {
      setSampleError(e instanceof Error ? e.message : String(e));
      if (!scenario) applyScenario(BLANK);
    } finally {
      setLoadingSample(false);
    }
  }

  useEffect(() => {
    loadSample();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns, typing]);

  function buildScenario(): ScenarioConfig {
    const s = scenario ?? BLANK;
    const lines = (t: string) =>
      t
        .split("\n")
        .map((x) => x.trim())
        .filter(Boolean);
    return { ...s, curveballs: lines(curveballsText), rubric: lines(rubricText) };
  }

  async function send() {
    const text = input.trim();
    if (!text || sending || offline) return;
    if (voiceOn) {
      // Mid voice session: same conversation over the socket; the reply comes back spoken.
      setInput("");
      addTurn("user", text);
      voiceRef.current?.sendText(text);
      return;
    }
    setSending(true);
    setTurnError(null);
    setInput("");
    const nextTurns: Turn[] = [...turns, { role: "user", content: text }];
    setTurns(nextTurns);
    const apiMessages: ChatMessage[] = nextTurns.map((t) => ({
      role: t.role,
      content: t.content,
    }));
    try {
      const res = await roleplayTurn(buildScenario(), apiMessages, provider || null);
      setTyping({ content: res.reply, note: res.director_note });
    } catch (e) {
      setTurnError(e instanceof Error ? e.message : String(e));
    } finally {
      setSending(false);
    }
  }

  function commitTyping() {
    if (!typing) return;
    setTurns((prev) => [
      ...prev,
      { role: "assistant", content: typing.content, note: typing.note },
    ]);
    setTyping(null);
  }

  // Scoring needs structured output. Groq + Gemini implement it; Echo doesn't — so
  // score with the selected provider, only steering away from echo to a ready one.
  const judgeProvider =
    provider && provider !== "echo"
      ? provider
      : (health?.providers.find(
          (p) => (p.name === "gemini" || p.name === "groq") && p.ready,
        )?.name ?? provider);

  async function scoreMe() {
    if (scoring || offline) return;
    setScoring(true);
    setScoreError(null);
    try {
      const apiMessages: ChatMessage[] = turns.map((t) => ({
        role: t.role,
        content: t.content,
      }));
      setCard(
        await roleplayScore(buildScenario(), apiMessages, judgeProvider || null),
      );
    } catch (e) {
      setScoreError(e instanceof Error ? e.message : String(e));
    } finally {
      setScoring(false);
    }
  }

  const hasReplies = turns.some((t) => t.role === "assistant");

  return (
    <div className="mx-auto grid max-w-6xl gap-5 px-4 py-8 lg:grid-cols-[minmax(320px,380px)_1fr]">
      {/* ---- Scenario panel ---- */}
      <Card className="h-fit space-y-4 p-5">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Scenario</h2>
          <Button
            size="sm"
            variant="ghost"
            onClick={loadSample}
            disabled={loadingSample || offline}
          >
            <RefreshCw
              size={14}
              className={loadingSample ? "animate-spin" : ""}
            />
            Sample
          </Button>
        </div>

        {sampleError && (
          <Banner tone="warn">Couldn&apos;t load sample: {sampleError}</Banner>
        )}

        {loadingSample && !scenario ? (
          <div className="space-y-3">
            <Skeleton className="h-9" />
            <Skeleton className="h-20" />
            <Skeleton className="h-20" />
            <Skeleton className="h-9" />
          </div>
        ) : (
          <>
            <Field label="Title">
              <Input
                value={scenario?.title ?? ""}
                onChange={(e) =>
                  setScenario((s) => ({ ...(s ?? BLANK), title: e.target.value }))
                }
                placeholder="Behavioral interview…"
              />
            </Field>
            <Field label="Persona" hint="Who the AI plays">
              <Textarea
                rows={3}
                value={scenario?.persona ?? ""}
                onChange={(e) =>
                  setScenario((s) => ({ ...(s ?? BLANK), persona: e.target.value }))
                }
              />
            </Field>
            <Field label="Objective" hint="What you are trying to do">
              <Textarea
                rows={2}
                value={scenario?.objective ?? ""}
                onChange={(e) =>
                  setScenario((s) => ({
                    ...(s ?? BLANK),
                    objective: e.target.value,
                  }))
                }
              />
            </Field>
            <Field label="Difficulty">
              <Select
                value={scenario?.difficulty ?? "medium"}
                onChange={(e) =>
                  setScenario((s) => ({
                    ...(s ?? BLANK),
                    difficulty: e.target.value,
                  }))
                }
              >
                <option value="easy">easy</option>
                <option value="medium">medium</option>
                <option value="hard">hard</option>
              </Select>
            </Field>
            <Field label="Curveballs" hint="One per line — the Director injects these">
              <Textarea
                rows={3}
                value={curveballsText}
                onChange={(e) => setCurveballsText(e.target.value)}
              />
            </Field>
            <Field label="Rubric" hint="One criterion per line — the Evaluator scores these">
              <Textarea
                rows={3}
                value={rubricText}
                onChange={(e) => setRubricText(e.target.value)}
              />
            </Field>
          </>
        )}
      </Card>

      {/* ---- Chat + score ---- */}
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <h2 className="truncate text-lg font-semibold">
              {scenario?.title || "Role-play"}
            </h2>
            <p className="text-sm text-muted">
              {turns.filter((t) => t.role === "assistant").length} replies ·{" "}
              {provider || "default"}
            </p>
          </div>
          <div className="flex gap-2">
            {turns.length > 0 && !voiceOn && (
              <Button
                variant="subtle"
                size="sm"
                onClick={() => {
                  setTurns([]);
                  setTyping(null);
                  setTurnError(null);
                  setMetrics(null);
                }}
              >
                Reset
              </Button>
            )}
            <Button
              onClick={scoreMe}
              disabled={!hasReplies || scoring || offline || voiceOn}
              title={!hasReplies ? "Have a conversation first" : "Score this session"}
            >
              <Trophy size={16} />
              {scoring ? "Scoring…" : "Score me"}
            </Button>
          </div>
        </div>

        {offline && (
          <Banner tone="warn">
            Backend is offline — start the FastAPI server to role-play.
          </Banner>
        )}

        <Card className="flex min-h-[420px] flex-col gap-3 p-4">
          {turns.length === 0 && !typing && (
            <div className="m-auto max-w-sm text-center text-sm text-muted">
              <Sparkles size={22} className="mx-auto mb-2 text-accent" />
              Tweak the scenario, then say your opening line. The AI stays in
              character; hit <span className="text-text">Score me</span> when
              you&apos;re done.
            </div>
          )}

          <AnimatePresence initial={false}>
            {turns.map((t, i) => (
              <motion.div
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={
                  t.role === "user" ? "flex flex-col items-end" : "flex flex-col items-start"
                }
              >
                {t.role === "assistant" && t.note && (
                  <Chip tone="warn" className="mb-1">
                    <Zap size={11} />
                    curveball incoming
                  </Chip>
                )}
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
                    t.role === "user"
                      ? "rounded-tr-sm bg-accent text-white"
                      : "rounded-tl-sm border border-border bg-white/5"
                  }`}
                >
                  {t.content}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {typing && (
            <div className="flex flex-col items-start">
              {typing.note && (
                <Chip tone="warn" className="mb-1">
                  <Zap size={11} />
                  curveball incoming
                </Chip>
              )}
              <TypingBubble content={typing.content} onDone={commitTyping} />
            </div>
          )}

          {sending && !typing && (
            <div className="flex items-center gap-1.5 self-start px-2 text-muted">
              <span className="h-2 w-2 animate-bounce rounded-full bg-muted [animation-delay:-0.2s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-muted [animation-delay:-0.1s]" />
              <span className="h-2 w-2 animate-bounce rounded-full bg-muted" />
            </div>
          )}
          <div ref={endRef} />
        </Card>

        {turnError && <Banner tone="danger">{turnError}</Banner>}
        {voiceError && <Banner tone="danger">{voiceError}</Banner>}
        {metrics && !voiceOn && <DeliveryCard m={metrics} />}

        {voiceOn && (
          <div className="flex items-center justify-between gap-3 rounded-xl border border-accent/40 bg-accent/10 px-4 py-2.5">
            <div className="flex items-center gap-2 text-sm">
              <motion.span
                className="h-2.5 w-2.5 rounded-full bg-accent"
                animate={
                  voiceState === "listening" || voiceState === "speaking"
                    ? { scale: [1, 1.5, 1], opacity: [1, 0.5, 1] }
                    : {}
                }
                transition={{ repeat: Infinity, duration: 1.2 }}
              />
              {VOICE_LABEL[voiceState]}
            </div>
            <span className="hidden text-xs text-muted sm:block">
              {voiceInfo} · headphones recommended
            </span>
          </div>
        )}
        {scoreError && (
          <Banner tone="danger">
            {scoreError}
            {judgeProvider === "echo" && (
              <span className="mt-1 block text-xs text-muted">
                Scoring needs structured output — Groq and Gemini support it, Echo
                doesn&apos;t. Pick Groq or Gemini as the provider.
              </span>
            )}
          </Banner>
        )}

        <div className="flex items-end gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              voiceRef.current?.muteFor(1500); // keyboard clicks must not become voice turns
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder={voiceOn ? "Speak, or type here…  (Enter to send)" : "Your turn…  (Enter to send)"}
            rows={2}
            disabled={offline}
          />
          <Button
            variant={voiceOn ? "danger" : "ghost"}
            onClick={voiceOn ? stopVoice : startVoice}
            disabled={offline || voiceState === "connecting" || voiceState === "ending"}
            className="h-[46px]"
            title={voiceOn ? "End the voice session and get delivery metrics" : "Talk instead of typing"}
          >
            {voiceOn ? <MicOff size={16} /> : <Mic size={16} />}
            {voiceOn ? "End" : "Talk"}
          </Button>
          <Button
            onClick={send}
            disabled={sending || offline}
            className="h-[46px]"
          >
            Send
          </Button>
        </div>
      </div>

      {/* ---- Scorecard reveal overlay ---- */}
      <AnimatePresence>
        {card && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8"
            onClick={() => setCard(null)}
          >
            <motion.div
              initial={{ opacity: 0, y: 24, scale: 0.98 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 24, scale: 0.98 }}
              transition={{ type: "spring", stiffness: 260, damping: 26 }}
              className="w-full max-w-2xl"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-2xl font-bold">Your scorecard</h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setCard(null)}
                  aria-label="Close scorecard"
                >
                  <X size={16} />
                </Button>
              </div>
              <Scorecard card={card} />
              {metrics && (
                <div className="mt-4">
                  <DeliveryCard m={metrics} />
                </div>
              )}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
