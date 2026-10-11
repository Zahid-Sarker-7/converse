"use client";

import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { RefreshCw, Sparkles, Trophy, X, Zap } from "lucide-react";
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

  const endRef = useRef<HTMLDivElement>(null);
  const offline = !health;

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
            {turns.length > 0 && (
              <Button
                variant="subtle"
                size="sm"
                onClick={() => {
                  setTurns([]);
                  setTyping(null);
                  setTurnError(null);
                }}
              >
                Reset
              </Button>
            )}
            <Button
              onClick={scoreMe}
              disabled={!hasReplies || scoring || offline}
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
        {scoreError && (
          <Banner tone="danger">
            {scoreError}
            <span className="mt-1 block text-xs text-muted">
              Scoring needs structured output — Groq and Gemini support it, Echo
              doesn&apos;t. Pick Groq or Gemini as the provider.
            </span>
          </Banner>
        )}

        <div className="flex items-end gap-2">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Your turn…  (Enter to send)"
            rows={2}
            disabled={offline}
          />
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
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
