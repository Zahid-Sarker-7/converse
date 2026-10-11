export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface HealthProvider {
  name: string;
  ready: boolean;
}

export interface Health {
  status: string;
  app: string;
  version: string;
  environment: string;
  default_provider: string;
  providers: HealthProvider[];
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${API_URL}/health`, { cache: "no-store" });
  if (!res.ok) throw new Error(`health ${res.status}`);
  return res.json();
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

/**
 * POST /api/chat and consume the SSE stream, calling onDelta for each text chunk.
 * The backend emits `data: {json}\n\n` lines with {delta}, {done} or {error}.
 */
export async function streamChat(
  messages: ChatMessage[],
  provider: string | null,
  onDelta: (text: string) => void,
): Promise<void> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, provider }),
  });
  if (!res.ok || !res.body) throw new Error(`chat ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() ?? ""; // keep the trailing partial event
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith("data:")) continue;
      const payload = JSON.parse(line.slice(5).trim());
      if (payload.error) throw new Error(payload.error);
      if (payload.delta) onDelta(payload.delta);
      // payload.done -> stream finished
    }
  }
}

// ---------------------------------------------------------------------------
// Role-play — types mirror backend/app/agents/schemas.py
// ---------------------------------------------------------------------------

export interface ScenarioConfig {
  title: string;
  persona: string; // who the Role-Player IS
  objective: string; // what the LEARNER is trying to do
  difficulty: string; // easy | medium | hard
  curveballs: string[]; // objections the Director can inject
  rubric: string[]; // criteria names the Evaluator scores
}

export interface CriterionScore {
  criterion: string;
  score: number; // 1–5
  evidence: string; // verbatim quote from the learner's turns
  rationale: string; // one sentence: why this score
}

export interface Scorecard {
  criteria: CriterionScore[];
  overall: number; // 1–5
  summary: string;
  focus_next: string; // the single thing to work on next
}

export interface TurnResponse {
  reply: string;
  director_note: string | null;
  turn_count: number;
}

/** Best-effort friendly message from a failed response — FastAPI puts it in `detail`. */
async function errorDetail(res: Response, label: string): Promise<string> {
  try {
    const body = await res.json();
    if (body?.detail) return `${label} ${res.status}: ${body.detail}`;
  } catch {
    // non-JSON body — fall through
  }
  return `${label} ${res.status}`;
}

/** GET /api/roleplay/sample → a ready-to-use ScenarioConfig to prefill the studio. */
export async function getSampleScenario(): Promise<ScenarioConfig> {
  const res = await fetch(`${API_URL}/api/roleplay/sample`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`sample ${res.status}`);
  return res.json();
}

/** POST /api/roleplay/turn → one learner turn, Role-Player reply (+ director_note). */
export async function roleplayTurn(
  scenario: ScenarioConfig,
  messages: ChatMessage[],
  provider: string | null,
): Promise<TurnResponse> {
  const res = await fetch(`${API_URL}/api/roleplay/turn`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario, messages, provider }),
  });
  if (!res.ok) throw new Error(await errorDetail(res, "turn"));
  return res.json();
}

/**
 * POST /api/roleplay/score → grade the transcript into a Scorecard.
 * Prefer a structured-output-capable provider (Gemini) — pass `provider` through.
 */
export async function roleplayScore(
  scenario: ScenarioConfig,
  messages: ChatMessage[],
  provider: string | null,
): Promise<Scorecard> {
  const res = await fetch(`${API_URL}/api/roleplay/score`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario, messages, provider }),
  });
  if (!res.ok) throw new Error(await errorDetail(res, "score"));
  return res.json();
}
