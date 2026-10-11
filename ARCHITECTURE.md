# Converse — Architecture & Requirements (plain-English, tech-agnostic)

> **Read this to understand *what* we're building and *why the pieces exist* — before any code.**
> It is deliberately technology-agnostic: it names **options + tradeoffs** for each layer, then marks
> our current pick. Concepts are explained from scratch (including "do we even need agents?").
>
> Companion docs: `README.md` (how to run it) · `../ventures/11-milestone-project.md` (product/business
> spec) · `../ventures/roadmap/project-plan.md` (phase-by-phase build) · `../ventures/roadmap/STATE.md`
> (where we are right now).

---

## 1. What are we building? (one paragraph)

A **reusable AI engine** that lets a person have a **real-time spoken conversation with an AI
character** (an interviewer, an examiner, a tough customer), and then get **scored and coached** on
how they did. The same engine ships as different products by swapping a "**product pack**"
(who the character is, what rubric scores them, the UI). v1 = a **spoken-English + interview coach**
for South Asia; the B2B "role-play trainer" is the same engine with a different pack.

The point of the project is **breadth**: it forces you to build every layer of a modern AI system —
model routing, orchestration, voice, vision, memory, evals, guardrails, observability, and a SaaS
shell — each as a **swappable part**.

---

## 2. Requirements

### 2.1 Functional (what it must *do*)

| # | Requirement | Plain meaning |
|---|-------------|---------------|
| F1 | Text conversation with an in-character AI | You type, the "interviewer" replies in character, across many turns |
| F2 | Scenario setup | A scenario defines the persona, goal, difficulty, and curveballs |
| F3 | Scoring / feedback | After a session, produce a **scorecard**: per-criterion score + a **quote** as evidence + rationale |
| F4 | Voice conversation *(Phase 2)* | Speak instead of type; the AI talks back; you can interrupt it (barge-in) |
| F5 | Vision *(Phase 3, optional)* | Show a passage/slide the learner must reference |
| F6 | Memory *(Phase 5)* | Remembers your past sessions, weak spots, progress over time |
| F7 | Provider-swappable | Change the underlying LLM by config, not code |
| F8 | Multi-product | A new product = a new "pack", not a new codebase |
| F9 | SaaS shell *(Phase 8)* | Accounts, billing, dashboards |

### 2.2 Non-functional (how *well* it must behave)

| Property | Target | Why it matters here |
|----------|--------|---------------------|
| **Latency** | Text < ~2s to first token; **voice < ~800ms** round-trip feel | A laggy "conversation" breaks the illusion; voice is the hard one |
| **Cost** | **$0 / free-tier** for dev + demo | Solo, moonlighting, Bangladesh — no card-required services |
| **Statelessness** | Backend holds no per-user state in memory | Lets us scale horizontally later (see system-design notes) |
| **Swappability** | Every external dependency behind a port | Swap model/voice/DB without rewriting core logic |
| **Reliability of scoring** | Structured, schema-valid, stable (±1 on rerun) | A judge that returns prose or drifts is useless |
| **Observability** | Per-step traces + token/cost *(Phase 7)* | You can't improve or debug an agent you can't see |

### 2.3 Constraints (the reality box)

- **Solo, ~15 hrs/week.** Architecture must be *learnable one layer at a time*, not big-bang.
- **$0 first.** Paid providers (e.g. Claude API) are *supported* but not required to run.
- **Realtime voice ≠ serverless.** A live audio connection needs a persistent server, which free
  serverless tiers dislike — so voice stays browser-side or on a small always-on host (noted in Phase 2).

---

## 3. Clearing the core confusion: LLM call vs Workflow vs Agent vs Multi-agent

This is the most important section. These four words describe **increasing amounts of "who decides
what happens next."** The single question that separates them:

> **Who controls the flow — you (the developer, in code) or the model (deciding at runtime)?**

```
  MORE developer control ◄─────────────────────────────────► MORE model control
  ┌────────────────┬─────────────────┬──────────────────┬─────────────────────┐
  │ 1. Raw LLM call│ 2. Workflow      │ 3. Agent          │ 4. Multi-agent      │
  │ (one Q→A)      │ (fixed steps)    │ (model picks steps)│ (a team of agents)  │
  └────────────────┴─────────────────┴──────────────────┴─────────────────────┘
   "answer this"    "do A, then B,     "here's a goal +    "specialists that
                     then C"            tools — figure it   hand off / debate"
                                        out, loop till done"
```

### 3.1 Definitions in plain terms

| Term | What it is | Everyday analogy | Who drives |
|------|-----------|------------------|-----------|
| **Raw LLM call** | Send text → get text. No memory, one shot. | Asking a brilliant stranger one question who then forgets you | You (trivially) |
| **Workflow** *(a.k.a. pipeline/chain)* | **You** hard-code a fixed sequence of steps; the LLM fills each step. Predictable, repeatable. | An assembly line: station 1 → 2 → 3 | **You (in code)** |
| **Agent** | You give the model a **goal + tools**; the **model decides** which tool to call, whether to loop, and when it's done. | A smart assistant told "book my trip" who figures out the steps | **The model** |
| **Multi-agent** | Several agents, each with a **role/persona**, collaborating (handoffs, debate, shared state). | A team: a director, an actor, a judge | The model(s), coordinated |

### 3.2 The distinction that actually matters

- A **workflow** is **LLM steps wired together by *your code***. The path is known in advance. You
  can draw it as a flowchart before running it. → **predictable, debuggable, cheap.**
- An **agent** is **the LLM directing its own process** — choosing tools and looping until it decides
  it's done. The path is *not* known in advance. → **flexible, but slower, pricier, harder to test.**

> Rule of thumb (industry consensus, 2026): **use a workflow whenever the steps are known; reach for
> a true agent only when the task genuinely needs the model to decide the path** (open-ended research,
> unknown tool sequences). Most "agent" products are actually workflows — and that's a good thing.

### 3.3 So what does Converse actually need? (honest answer)

**Mostly a workflow with role-specialized prompts — *not* autonomous agents.** Look at a real turn:

```
CHAT TURN (you say something):
   your message
      │
      ▼
   [Scenario-Director]  ← picks the next curveball (rule-based today)
      │  passes a note
      ▼
   [Role-Player]        ← one LLM call, in character, replies
      │
      ▼
   reply streamed back

SCORE TURN (session ends):
   full transcript
      │
      ▼
   [Evaluator]          ← one LLM call, returns a structured Scorecard
      │
      ▼
   scorecard
```

- The **path is fixed** (Director → Role-Player → done; or Evaluator → done). *You* wired those
  edges. That makes it a **workflow**, even though we run it on a graph library.
- We call the roles "agents" because they are **distinct personas with distinct prompts** (the actor
  must never also be the judge). But none of them *chooses its own next step* — so they are **not
  agents in the autonomous sense.** They're **specialized workers on a developer-defined line.**
- True agent behavior (model picks tools, loops) is a **later, optional** upgrade (e.g. a Director
  that *invents* curveballs, or a Coach that decides when to call a "look up grammar rule" tool).

**This is why the earlier LangGraph question mattered:** for today's fixed path, the graph is
optional structure, not a necessity. We keep it for the *later* phases (memory, branching, loops),
not because Phase 1 demands it.

### 3.4 "Why not just call the LLM SDK directly and answer the user?"

You genuinely *could* for a **single Q&A turn** — and for a plain chatbot you *should*; adding a
framework there is over-engineering. But Converse is not a Q&A bot. Here's what a single raw call
**cannot** give you, and which layer solves it:

| What a plain `llm.chat(prompt)` can't do | Why we need it | Solved by |
|------------------------------------------|----------------|-----------|
| Keep the **actor** and the **judge** truly separate | Mixing "be the customer" + "grade the learner" in one prompt corrupts both (the character gets judgy; the judge gets soft) | Separate nodes/prompts (**workflow**) |
| Produce **reliable structured scores** | A scorecard must be machine-readable JSON with quotes, stable across reruns — free-text won't do | **Structured output** + validation (Evaluator) |
| Remember the conversation / past sessions | The character must stay consistent across turns; coaching needs your history | **State + Memory port** |
| Swap models / add voice / add tools / add safety later | Each is a different concern that must plug in without rewrites | **Ports & adapters** (§4) |
| See what happened when a session goes wrong | Debug + measure quality + cost | **Observability** |

**The orchestration layer is not magic — it is *plumbing* that manages multiple LLM calls, shared
state, and swappable parts.** If you only ever needed one call, you wouldn't need it.

---

## 4. The architecture (tech-agnostic)

We use **hexagonal / ports-and-adapters**: the **core logic** talks only to **ports** (interfaces);
each port has interchangeable **adapters** (implementations). This is what makes everything swappable
(requirement F7/F8).

```
              ┌──────────── PRODUCT PACK (swap to make a new product) ────────────┐
              │  personas · prompts · scoring rubric · UI theme · domain config    │
              └───────────────────────────────┬───────────────────────────────────┘
                                               │ plugs into
   Frontend (web/mobile) ── text now; voice + live transcript later
        │  HTTP/SSE now  ·  WebSocket for voice later
   ┌────▼──────────────────────────── CORE ENGINE ──────────────────────────────────┐
   │  Orchestrator  ── runs the workflow/graph: Director → Role-Player, or Evaluator  │
   │     │                                                                            │
   │     ├─ Model port ──── talk to any LLM provider        (adapters: Groq, Gemini,  │
   │     │                                                    Claude, local, echo)    │
   │     ├─ Voice port ──── speech-in / speech-out          (browser | Whisper|Piper) │  ← Phase 2
   │     ├─ Vision port ─── understand an image/screen      (a vision LLM)            │  ← Phase 3
   │     ├─ Memory port ─── short-term + long-term recall   (in-mem | Postgres+vec)   │  ← Phase 5
   │     ├─ Tool port ───── functions the model may call    (registry | MCP)          │  ← later
   │     ├─ Eval port ───── LLM-as-judge rubric scoring     (Evaluator | DeepEval)    │  ← Phase 1/4
   │     └─ Guardrail port  input/output safety             (rules | Llama Guard)     │  ← Phase 6
   │                                                                                  │
   │  Cross-cutting: Config · Logging · Observability · Persistence · Auth · Billing  │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

**How to read it:** the Orchestrator never knows *which* LLM or *which* voice engine it's using — it
only knows the **port**. Swapping Groq→Claude, or browser-voice→Whisper, is an adapter change, not a
rewrite. That single idea is the whole architecture.

---

## 5. Every layer: options & tradeoffs

For each layer: what it's for, the realistic options, the tradeoff, and **our current pick + why**.
Per **Principle 0**, re-verify the "pick" before you build that phase — this space moves monthly.

### 5.1 Orchestration (how the steps are wired)
| Option | Pros | Cons |
|--------|------|------|
| **Plain Python** (if/else, function calls) | Zero deps, fully transparent, trivial to debug | You hand-build memory/branching/loops later |
| **Graph framework** (e.g. LangGraph) | Free checkpointing, HITL, cycles, streaming; industry-standard (portfolio value) | Extra abstraction; overkill for a linear flow |
| **Role-based framework** (e.g. CrewAI) | Fastest multi-agent prototype | Higher token overhead; less control |
| **Model-driven SDK** (OpenAI Agents / Claude Agent SDK) | Simple, model drives | Less explicit control; provider-aligned |

**Pick:** graph framework (LangGraph), **because later phases** (memory, branching, loops) cash in on
it and it's a recognized skill — *not* because Phase 1 needs it. Lock-in is low (nodes are plain
functions calling our own Model port). A plain-Python escape hatch exists.

### 5.2 Model provider (the LLM itself)
| Option | Pros | Cons |
|--------|------|------|
| **Groq** (hosted open models) | Free tier, no card, very fast | Model catalog churns; quality varies by model |
| **Gemini** (free tier) | Free, strong, great structured output & vision | Free tier trains on inputs; region limits |
| **Claude** | Top quality for role-play + judging | Paid (no free tier) |
| **Local (Ollama)** | $0, private, offline | Needs a capable machine; slower |

**Pick:** provider-agnostic **Model port**; default **Groq**, with **Gemini** as the second free
provider (and the reliable one for **structured output**, used by the Evaluator). Claude is supported
for when a paid key exists. An **Echo** adapter runs the whole app at $0 with no keys.

### 5.3 Voice *(Phase 2)*
| Option | Pros | Cons |
|--------|------|------|
| **Browser Web Speech API** | $0, no server, easy | Quality/consistency varies by browser; less control |
| **Whisper (STT) + Piper/Kokoro (TTS)** self-host | High quality, licensed for commercial use | Needs a host; more setup/latency work |
| **Realtime platforms** (LiveKit/Pipecat OSS) | Handles VAD/barge-in/transport | More moving parts; a persistent server |

**Pick (planned):** start **browser Web Speech** (keeps the free demo serverless), upgrade to
Whisper+Piper/Kokoro when quality/commercial use demands it. Barge-in needs VAD + streaming.

### 5.4 Vision *(Phase 3, optional)*
| Option | Pros | Cons |
|--------|------|------|
| **Vision-capable LLM** (Gemini/Claude) | No extra infra; just send the image | Cost per image; provider-dependent |
| **Dedicated OCR/vision models** | Precise text extraction | More plumbing; another dependency |

**Pick (planned):** reuse the **Model port** with a vision-capable adapter — no new infra.

### 5.5 Memory *(Phase 5)*
| Option | Pros | Cons |
|--------|------|------|
| **In-memory / session only** | Trivial; fine for Phase 1 | Lost on restart; single instance only |
| **Postgres** (structured history) | Durable, queryable, free tiers (Supabase) | A schema to maintain |
| **Postgres + pgvector** (semantic recall) | "Remembers similar past moments" | Embeddings + retrieval to build |
| **Managed vector DB** | Scales, little ops | Another vendor; possible cost |

**Pick (planned):** **Postgres + pgvector** (short-term = conversation; long-term = learner profile,
weak spots, trends). Kept behind a **Memory port**; Phase 1 uses in-graph state only.

### 5.6 Evals / scoring *(Phase 1 core, hardened Phase 4)*
| Option | Pros | Cons |
|--------|------|------|
| **LLM-as-judge + strict schema** (ours) | Flexible, gives quotes + rationale | Judge bias/drift; must validate + calibrate |
| **Eval libraries** (DeepEval, Promptfoo) | Ready metrics, CI integration | Learning curve; opinionated |
| **Human-labeled set** | Ground truth | Slow; you must create/maintain it |

**Pick:** **LLM-as-judge** returning a schema-valid `Scorecard` (score + verbatim quote + rationale),
with **bias guards** (ignore length, score each criterion independently) and later a **calibration
set** that doubles as the **CI regression gate**.

### 5.7 Guardrails *(Phase 6)*, Observability *(Phase 7)*, Transport, Frontend, Persistence, Auth, Billing, Deploy
| Layer | Options | Our pick (planned) |
|-------|---------|--------------------|
| **Guardrails** | prompt rules · moderation API · Llama Guard | keep role-player in scope + PII/safety; start with rules + a moderation adapter |
| **Observability** | logs only · OpenTelemetry · Langfuse (OSS) | OTel spans per step + token/cost/latency; Langfuse self-host |
| **Transport** | REST/SSE (now) · WebSocket (voice) | SSE for text streaming; WebSocket added for voice |
| **Frontend** | Next.js/React · plain web | Next.js + React (Vercel free) |
| **Persistence** | SQLite · Postgres | Postgres (Supabase free) |
| **Auth** | homegrown · Supabase Auth · Clerk | multi-tenant via Supabase/managed |
| **Billing** | Stripe · Paddle | Stripe (per-seat) |
| **Deploy** | Vercel (frontend) + small always-on host (realtime) | Vercel + a small host for the voice/WS path |

---

## 6. Data model (minimum, Phase 8)

```
Org ──< User(role: admin|learner)
Org ──< Scenario(persona, objective, difficulty, rubric[])
User ──< Session(scenario_id, transcript, audio_meta)
Session ──< Score(criterion, value 1–5, evidence_quote, rationale)
User ──1 LearnerProfile(skill_trends, weaknesses)   ← Phase 5 memory
```

Phase 1 needs only `Scenario` + an in-memory `Session` transcript + a `Scorecard`. The rest arrives
with SaaS (Phase 8) and Memory (Phase 5).

---

## 7. Request lifecycles (what happens on each call)

```
TEXT CHAT TURN (exists today, Phase 1)
 user types ─► POST /api/roleplay/turn ─► Orchestrator(mode="chat")
   └─ Scenario-Director picks curveball ─► Role-Player calls Model port ─► stream reply (SSE) ─► UI

SCORE TURN (exists today, Phase 1)
 end session ─► POST /api/roleplay/score ─► Orchestrator(mode="score")
   └─ Evaluator calls Model port with schema ─► validate Scorecard ─► return JSON ─► UI renders scorecard

VOICE TURN (Phase 2)
 mic ─► STT (Voice port) ─► [same chat turn as above] ─► TTS (Voice port) ─► speaker
   (VAD detects you speaking ─► barge-in cancels TTS)
```

---

## 8. What exists today vs planned

| Layer | Status | Where |
|-------|--------|-------|
| Model port + Groq/Gemini/Echo adapters (streaming) | ✅ done (Phase 0) | `backend/app/core/` |
| FastAPI + SSE chat + health + config + logging | ✅ done | `backend/app/` |
| Next.js chat UI (provider swap, health badge) | ✅ done | `frontend/` |
| Orchestrator graph + Scenario-Director + Scorecard schema | ✅ scaffolded (Phase 1) | `backend/app/agents/` |
| Role-Player node + Evaluator node (+ structured output) | ⬜ **your TODO** (Phase 1) | `backend/app/agents/graph.py` |
| Voice / Vision / Memory / Guardrails / Observability / SaaS | ⬜ planned (Phases 2–8) | — |
| Boilerplate extraction (`create-ai-app`) | ⬜ planned (Phase 9) | — |

---

## 9. Decision summary (the "why" in one screen)

| Question | Answer |
|----------|--------|
| Do we need agents? | **Not yet.** Phase 1 is a **workflow** (fixed steps) with role-specialized prompts. True autonomous agents are a later, optional upgrade. |
| Why not just call the LLM SDK? | Fine for one Q&A. Converse needs **separated roles, structured scoring, memory, and swappable voice/vision/safety** — that's what the orchestration + ports provide. |
| Why a graph library then? | For **later** phases (memory/branching/loops) + learning value. Lock-in is low; a plain-Python version is ~15 lines. |
| Why ports/adapters everywhere? | So models, voice, DB, etc. **swap by config** and new products = new **packs**, not new code. |
| Why $0 / free-tier? | Solo, moonlighting, Bangladesh — the build must run with no paid keys (Echo/Groq/Gemini + Supabase/Vercel free). |

---

## 10. Glossary

- **Port / adapter (hexagonal):** a port is an *interface* (e.g. "talk to an LLM"); an adapter is one
  *implementation* (Groq, Gemini…). Core code depends on the port, so implementations swap freely.
- **Workflow / pipeline / chain:** LLM steps wired in a **fixed order by developer code**. Predictable.
- **Agent:** the **model** decides the next step (which tool, loop or stop). Flexible, less predictable.
- **Multi-agent:** several role-specialized agents collaborating (handoffs/debate/shared state).
- **Orchestrator:** the component that runs the workflow/graph and threads shared state between steps.
- **Structured output:** forcing the model to return schema-valid JSON (used for the scorecard).
- **LLM-as-judge / eval:** using an LLM to score outputs against an explicit rubric; calibrated + run
  in CI to catch regressions.
- **Product pack:** the swappable bundle (personas, prompts, rubric, UI, config) that turns the engine
  into a specific product.
- **Barge-in:** interrupting the AI's speech mid-sentence (needs voice-activity detection + streaming).
- **SSE / WebSocket:** SSE = server streams text to the browser (chat); WebSocket = two-way realtime
  (voice).
- **MCP (Model Context Protocol):** a standard way to expose tools to models; here it's *one optional
  tool adapter*, not the core.
```
