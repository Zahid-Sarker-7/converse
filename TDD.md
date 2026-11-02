# Converse — Technical Design Document

> Engineering design + phased delivery plan for the **Converse engine** (one AI engine,
> many products). This is the *technical* source of truth; the *learning/career* protocol
> and full feature taxonomy live in `../ventures/roadmap/` (`project-plan.md`, `STATE.md`)
> and `../ventures/11-milestone-project.md`.

**Status at a glance** — _updated 2026-10-04_

| Phase | Title | Status |
|:--:|-------|--------|
| 0 | Foundations (FastAPI + Next.js + Model port) | ✅ **Done** |
| 1 | Multi-agent text core (LangGraph) | ✅ **Done** |
| 2 | Voice (STT→LLM→TTS, barge-in) | ✅ **Hand-built done** (Pipecat migration planned, see STATE 2026-11-01) |
| 3 | Vision (optional) | ⬜ Planned |
| 4 | Evals & CI | ⬜ Planned |
| 5 | Memory & personalization | ⬜ Planned |
| 6 | Guardrails & safety | ⬜ Planned |
| 7 | Observability & cost | ⬜ Planned |
| 8 | SaaS shell | ⬜ Planned |
| 9 | Boilerplate extraction (`create-ai-app`) | ⬜ Planned |

---

## 1. Purpose & scope

Build **one flexible AI engine** — model routing, multi-agent orchestration, voice, vision,
memory, evals, guardrails, observability — and ship products by swapping a lightweight
**product pack** (personas, prompts, rubrics, UI). v1 product = a **role-play trainer /
spoken-English + interview coach**: the learner converses with an AI persona, then an
LLM-judge scores the transcript against a rubric with quoted evidence.

**Goals**
- Every engine capability sits behind a **port** (interface); providers/features are swappable adapters.
- $0 / free-tier first (Groq + Gemini free tiers; local where possible).
- Each phase ships something demoable and teaches one ecosystem layer.

**Non-goals**
- No defensible moat required; a solid, sellable product is the bar.
- Not competing with frontier labs — the model is a swappable component.
- One phase at a time, shipped; don't parallelize phases.

---

## 2. Architecture

```
              Next.js frontend (:3000)  — chat UI · provider selector · (later) voice/dashboards
                        │  REST/SSE  (WebSocket in Phase 2 for voice)
 ┌───────────────────────▼──────────────────────────────────────────────────────────────┐
 │ FastAPI backend (:8000)                                                                │
 │   config (pydantic-settings) · structlog · /health                                     │
 │                                                                                        │
 │   Model port ──────► stream_chat() · structured_output()                               │
 │     ├─ EchoAdapter   ✅   ├─ GroqAdapter ✅   └─ GeminiAdapter ✅                        │
 │                                                                                        │
 │   Orchestrator (LangGraph) ✅  — graph of agent nodes over shared state (reused by voice) │
 │     ├─ Scenario-Director ✅   ├─ Role-Player ✅   └─ Evaluator (LLM-judge) ✅            │
 │                                                                                        │
 │   Voice port 🟡 ── STT (transcribe) · TTS (synthesize)   (adapters: echo ✅,            │
 │     │               Groq-Whisper ✅, Kokoro ✅)  +  WS pipeline turn→STT→graph→TTS       │
 │     └─ hands-on TODOs: turn detection · barge-in · delivery metrics                     │
 │                                                                                        │
 │   [Planned] Vision port · Memory port · Guardrail port · Eval port · OTel              │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

**Pattern:** hexagonal / ports-and-adapters + dependency injection. The rest of the app
depends only on the port interface, never a vendor SDK. LangGraph nodes are plain
`(state) -> dict` functions, so the hand-built `ModelPort` adapters plug in directly — **no
LangChain model wrappers** and no extra provider deps.

---

## 3. Key design decisions

| # | Decision | Rationale | Alternative rejected |
|--:|----------|-----------|----------------------|
| D1 | **Direct provider SDKs behind a hand-written `ModelPort`** | building the adapter *is* the learning goal; keeps the abstraction visible | LiteLLM (would erase the abstraction) |
| D2 | **Groq (default) + Gemini** free tiers, Claude deferred | $0 build; Anthropic API is paid | Claude default (port already accepts it later) |
| D3 | **Gemini streaming via Interactions API** (`interactions.create`) | Google's GA/recommended surface; verified on `google-genai` 2.20.0 | legacy `generate_content_stream` (a prior "fix" here was wrong + reverted) |
| D4 | **LangGraph** for orchestration; nodes call `ModelPort` directly | provider-agnostic; no wrapper deps; teaches the state-machine model | LangChain agents / custom loop |
| D5b | **Groq structured output now implemented too** (2026-09-13) — OpenAI-compat `response_format: {type: json_schema, json_schema:{name, schema}}` + validate-and-retry-once; live-verified via `/api/roleplay/score` with `provider=groq`. Frontend scores with the selected provider (Groq/Gemini), steering only away from Echo. | lets the demo score on the default Groq provider without forcing a Gemini switch | forcing Gemini for every score (previous FE stopgap) |
| D5 | **Evaluator uses Gemini structured output via the Interactions API** — non-streaming `interactions.create(..., response_format={"type":"text","mime_type":"application/json","schema": …})` → `interaction.output_text` → `model_validate_json` | matches the live Gemini structured-output doc; Interactions is the current surface (`generate_content` is legacy); live-verified via `/score` | Groq `response_format` (gpt-oss-120b sometimes ignores schema → validate+retry). *(Note: an earlier revision of this row said `generate_content` — corrected 2026-09-07 after checking the live doc + `google-genai 2.20.0` `_gaos` types.)* |
| D6 | **Postgres deferred to Phase 5** | Phase 1 is a text-core demo; DB belongs with memory + LangGraph checkpointer | scaffolding SQLAlchemy now (YAGNI) |
| D7 | **Frontend = Tailwind v4 + `motion` + `lucide-react`, hand-rolled design system** | most polish per line, React 19 + SSR native, zero JS config; gauge is custom SVG regardless | **shadcn/ui** (radix + CLI ceremony for little gain here), `next-themes` (dark-only), `react-countup`/chart libs (motion + SVG cover it) |
| D8 | **Voice = our own WS pipeline behind a `VoicePort` (STTPort+TTSPort); STT Groq Whisper (free, word timestamps, batch-per-utterance) · TTS Kokoro (Apache-2.0, local, sentence-chunked) · turn = Silero VAD + Smart-Turn v3 (all $0)** | reuses the port pattern (mirrors D1 — hand-building is the lesson, keeps the abstraction visible, zero lock-in); reuses the Phase-1 LangGraph loop unchanged; WS (not SSE) for two-way interruptible audio; verified current Oct-2026 (Principle 0) | **LiveKit/Pipecat framework** (owns the loop, hides the learning — deferred for telephony/scale); browser Web Speech (prototype-grade, no word timestamps → weak metrics — kept as FE v0 only); Piper TTS (active fork is GPL-3.0); Deepgram STT (true streaming but paid) |

Model defaults (verified live): `GROQ_MODEL=openai/gpt-oss-120b` (Groq dropped Llama 3.x
instruct), `GEMINI_MODEL=gemini-3.1-flash-lite` (`gemini-3.7-flash` is quota-blocked on free tier).

---

## 4. Data contracts & API surface

**Core types** (`backend/app/core/model_port.py`): `Role`, `ChatMessage(role, content)`,
`StreamChunk(delta, done)`, `ModelPort.stream_chat()` / `.structured_output()`.

**Phase-1 domain models** (`backend/app/agents/schemas.py`):

```
ScenarioConfig(title, persona, objective, difficulty, curveballs[], rubric[])   # the "pack"
CriterionScore(criterion, score 1–5, evidence=verbatim quote, rationale)
Scorecard(criteria[], overall 1–5, summary, focus_next)                         # judge output = JSON schema
```

**Endpoints**

| Method | Path | Status | Purpose |
|--------|------|:--:|---------|
| GET | `/health` | ✅ | app + per-provider readiness |
| POST | `/api/chat` | ✅ | provider-agnostic streaming chat (SSE) |
| GET | `/api/roleplay/sample` | ✅ | a ready-to-use `ScenarioConfig` |
| POST | `/api/roleplay/turn` | ✅ | one learner turn → Role-Player reply (graph `mode=chat`) |
| POST | `/api/roleplay/score` | ✅ | grade transcript → `Scorecard` (graph `mode=score`); Gemini judge live-verified |
| WS | `/api/voice/session` | 🟡 | bidirectional voice: mic PCM16@16k up → turn→STT→graph→TTS → PCM16@24k + `VoiceEvent`s down |
| GET | `/api/voice/providers` | 🟡 | readiness of the STT/TTS adapters (for UI/health) |

---

## 5. Phase breakdown

### Phase 0 — Foundations ✅ Done (2026-08-28)
- **Delivered:** FastAPI backend (config, structlog, `/health`, SSE `/api/chat`),
  `ModelPort` ABC + provider registry, **Echo/Groq/Gemini adapters** (Groq + Gemini
  live-verified streaming), Next.js chat UI (provider swap + health badge), Makefile,
  tests, VS Code debug config.
- **Exit criteria met:** one interface streams from either real provider, swappable by config.

### Phase 1 — Multi-agent text core ✅ Done (2026-09-07)
**Goal:** a LangGraph graph orchestrating Scenario-Director + Role-Player + Evaluator →
text role-play + a rubric scorecard with quoted evidence.

```
   START ──route(mode)──┬─ "chat"  ─→ scenario_director → role_player ─┐
                        │                                             ├─→ END
                        └─ "score" ──────────────→ evaluator ────────┘
   State: ConverseState { scenario, messages(+append reducer), mode, director_note, scorecard }
```

> **Note:** the LangGraph construction was a *learning* deliverable, so it was built hands-on by
> Zahid (T1–T6) against a guided skeleton; AI provided only the non-LangGraph frame (schemas +
> HTTP wiring) and reviewed. All complete.

| Component | Status | Notes |
|-----------|:--:|-------|
| Domain schemas + sample scenario (`agents/schemas.py`) | ✅ | `Scorecard` doubles as the judge JSON schema |
| API routes + graph wiring (`api/agents.py`) | ✅ | `/turn`, `/score`, `/sample` |
| **T1** `ConverseState` (TypedDict + `operator.add` reducer) | ✅ | state + reducer |
| **T2** Scenario-Director node | ✅ | rule-based curveball rotation |
| **T3** `route_mode` (conditional edge fn) | ✅ | chat/score branching |
| **T4** `build_graph` (nodes + edges + compile) | ✅ | conditional entry on `mode` |
| **T5** Role-Player node | ✅ | async; streams `chat_port`; appends reply |
| **T6** Evaluator node + `GeminiAdapter.structured_output` | ✅ | Interactions API `response_format` + `output_text`; **live-verified** |
| Tests (`tests/test_graph.py`) | ✅ | 6 green (director, router, both graph paths) |
| **Frontend** role-play studio + scorecard reveal + quick chat (`frontend/`) | ✅ | Next.js/Tailwind v4/`motion`; animated Hero + editable scenario + typewriter bubbles + director-note tag + radial-gauge scorecard; graceful offline/501 states; `npm run build` clean. `lib/api.ts` fully wired (`sample`/`turn`/`score`, shared `errorDetail` surfaces FastAPI `detail`/501) |

**Exit criteria (met):** in-character multi-turn conversation; `/score` returns a schema-valid
`Scorecard` with a verbatim quote per criterion (live-verified with the Gemini judge); all tests green.

### Phase 2 — Voice 🟡 Scaffolded (2026-10-04)
**Goal:** speak to the persona live and get scored incl. *delivery*. A `VoicePort`
(STT + TTS, mirroring `ModelPort`) feeds a WebSocket pipeline that reuses the Phase-1
LangGraph loop **unchanged**: `turn-detect → STT → graph(mode=chat) → sentence-chunked TTS`.

```
  🎙 mic ─PCM16@16k─► WS /api/voice/session
     TurnDetector (Silero VAD + Smart-Turn v3) ─utterance─► STTPort.transcribe (Groq Whisper)
        └─ barge-in: VAD fires while speaking → cancel TTS <60ms + drain
     → build_graph(...).ainvoke(mode="chat")  [UNCHANGED Phase-1 brain]
     → TTSPort.synthesize (Kokoro, per sentence) ─PCM16@24k─► 🔊 + DeliveryMetrics on end
```

| Component | Status | Where |
|-----------|:--:|-------|
| `VoicePort` = `STTPort` + `TTSPort` + types (`Word`/`Transcript`/`AudioChunk`) | ✅ | `core/voice_port.py` |
| `EchoSTT`/`EchoTTS` ($0, keyless — run the whole pipeline) | ✅ | `core/voice/echo_voice.py` |
| `GroqWhisperSTT` (free tier, `verbose_json` word timestamps, batch per-utterance) | ✅ | `core/voice/groq_whisper_stt.py` |
| `KokoroTTS` (Apache-2.0, local, 24kHz, float32→PCM16) | ✅ | `core/voice/kokoro_tts.py` |
| STT/TTS registries (register on key/package present) | ✅ | `core/voice/__init__.py` |
| WS route `/session` + `/providers`; config/`.env`/`requirements-voice.txt`/`make voice` | ✅ | `api/voice.py`, `config.py` |
| Pipeline orchestration (STT→graph→sentence-chunk TTS, out-queue) | ✅ | `app/voice/pipeline.py` |
| Frontend voice client (mic→16k PCM16→WS→24k playback) | ✅ scaffold | `frontend/lib/voice.ts` *(not tsc-verified yet)* |
| **TODO #1** Turn detection — Silero VAD + Smart-Turn v3.2-cpu, both raw ONNX via onnxruntime (no torch); 512-sample chunk buffering, Smart-Turn on the whole turn after 200ms silence, 3s max-silence fallback | ✅ (AI-implemented on request, 2026-10-24) | `app/voice/turn.py` |
| **TODO #2** Barge-in — `speech_ms`≥200ms guard, cancel + drop queued audio + client `interrupt` flush; blips ignored | ✅ (AI, 2026-11-01) | `app/voice/pipeline.py::_barge_in` |
| E2E client (scripted mic over the real WS) | ✅ | `backend/scripts/voice_client.py` |
| **TODO #3** Delivery metrics — WPM/fillers (punctuation-normalized, two-word phrases)/talk-ratio/longest in-turn pause | ✅ (Zahid, 2026-11-01) | `app/voice/metrics.py` |
| Hardening — WS Origin check, handshake validation, sentinel-flushed metrics, 60s turn cap, Kokoro off-loop, PII at DEBUG | ✅ | `api/voice.py`, `voice/*` · `tests/test_voice_ws.py` |
| Tests (`tests/test_voice.py`) — skip-gated acceptance, 2 live + 3 TODO-gated | ✅ | 8 pass / 3 skip |

v0 browser Web Speech (serverless) is noted as an even-cheaper FE fallback; the server
pipeline is chosen because the three learning TODOs (VAD/turn, barge-in, word-timestamp
metrics) only exist there. **LiveKit/Pipecat deferred** (see D8). **Exit:** talk to the
persona live, be interrupted cleanly, and get a scorecard with delivery metrics.

### Phase 3 — Vision ⬜ Planned (optional)
"AI sees your slide": screen/slide capture (`getDisplayMedia`) → vision LLM feeds "did they
use it correctly?" into scoring. **Exit:** present a slide → feedback references it.

### Phase 4 — Evals & CI ⬜ Planned
Rubric eval harness; judge calibration vs a ~30-session human gold set; **CI regression
gate**; red-team persona prompts (Promptfoo). **Exit:** a bad prompt/model change fails CI.

### Phase 5 — Memory & personalization ⬜ Planned
**Postgres + pgvector**; LangGraph `PostgresSaver` checkpointing; learner profile + weakness
extraction; adaptive difficulty; semantic scenario recall. **Exit:** "it remembers your weak
spots and ramps difficulty." *(Postgres data model D6 lands here.)*

### Phase 6 — Guardrails & safety ⬜ Planned
Keep Role-Player in scope; PII/safe-content classifier on the transcript (off hot path);
confirm risky actions. **Exit:** persona can't be derailed; unsafe content flagged.

### Phase 7 — Observability & cost ⬜ Planned
Per-step OTel GenAI spans; token/cost/latency dashboards; per-session trace viewer
(Langfuse). **Exit:** see cost + full trace of any session.

### Phase 8 — SaaS shell ⬜ Planned
Multi-tenant org/teams/roles; SSO (Auth.js/Clerk); scenario-builder UI; manager dashboards;
**Stripe** per-seat + voice-minute metering. **Exit:** a manager signs up a team, assigns a
scenario, sees scores, and pays.

### Phase 9 — Boilerplate extraction ⬜ Planned
Split **engine** vs **product pack**; `create-ai-app` CLI scaffolder; prove reuse by
generating pack #2 (consumer English/interview coach). **Exit:** `create-ai-app` spins up a
running app from the same engine.

---

## 6. Testing strategy

| Layer | Approach | Now |
|-------|----------|-----|
| Adapters | echo smoke test (no keys/network); real providers live-verified over HTTP | ✅ |
| Graph | unit-test node logic (director), routing, topology (both paths) with `EchoAdapter` + `asyncio.run` (no pytest-asyncio dep) | ✅ |
| Evaluator | score-path unit test via a `FakeJudge` (no keys); Gemini judge live-verified over HTTP | ✅ |
| Voice | echo STT/TTS drive the full pipeline (no keys/network): turn-detector start/end + one STT→graph→TTS turn; 3 skip-gated acceptance tests tied to the TODOs (turn-waits-through-pause, barge-in-stops-and-drains, metrics-from-timestamps) | 🟡 2 live / 3 TODO-gated |
| Regression (Phase 4) | judge calibration vs gold set; CI gate blocks on score regression | ⬜ |

Run: `make test` · `make lint` · `make providers` (installs `groq`, `google-genai`, `langgraph`).

---

## 7. Risks & open questions

| Risk | Mitigation |
|------|------------|
| Groq gpt-oss-120b ignores structured-output schema | **implemented** on Groq via OpenAI-compat `response_format: json_schema` + **validate-and-retry-once** (feeds the validation error back); live-verified returning a valid Scorecard. Gemini remains the more reliable judge. |
| Provider model IDs / free tiers drift monthly | verify IDs live before relying on defaults (Principle 0) |
| Realtime voice needs a persistent host (Vercel can't) | Phase 2: LiveKit Cloud + Modal/Deno backend; keep FE on Vercel |
| Gemini free tier trains on inputs, no EEA/UK | non-sensitive test data only; note in privacy stance (Phase 8) |
| LLM-judge drift over time | calibration set + CI gate (Phase 4) |
