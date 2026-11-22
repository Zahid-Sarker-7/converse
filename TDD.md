# Converse — Technical Design Document

> Engineering design + phased delivery plan for the **Converse engine** (one AI engine,
> many products). This is the *technical* source of truth; the *learning/career* protocol
> and full feature taxonomy live in `../ventures/roadmap/` (`project-plan.md`, `STATE.md`)
> and `../ventures/11-milestone-project.md`.

**Status at a glance** — _updated 2026-11-22_

| Phase | Title | Status |
|:--:|-------|--------|
| 0 | Foundations (FastAPI + Next.js + Model port) | ✅ **Done** |
| 1 | Multi-agent text core (LangGraph) | ✅ **Done** |
| 2 | Voice (STT→LLM→TTS, barge-in) | ✅ **Done — on Pipecat** (cut over 2026-11-22; hand-built reference = tag `v0.2-hand-built-voice`, §5 Phase 2b) |
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
 │   Voice (Pipecat 1.15) ✅ ── STT service (Groq Whisper · echo) · TTS (Kokoro · echo)    │
 │     │   turn = Silero VAD + Smart-Turn v3 · barge-in = InterruptionFrame                 │
 │     └─ LangGraph plugs in as ConverseBrain; DeliveryTracker observes frames → metrics    │
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
| D8 | ~~superseded by D9 (2026-11-22)~~ **Voice = our own WS pipeline behind a `VoicePort` (STTPort+TTSPort); STT Groq Whisper (free, word timestamps, batch-per-utterance) · TTS Kokoro (Apache-2.0, local, sentence-chunked) · turn = Silero VAD + Smart-Turn v3 (all $0)** | reuses the port pattern (mirrors D1 — hand-building is the lesson, keeps the abstraction visible, zero lock-in); reuses the Phase-1 LangGraph loop unchanged; WS (not SSE) for two-way interruptible audio; verified current Oct-2026 (Principle 0) | **LiveKit/Pipecat framework** (owns the loop, hides the learning — deferred for telephony/scale); browser Web Speech (prototype-grade, no word timestamps → weak metrics — kept as FE v0 only); Piper TTS (active fork is GPL-3.0); Deepgram STT (true streaming but paid) |
| D9 | **Voice plumbing on Pipecat 1.15 (supersedes the "own WS pipeline" half of D8; Decision 2026-11-01).** Keep the brain (LangGraph), scoring and metrics maths; Pipecat owns transport, turn-taking (Silero + Smart-Turn v3, its *default* strategy = what `turn.py` hand-built), barge-in (`InterruptionFrame`) and TTS pacing. The graph plugs in as a custom `FrameProcessor` in place of `LLMService`; a custom `FrameSerializer` keeps the browser wire protocol byte-identical (frontend untouched); migrated behind a `VOICE_ENGINE` strangler flag, cut over 2026-11-22 (flag + hand-built plumbing removed). Turn calibration re-measured on Pipecat: VAD `stop_secs` **0.3** (0.5 made Smart-Turn call mid-thought pauses complete). Verified on **Python 3.14** (2026-11-08): core + Silero + Smart-Turn + Groq + FastAPI WS import and run; use `pipecat-ai[websocket,groq]`, **not** `[kokoro]` (pins kokoro-onnx ≥0.5 = Py<3.14; `KokoroTTSService` runs fine on 0.4.7). Uses the current `PipelineWorker` + `WorkerRunner.add_workers()` API (`PipelineTask`/`PipelineRunner` deprecated since 1.3). | hand-built plumbing taught the concepts; re-owning queue/cancel code has low marginal value; shipping on a framework is the marketable skill; no Py downgrade needed | pin a 3.12/3.13 venv (unnecessary); RTVI protocol + Pipecat JS client (would rewrite `voice.ts` — later, with a transport change); LiveKit Agents (phone/SIP strengths not needed yet; next rung of the voice ladder) |

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
| WS | `/api/voice/session` | ✅ | bidirectional voice: mic PCM16@16k up → turn→STT→graph→TTS → PCM16@24k + `VoiceEvent`s down |
| GET | `/api/voice/providers` | ✅ | which STT/TTS names a session can use (for UI/health) |

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

### Phase 2 — Voice ✅ Done (hand-built 2026-11-03 → Pipecat 2026-11-22, see Phase 2b)
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
| `KokoroTTS` via **kokoro-onnx** (int8, onnxruntime, no torch — official `kokoro` can't build on Py3.14); ~0.5s first audio, ~5× realtime on CPU; runs in a thread | ✅ (2026-11-02) | `core/voice/kokoro_tts.py` |
| STT/TTS registries (register on key/package present) | ✅ | `core/voice/__init__.py` |
| WS route `/session` + `/providers`; config/`.env`/`requirements-voice.txt`/`make voice` | ✅ | `api/voice.py`, `config.py` |
| Pipeline orchestration (STT→graph→sentence-chunk TTS, out-queue) | ✅ | `app/voice/pipeline.py` |
| Frontend voice: **Talk/End** in Role-play studio, live status line, transcripts in the thread, Delivery card (inline + in scorecard); client fixes (mic before socket, wait for metrics before close, onClose, AEC constraints) | ✅ tsc + build clean; **real-UI E2E verified** (Chrome fake mic → Groq STT → LLM → Kokoro → Delivery card) | `frontend/components/RoleplayStudio.tsx`, `lib/voice.ts` |
| **TODO #1** Turn detection — Silero VAD + Smart-Turn v3.2-cpu, both raw ONNX via onnxruntime (no torch); 512-sample chunk buffering, Smart-Turn on the whole turn after 200ms silence, 3s max-silence fallback | ✅ (AI-implemented on request, 2026-10-24) | `app/voice/turn.py` |
| **TODO #2** Barge-in — `speech_ms`≥200ms guard, cancel + drop queued audio + client `interrupt` flush; blips ignored | ✅ (AI, 2026-11-01) | `app/voice/pipeline.py::_barge_in` |
| E2E client (scripted mic over the real WS) | ✅ | `backend/scripts/voice_client.py` |
| Live-use fixes (2026-11-03, from real sessions): judge gets transcript as one labelled block (Gemini 400 on trailing model turn) + provider errors → 502; text+voice one conversation (history in handshake, typed turns mid-session); turn = Smart-Turn at 200ms but end ≥500ms; continuation merge; noise floor 250ms + Whisper-phantom filter ("Thank you."); mute-while-typing; talk:listen vs agent speech; pre-roll 320ms; carry untranscribed audio across barge-in; VAD-measured pauses (Whisper absorbs silence); Whisper filler/vocab prompt; hold 3s on unfinished-sounding text; end-reason logging | ✅ 32 tests · live-verified | `voice/*`, `api/*`, `agents/graph.py`, `RoleplayStudio.tsx` |
| **TODO #3** Delivery metrics — WPM/fillers (punctuation-normalized, two-word phrases)/talk-ratio/longest in-turn pause | ✅ (Zahid, 2026-11-01) | `app/voice/metrics.py` |
| Hardening — WS Origin check, handshake validation, sentinel-flushed metrics, 60s turn cap, Kokoro off-loop, PII at DEBUG | ✅ | `api/voice.py`, `voice/*` · `tests/test_voice_ws.py` |
| Tests (`tests/test_voice.py`) — skip-gated acceptance, 2 live + 3 TODO-gated | ✅ | 8 pass / 3 skip |

v0 browser Web Speech (serverless) is noted as an even-cheaper FE fallback; the server
pipeline is chosen because the three learning TODOs (VAD/turn, barge-in, word-timestamp
metrics) only exist there. ~~LiveKit/Pipecat deferred (D8)~~ → superseded by **D9**: plumbing moves to Pipecat now that the concepts are learned (Phase 2b). **Exit:** talk to the
persona live, be interrupted cleanly, and get a scorecard with delivery metrics.

### Phase 2b — Voice on Pipecat ✅ Done (scaffolded 2026-11-08 · cut over 2026-11-22)
**Goal:** same product behaviour, framework plumbing. The hand-built engine it replaced is tag
`v0.2-hand-built-voice` (read it to see what Pipecat now does for us). Browser protocol and
`voice.ts` unchanged.

```
  WS ─► transport.input ─► STT ─► VoiceEvents ─► DeliveryTracker ─► user aggregator
        (ConverseSerializer)  (Groq words | echo)                   (Silero VAD + Smart-Turn v3)
     ─► ConverseBrain (LangGraph, unchanged) ─► TTS (Kokoro | echo)
     ─► transport.output (ConverseSerializer) ─► WS               ─► assistant aggregator
  barge-in: user speech → InterruptionFrame → processors cancel in-flight work, output drops
            unsent audio, serializer sends {"type":"interrupt"} → browser flushes playback
  typed:    {"type":"text"} → interrupt, then LLMMessagesAppendFrame(run_llm) → brain
  end:      {"type":"end"}  → interrupt, then EndWorkerFrame → tracker emits metrics → close
  errors:   brain → `error` event; STT/TTS ErrorFrames → on_pipeline_error → `error` event
```

**Hand-built → Pipecat map**

| Hand-built (v0.2) | Pipecat 1.15 | Status |
|---|---|---|
| `turn.py` Silero + Smart-Turn, vendored `whisper_features.py` | default user-turn strategy (`SileroVADAnalyzer` + `LocalSmartTurnAnalyzerV3`, v3.2 ONNX bundled) | deleted |
| knobs: check at 200ms / end ≥500ms / 3s max silence / 200ms barge-in guard | `VADParams(start_secs=0.2, stop_secs=0.3, confidence=0.5)`, `SmartTurnParams(stop_secs=3.0)` — Pipecat has ONE endpoint knob; 0.5 split "because [pause] the old database…" into two turns, 0.2/0.3 keep it (real-model test) | re-calibrated |
| pre-roll 320ms · Silero RNN reset · carry untranscribed audio across barge-in | `SegmentedSTTService` 1s pre-buffer + background STT queue · built-in reset | deleted |
| 60s turn cap | — | dropped (`ponytail:` in `bot.py`) |
| `pipeline.py` `_respond` task, `_barge_in`, `_drop_queued_audio`, out-queue + pump | interruption semantics + output transport | deleted |
| sentence chunking for TTS | TTS text aggregation | deleted |
| `GroqWhisperSTT` (filler/vocab prompt, word timestamps) + phantom filter | `GroqWordsSTT` (`TranscriptionFrame.result` = our `Transcript`) | ported |
| `KokoroTTS` · `VoicePort` + registries | `KokoroTTSService` · `voice_providers()` (names only) | deleted |
| echo STT/TTS | `EchoSTT` / `EchoTTS` services (keyless; Pipecat is a core dep for this) | ported |
| origin check + handshake (`api/voice.py`) | unchanged, runs before Pipecat | kept |
| `graph.py`, `compute_delivery_metrics`, Scorecard | unchanged | kept |
| unfinished-hold, joining continuations, `VOICE_STYLE` | `ConverseBrain` (`to_chat_messages` joins consecutive user lines) | moved |
| delivery bookkeeping spread through the pipeline | `DeliveryTracker` (pure observer); longest pause = VAD silence the learner resumes from **within one turn** (the turn end clears it — a live run read 5.28s of endpointing + agent latency before this) | rebuilt |

| Component | Where |
|---|---|
| `ConverseSerializer` · `GroqWordsSTT` · echo services · `VoiceEvents` (state, user transcript, typed/end control) · `voice_providers` · `run_session` | `voice/bot.py` |
| `ConverseBrain` (TODO #1 — LangGraph as a frame processor) | `voice/brain.py` |
| `DeliveryTracker` (TODO #2 — metrics inputs from frames) | `voice/delivery.py` |
| Route: origin + handshake → `run_session` | `api/voice.py` |

**Verified 2026-11-22:** 26 tests (incl. a real-model Silero + Smart-Turn test and the WebSocket
contract on Pipecat); fresh venv from `requirements-dev.txt` installs + passes + boots keyless (echo);
real speech over the WS (`say` → Groq Whisper → graph → Kokoro: barge-in twice, End mid-reply →
metrics); **real-UI E2E** (Chrome fake mic: text → Talk → thinking pause → talk over the agent → type
mid-voice → End → Delivery card → Score me), 0 server errors.

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
| Voice | metric maths + unfinished-text check (`test_voice.py`); processors via `pipecat.tests.utils.run_test` or direct `process_frame` with the echo LLM (`test_voice_pipecat.py`), incl. one REAL Silero + Smart-Turn run on `say` speech guarding the turn calibration; WebSocket contract over a real socket on echo (`test_voice_ws.py`) | ✅ |
| Regression (Phase 4) | judge calibration vs gold set; CI gate blocks on score regression | ⬜ |

Run: `make test` · `make lint` · `make providers` (installs `groq`, `google-genai`, `langgraph`).

---

## 7. Risks & open questions

| Risk | Mitigation |
|------|------------|
| Groq gpt-oss-120b ignores structured-output schema | **implemented** on Groq via OpenAI-compat `response_format: json_schema` + **validate-and-retry-once** (feeds the validation error back); live-verified returning a valid Scorecard. Gemini remains the more reliable judge. |
| Provider model IDs / free tiers drift monthly | verify IDs live before relying on defaults (Principle 0) |
| Realtime voice needs a persistent host (Vercel can't) | small always-on host (Modal/Fly/VM) for the FastAPI+Pipecat backend; keep FE on Vercel |
| Gemini free tier trains on inputs, no EEA/UK | non-sensitive test data only; note in privacy stance (Phase 8) |
| Pipecat API churn (weekly releases; `PipelineTask` deprecated in 1.3) | pin `pipecat-ai>=1.15,<2`; current `PipelineWorker`/`WorkerRunner` API; fix deprecation warnings as they appear |
| `[kokoro]` extra unusable on Py3.14 | install `pipecat-ai[websocket,groq]` + `kokoro-onnx 0.4.x` + `requests` (verified) |
| Kokoro ONNX loaded per voice session (0.3s, measured) | `ponytail:` in `bot.py` — share one `Kokoro` if sessions pile up |
| LLM-judge drift over time | calibration set + CI gate (Phase 4) |
