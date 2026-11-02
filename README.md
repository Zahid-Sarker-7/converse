# Converse

The reusable AI engine behind the milestone project — **one engine, many products**. Every
capability (model routing, orchestration, later voice/vision/memory) sits behind a **port**
so providers and features swap by config, not by code. v1 product: a role-play trainer /
spoken-English + interview coach — converse with an AI persona, then get an LLM-judged
rubric scorecard.

> **Design & phase plan:** see [`TDD.md`](TDD.md) (architecture, decisions, status).
> Full product/feature spec: `../ventures/11-milestone-project.md` and `../ventures/roadmap/`.

## Architecture

```
                    Next.js frontend (:3000)
              provider selector · streaming chat · health badge
                              │  POST /api/chat (SSE)  ·  GET /health  ·  /api/roleplay/*
 ┌────────────────────────────▼─────────────────────────────────────┐
 │ FastAPI backend (:8000)                                           │
 │   config (pydantic-settings)   ·   structlog   ·   /health        │
 │                                                                   │
 │   ModelPort (ABC)  ── stream_chat() · structured_output()        │
 │     ├─ EchoAdapter    reference impl (runs at $0, no keys)        │
 │     ├─ GroqAdapter    default free provider                      │
 │     └─ GeminiAdapter  second free provider                       │
 │                                                                   │
 │   Orchestrator (LangGraph)  ── graph of agent nodes / shared state│
 │     Scenario-Director · Role-Player · Evaluator (LLM-judge)      │
 └───────────────────────────────────────────────────────────────────┘
```

**Why a port/adapter?** Everything in the engine talks to `ModelPort`, never to a vendor
SDK — swapping Groq ↔ Gemini ↔ (later) Claude is a dropdown / env change. LangGraph nodes
are plain `(state) -> dict` functions, so the same adapters plug straight into the agent
graph with no extra wrapper deps. See `backend/app/core/model_port.py`.

## Quickstart

```bash
cd converse
make setup        # venv + backend deps + npm install (+ seeds .env / .env.local)
make dev          # backend on :8000, frontend on :3000
```

Open <http://localhost:3000>, keep the provider on **echo**, and send a message — it streams
back token-by-token with no keys and no network.

**Use a real provider:** `make providers` (installs the SDKs) → add `GROQ_API_KEY` and/or
`GOOGLE_API_KEY` to `backend/.env` → set `DEFAULT_PROVIDER=groq` (or pick from the UI
dropdown). Groq's free tier needs no card; Gemini's free tier covers Flash models.

Other targets: `make backend`, `make frontend`, `make stop`, `make health`, `make test`,
`make lint`, `make fmt`. Run `make help` for the list.

> `uv` works too: `cd backend && uv venv && uv pip install -r requirements-dev.txt`.

### Voice (Phase 2)

Speak to the persona over a WebSocket: turn detection (Silero VAD + Smart-Turn v3, local ONNX)
→ STT (Groq Whisper) → the same LangGraph loop → TTS, with barge-in and delivery metrics.

```bash
make providers voice     # Groq SDK + onnxruntime; downloads the two turn-detection models
make backend             # terminal 1 — watch the per-step logs
cd backend               # terminal 2 — a scripted "mic" over the real WebSocket:
say -o /tmp/me.wav --data-format=LEI16@16000 "Um, so I led the migration. [[slnc 1500]] It was hard."
.venv/bin/python scripts/voice_client.py /tmp/me.wav --stt groq   # prints events + metrics
```

Defaults are `VOICE_STT_PROVIDER=echo` / `VOICE_TTS_PROVIDER=echo` ($0, no keys, silent audio).
Real speech out needs Kokoro: `pip install kokoro` in a Python 3.12/3.13 venv (its spacy
dependency doesn't build on 3.14); it auto-registers as `--tts kokoro`.
Browser origins must be in `CORS_ORIGINS` (the WebSocket checks `Origin`; CORS alone doesn't).

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | app + per-provider readiness |
| POST | `/api/chat` | provider-agnostic streaming chat (SSE) |
| GET | `/api/roleplay/sample` | a ready-to-use `ScenarioConfig` |
| POST | `/api/roleplay/turn` | one learner turn → in-character persona reply |
| POST | `/api/roleplay/score` | grade a transcript → rubric `Scorecard` |
| WS | `/api/voice/session` | voice session: JSON config → PCM16@16k mic up; events + PCM16@24k speech down; `{"type":"end"}` → delivery metrics |
| GET | `/api/voice/providers` | STT/TTS adapter readiness |

Interactive API docs at <http://localhost:8000/docs>. Current implementation status per
endpoint is tracked in [`TDD.md`](TDD.md) §4.

## Hot-reload & debugging

| | Command | Hot-reload | Breakpoints |
|---|---|:--:|:--:|
| Backend | `make backend` (uvicorn `--reload`) | ✅ | ✖ |
| Frontend | `make frontend` (Next.js Fast Refresh) | ✅ | ✖ |
| Backend debug | VS Code ▶ **Backend: FastAPI (debug)** | ✖ (no `--reload`) | ✅ |
| Frontend (client) | VS Code ▶ **Frontend: Chrome (client)** | ✅ | ✅ |
| Frontend (server) | VS Code ▶ **Frontend: Next.js server (debug)** | ✅ | ✅ |

The backend debug config drops `--reload` because uvicorn's reloader runs the server in a
child process the debugger can't reliably attach to (breakpoints silently miss). Use
`make backend` for reload; use the debug config for breakpoints. For client-side React
breakpoints, run `make frontend` first, then the Chrome config (or the **Full stack**
compound). **Open the `converse/` folder in VS Code** so `.vscode/` is picked up.

> Ports stuck (`Address already in use`, or Next.js jumping to :3001)? Run `make stop` to
> free :8000/:3000/:3001, then `make dev` again.

## Layout

```
converse/
├── TDD.md                     technical design + phase status
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app + lifespan (builds the provider registry)
│   │   ├── config.py          pydantic-settings (providers, keys, defaults)
│   │   ├── logging.py         structlog setup (pretty dev / JSON prod)
│   │   ├── api/               health.py · chat.py (SSE) · agents.py (/api/roleplay)
│   │   ├── core/
│   │   │   ├── model_port.py  the ModelPort interface + message types
│   │   │   └── providers/     echo (reference) · groq · gemini
│   │   ├── agents/            LangGraph graph + agent nodes + domain schemas
│   │   └── models/schemas.py  request/response models
│   ├── tests/                 health/echo smoke + graph wiring tests
│   ├── requirements*.txt      core / dev / providers
│   └── .env.example
├── frontend/                  Next.js (App Router) chat UI
└── .vscode/                   debug + editor config
```

## Notes & constraints

- **$0 build.** Groq's free tier needs no card; Gemini's free tier covers Flash models (but
  trains on your inputs and is unavailable in the EEA/UK — use non-sensitive test data). A
  Claude adapter is deferred until a (paid) Anthropic key exists; the port already accepts it.
- **Model IDs drift.** Verify current IDs against each provider's console before relying on
  the defaults in `.env.example`.
- **Structured output.** The LLM-judge uses Gemini's `response_schema` (reliable schema
  enforcement); Groq's `gpt-oss-120b` can ignore schemas, so validate + retry if used there.
- **Deploy (later phases).** Vercel for the Next.js frontend; the realtime/voice backend
  needs a persistent host (Modal / Deno Deploy / small always-on box), not pure serverless.
