.DEFAULT_GOAL := help
PY ?= python3
BACKEND := backend
VENV := $(BACKEND)/.venv
BIN := $(VENV)/bin

.PHONY: help setup setup-backend setup-frontend backend frontend dev stop health graph test lint fmt providers voice clean

help: ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n",$$1,$$2}'

setup: setup-backend setup-frontend ## install backend + frontend deps

setup-backend: ## create venv, install backend deps, seed .env
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r $(BACKEND)/requirements-dev.txt
	@test -f $(BACKEND)/.env || cp $(BACKEND)/.env.example $(BACKEND)/.env
	@echo "backend ready. run: make backend"

setup-frontend: ## install frontend deps, seed .env.local
	cd frontend && npm install
	@test -f frontend/.env.local || cp frontend/.env.local.example frontend/.env.local
	@echo "frontend ready. run: make frontend"

providers: ## install provider SDKs (do this when starting the adapter TODOs)
	$(BIN)/pip install -r $(BACKEND)/requirements-providers.txt

voice: ## install Phase-2 voice deps (Kokoro TTS + Silero/Smart-Turn) for the voice TODOs
	$(BIN)/pip install -r $(BACKEND)/requirements-voice.txt

backend: ## run FastAPI with reload (http://localhost:8000)
	cd $(BACKEND) && .venv/bin/uvicorn app.main:app --reload --port 8000

frontend: ## run Next.js dev server (http://localhost:3000)
	cd frontend && npm run dev

dev: ## run backend + frontend together
	@$(MAKE) -j2 backend frontend

stop: ## free stuck dev ports (8000 backend, 3000/3001 frontend)
	@for p in 8000 3000 3001; do \
		pids=$$(lsof -ti :$$p 2>/dev/null); \
		if [ -n "$$pids" ]; then echo "killing :$$p -> $$pids"; kill $$pids 2>/dev/null || true; fi; \
	done; echo "dev ports freed"

health: ## curl the health endpoint
	@curl -s http://localhost:8000/health | (command -v jq >/dev/null 2>&1 && jq . || cat)

graph: ## print the LangGraph diagram (Mermaid) — paste into https://mermaid.live
	@cd $(BACKEND) && .venv/bin/python -c "from app.agents.graph import build_graph; from app.core.providers.echo_adapter import EchoAdapter as E; print(build_graph(E(), E()).get_graph().draw_mermaid())"

test: ## run backend tests
	cd $(BACKEND) && .venv/bin/pytest -q

lint: ## ruff check
	cd $(BACKEND) && .venv/bin/ruff check .

fmt: ## ruff format
	cd $(BACKEND) && .venv/bin/ruff format .

clean: ## remove venv, caches, node_modules, build output
	rm -rf $(VENV) $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache
	find $(BACKEND) -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf frontend/node_modules frontend/.next
