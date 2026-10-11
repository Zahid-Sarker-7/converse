"""FastAPI application entrypoint.

Run:  uvicorn app.main:app --reload --port 8000   (from the backend/ directory)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import agents, chat, health, voice
from .config import get_settings
from .core.providers import build_registry
from .core.voice import build_stt_registry, build_tts_registry
from .logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    registry = build_registry(settings)
    app.state.settings = settings
    app.state.registry = registry
    app.state.stt_registry = build_stt_registry(settings)
    app.state.tts_registry = build_tts_registry(settings)

    log = get_logger("converse")
    log.info(
        "startup",
        providers=list(registry.keys()),
        stt=list(app.state.stt_registry.keys()),
        tts=list(app.state.tts_registry.keys()),
        default_provider=settings.default_provider,
        environment=settings.environment,
    )
    yield
    log.info("shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)

    cors_kwargs: dict = {
        "allow_origins": settings.cors_origins,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    if settings.cors_allow_localhost:
        cors_kwargs["allow_origin_regex"] = r"http://(localhost|127\.0\.0\.1):\d+"
    app.add_middleware(CORSMiddleware, **cors_kwargs)

    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api")
    app.include_router(agents.router, prefix="/api")
    app.include_router(voice.router, prefix="/api")

    @app.get("/")
    async def root() -> dict:
        return {
            "name": settings.app_name,
            "version": settings.version,
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
