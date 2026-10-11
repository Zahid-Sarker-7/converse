"""Typed application settings, loaded from environment / .env.

pydantic-settings reads each field from an env var of the same name (case-insensitive):
`default_provider` <- DEFAULT_PROVIDER, `groq_api_key` <- GROQ_API_KEY, etc.
Complex fields (lists) are parsed as JSON, e.g. CORS_ORIGINS=["http://localhost:3000"].
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "Converse"
    version: str = "0.1.0"
    environment: str = "development"  # development | production
    log_level: str = "INFO"
    log_json: bool = False  # True -> JSON logs (prod); False -> pretty console (dev)

    # --- Server / CORS ---
    cors_origins: list[str] = ["http://localhost:3000"]
    # Also allow any http://localhost:PORT / 127.0.0.1:PORT in dev — Next.js falls back
    # to :3001, :3002, … when :3000 is taken. Set False to lock down to cors_origins only.
    cors_allow_localhost: bool = True

    # --- Model routing ---
    # "echo" runs with no keys and no network — the default so the app works out of the box.
    # Switch to "groq" or "gemini" once you implement those adapters and add a key.
    default_provider: str = "echo"

    # --- Provider credentials (all optional; only needed for the real adapters) ---
    groq_api_key: str | None = None
    google_api_key: str | None = None
    anthropic_api_key: str | None = None  # reserved for a future (paid) Claude adapter

    # --- Default model IDs per provider (override in .env) ---
    # NOTE: provider catalogs churn — verify current IDs before relying on these.
    groq_model: str = "openai/gpt-oss-120b"
    gemini_model: str = "gemini-3.1-flash-lite"

    # --- Voice (Phase 2) ---
    # STT/TTS pick their adapter the same way chat picks a provider. "echo" needs no keys and
    # runs the whole voice pipeline at $0 (silent audio). Switch to groq/kokoro once ready.
    voice_stt_provider: str = "echo"  # echo | groq (Whisper, free tier — needs GROQ_API_KEY)
    voice_tts_provider: str = "echo"  # echo | kokoro (local, Apache-2.0 — pip install kokoro)
    whisper_model: str = "whisper-large-v3-turbo"  # Groq STT model (free tier)
    kokoro_voice: str = "af_heart"  # Kokoro voice id (a=American English)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so settings are parsed once per process."""
    return Settings()
