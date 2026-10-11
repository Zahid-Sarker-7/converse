"""Health / readiness route.

Reports the app version and every registered provider with a `ready` flag, so you can
see at a glance which adapters are configured and implemented.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    settings = request.app.state.settings
    registry = request.app.state.registry
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "default_provider": settings.default_provider,
        "providers": [
            {"name": name, "ready": port.ready} for name, port in registry.items()
        ],
    }
