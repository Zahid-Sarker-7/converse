"""Smoke test: the app boots, the health route works, echo is always registered."""

from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:  # `with` triggers the lifespan (builds the registry)
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        names = [p["name"] for p in body["providers"]]
        assert "echo" in names


def test_echo_chat_streams() -> None:
    with TestClient(app) as client:
        resp = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "hello world"}], "provider": "echo"},
        )
        assert resp.status_code == 200
        assert "echo>" in resp.text
        assert '"done": true' in resp.text
