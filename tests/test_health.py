from fastapi.testclient import TestClient

from app.api.v1.routes import health
from app.main import app

client = TestClient(app)


def test_health_returns_healthy_when_dependencies_are_available(monkeypatch):
    monkeypatch.setattr(
        health.qdrant_service.client,
        "get_collections",
        lambda: None,
    )
    monkeypatch.setattr(
        health.memory_service.client,
        "ping",
        lambda: True,
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "dependencies": {
            "qdrant": "healthy",
            "redis": "healthy",
        },
    }


def test_health_returns_degraded_when_redis_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        health.qdrant_service.client,
        "get_collections",
        lambda: None,
    )

    def redis_unavailable():
        raise ConnectionError("Redis unavailable")

    monkeypatch.setattr(
        health.memory_service.client,
        "ping",
        redis_unavailable,
    )

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["redis"] == "unhealthy"
