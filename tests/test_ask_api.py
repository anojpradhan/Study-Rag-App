from fastapi.testclient import TestClient

from app.api.v1.routes.ask import rag_service
from app.main import app

client = TestClient(app)


def test_ask_rejects_empty_query():
    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "",
            "study_space_id": "dev-study-space",
            "session_id": "test-session",
            "limit": 5,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_limit():
    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "What is cloud computing?",
            "study_space_id": "dev-study-space",
            "session_id": "test-session",
            "limit": 100,
        },
    )

    assert response.status_code == 422


def test_ask_returns_answer(monkeypatch):
    def fake_answer_question(
        query: str,
        study_space_id: str,
        session_id: str,
        limit: int = 5,
    ) -> dict:
        return {
            "answer": "Cloud computing provides computing resources over the internet.",
            "sources": [],
            "model_used": "test-model",
            "fallback_used": False,
            "session_id": session_id,
        }

    monkeypatch.setattr(
        rag_service,
        "answer_question",
        fake_answer_question,
    )

    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "What is cloud computing?",
            "study_space_id": "dev-study-space",
            "session_id": "test-session",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"].startswith("Cloud computing")
    assert data["session_id"] == "test-session"
    assert data["fallback_used"] is False
