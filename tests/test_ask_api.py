
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.api.v1.routes.ask import rag_service
from app.db.session import get_db
from app.main import app
from app.models.study_space import StudySpace
from app.models.user import User


client = TestClient(app)


# -------------------------------------------------------------------
# Test helpers
# -------------------------------------------------------------------

def make_user(email: str = "ask-test@example.com") -> User:
    return User(
        id=uuid.uuid4(),
        email=email,
        password_hash="test-hash",
        is_active=True,
        created_at=datetime.now(UTC),
    )


def make_study_space(user_id: uuid.UUID) -> StudySpace:
    now = datetime.now(UTC)

    return StudySpace(
        id=uuid.uuid4(),
        user_id=user_id,
        name="Cloud Computing",
        description="Cloud computing test materials",
        created_at=now,
        updated_at=now,
    )


# -------------------------------------------------------------------
# In-memory database mock
# -------------------------------------------------------------------

class FakeQuery:
    def __init__(self, db, model):
        self.db = db
        self.model = model
        self.items = list(db.items_for(model))

    def filter(self, *conditions):
        filtered = []

        for item in self.items:
            matches = True

            for condition in conditions:
                column = condition.left.key
                expected = condition.right.value
                actual = getattr(item, column)

                if condition.operator.__name__ == "eq":
                    matches = actual == expected
                else:
                    raise NotImplementedError(
                        f"Unsupported query operator: {condition.operator}"
                    )

                if not matches:
                    break

            if matches:
                filtered.append(item)

        self.items = filtered
        return self

    def first(self):
        return self.items[0] if self.items else None


class FakeDB:
    def __init__(self, users=None, study_spaces=None):
        self.users = list(users or [])
        self.study_spaces = list(study_spaces or [])

    def items_for(self, model):
        if model is User:
            return self.users

        if model is StudySpace:
            return self.study_spaces

        return []

    def query(self, model):
        return FakeQuery(self, model)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def setup_api():
    user = make_user()
    study_space = make_study_space(user.id)

    fake_db = FakeDB(
        users=[user],
        study_spaces=[study_space],
    )

    def override_db():
        yield fake_db

    def override_user():
        return user

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user

    yield user, study_space, fake_db

    app.dependency_overrides.pop(get_db, None)
    app.dependency_overrides.pop(get_current_user, None)


# -------------------------------------------------------------------
# Authentication tests
# -------------------------------------------------------------------

def test_ask_requires_authentication():
    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "What is cloud computing?",
            "study_space_id": str(uuid.uuid4()),
            "session_id": "test-session",
            "limit": 5,
        },
    )

    assert response.status_code == 401


# -------------------------------------------------------------------
# Validation tests
# -------------------------------------------------------------------

def test_ask_rejects_empty_query(setup_api):
    _, study_space, _ = setup_api

    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "",
            "study_space_id": str(study_space.id),
            "session_id": "test-session",
            "limit": 5,
        },
    )

    assert response.status_code == 422


def test_ask_rejects_invalid_limit(setup_api):
    _, study_space, _ = setup_api

    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "What is cloud computing?",
            "study_space_id": str(study_space.id),
            "session_id": "test-session",
            "limit": 100,
        },
    )

    assert response.status_code == 422


# -------------------------------------------------------------------
# RAG response tests
# -------------------------------------------------------------------

def test_ask_returns_answer(monkeypatch, setup_api):
    _, study_space, _ = setup_api

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
            "study_space_id": str(study_space.id),
            "session_id": "test-session",
            "limit": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"].startswith("Cloud computing")
    assert data["session_id"] == "test-session"
    assert data["fallback_used"] is False
    assert data["model_used"] == "test-model"


# -------------------------------------------------------------------
# Study Space ownership tests
# -------------------------------------------------------------------

def test_ask_rejects_another_users_study_space(setup_api):
    _, _, fake_db = setup_api

    another_user = make_user("another-user@example.com")
    another_study_space = make_study_space(another_user.id)

    fake_db.users.append(another_user)
    fake_db.study_spaces.append(another_study_space)

    response = client.post(
        "/api/v1/ask/",
        json={
            "query": "What is cloud computing?",
            "study_space_id": str(another_study_space.id),
            "session_id": "test-session",
            "limit": 5,
        },
    )

    assert response.status_code == 404
