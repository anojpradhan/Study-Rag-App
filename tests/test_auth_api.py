import uuid
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from app.core.security import create_access_token, hash_password
from app.db.session import get_db
from app.main import app
from app.models.user import User

client = TestClient(app)


def make_user(
    email: str = "student@example.com",
    password: str = "StudyRAG_Test_123",
    is_active: bool = True,
) -> User:
    return User(
        id=uuid.uuid4(),
        email=email,
        password_hash=hash_password(password),
        is_active=is_active,
        created_at=datetime.now(UTC),
    )


class FakeDB:
    """Minimal database mock for authentication endpoint tests."""

    def __init__(self, user: User | None = None):
        self.user = user
        self.added_user = None

    def scalar(self, query):
        # Used by registration to check for duplicate emails
        # and by login to find an existing account.
        return self.user

    def add(self, user: User):
        self.added_user = user
        self.user = user

        if user.id is None:
            user.id = uuid.uuid4()

        if user.created_at is None:
            user.created_at = datetime.now(UTC)

    def commit(self):
        pass

    def refresh(self, user: User):
        pass

    def get(self, model, user_id):
        if model is User and self.user is not None:
            if self.user.id == user_id:
                return self.user
        return None


def override_database(fake_db: FakeDB):
    def dependency():
        yield fake_db

    return dependency


def test_register_user():
    fake_db = FakeDB()
    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": "student@example.com",
                "password": "StudyRAG_Test_123",
            },
        )

        assert response.status_code == 201

        data = response.json()
        assert data["email"] == "student@example.com"
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data
        assert "password" not in data
        assert "password_hash" not in data

        # Ensure the password is hashed, not stored as plaintext.
        assert fake_db.added_user is not None
        assert fake_db.added_user.password_hash != "StudyRAG_Test_123"

    finally:
        app.dependency_overrides.clear()


def test_register_rejects_duplicate_email():
    existing_user = make_user()
    fake_db = FakeDB(user=existing_user)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.post(
            "/api/v1/auth/register",
            json={
                "email": existing_user.email,
                "password": "StudyRAG_Test_123",
            },
        )

        assert response.status_code == 409

    finally:
        app.dependency_overrides.clear()


def test_register_rejects_invalid_email():
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "not-an-email",
            "password": "StudyRAG_Test_123",
        },
    )

    assert response.status_code == 422


def test_register_rejects_short_password():
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "student@example.com",
            "password": "short",
        },
    )

    assert response.status_code == 422


def test_login_returns_access_token():
    user = make_user()
    fake_db = FakeDB(user=user)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": user.email,
                "password": "StudyRAG_Test_123",
            },
        )

        assert response.status_code == 200

        data = response.json()
        assert isinstance(data["access_token"], str)
        assert len(data["access_token"]) > 20
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    finally:
        app.dependency_overrides.clear()


def test_login_rejects_wrong_password():
    user = make_user()
    fake_db = FakeDB(user=user)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": user.email,
                "password": "WrongPassword_123",
            },
        )

        assert response.status_code == 401

    finally:
        app.dependency_overrides.clear()


def test_login_rejects_unknown_email():
    fake_db = FakeDB(user=None)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": "unknown@example.com",
                "password": "StudyRAG_Test_123",
            },
        )

        assert response.status_code == 401

    finally:
        app.dependency_overrides.clear()


def test_login_rejects_inactive_user():
    user = make_user(is_active=False)
    fake_db = FakeDB(user=user)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.post(
            "/api/v1/auth/login",
            json={
                "email": user.email,
                "password": "StudyRAG_Test_123",
            },
        )

        assert response.status_code == 403

    finally:
        app.dependency_overrides.clear()


def test_me_rejects_missing_token():
    response = client.get("/api/v1/auth/me")

    assert response.status_code == 401


def test_me_accepts_valid_token():
    user = make_user()
    token = create_access_token(subject=str(user.id))
    fake_db = FakeDB(user=user)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["email"] == user.email
        assert data["id"] == str(user.id)
        assert data["is_active"] is True
        assert "password_hash" not in data

    finally:
        app.dependency_overrides.clear()


def test_me_rejects_invalid_token():
    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": "Bearer invalid.token.value"},
    )

    assert response.status_code == 401


def test_me_rejects_inactive_user():
    user = make_user(is_active=False)
    token = create_access_token(subject=str(user.id))
    fake_db = FakeDB(user=user)

    app.dependency_overrides[get_db] = override_database(fake_db)

    try:
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 403

    finally:
        app.dependency_overrides.clear()
