import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_current_user
from app.core.security import create_access_token
from app.db.session import get_db
from app.main import app
from app.models.study_space import StudySpace
from app.models.user import User

client = TestClient(app)


# -------------------------------------------------------------------
# Test helpers
# -------------------------------------------------------------------


def make_user(
    email: str = "student@example.com",
    is_active: bool = True,
) -> User:
    return User(
        id=uuid.uuid4(),
        email=email,
        password_hash="test-hash",
        is_active=is_active,
        created_at=datetime.now(UTC),
    )


def make_study_space(
    user_id: uuid.UUID,
    name: str = "Machine Learning",
    description: str | None = "ML study notes",
) -> StudySpace:
    now = datetime.now(UTC)

    return StudySpace(
        id=uuid.uuid4(),
        user_id=user_id,
        name=name,
        description=description,
        created_at=now,
        updated_at=now,
    )


# -------------------------------------------------------------------
# In-memory database mock
# -------------------------------------------------------------------


class FakeQuery:
    """Supports the SQLAlchemy query operations used by Study Spaces."""

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

    def order_by(self, *expressions):
        # Ordering is not important for these CRUD tests.
        return self

    def first(self):
        return self.items[0] if self.items else None

    def all(self):
        return self.items


class FakeDB:
    """In-memory database mock for Study Spaces endpoint tests."""

    def __init__(self, users=None, study_spaces=None):
        self.users = list(users or [])
        self.study_spaces = list(study_spaces or [])
        self.deleted_spaces = []

    def items_for(self, model):
        if model is User:
            return self.users

        if model is StudySpace:
            return self.study_spaces

        return []

    def query(self, model):
        return FakeQuery(self, model)

    def get(self, model, item_id):
        return next(
            (item for item in self.items_for(model) if item.id == item_id),
            None,
        )

    def add(self, item):
        if isinstance(item, StudySpace):
            if item.id is None:
                item.id = uuid.uuid4()

            now = datetime.now(UTC)

            if item.created_at is None:
                item.created_at = now

            if item.updated_at is None:
                item.updated_at = now

            self.study_spaces.append(item)

        elif isinstance(item, User):
            self.users.append(item)

    def commit(self):
        pass

    def refresh(self, item):
        pass

    def delete(self, item):
        if isinstance(item, StudySpace) and item in self.study_spaces:
            self.study_spaces.remove(item)
            self.deleted_spaces.append(item)


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------


@pytest.fixture
def setup_api():
    user = make_user()
    fake_db = FakeDB(users=[user])

    def override_db():
        yield fake_db

    def override_user():
        return user

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_current_user] = override_user

    yield user, fake_db

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    user = make_user()
    token = create_access_token(subject=str(user.id))

    return {"Authorization": f"Bearer {token}"}


# -------------------------------------------------------------------
# Create tests
# -------------------------------------------------------------------


def test_create_study_space(setup_api):
    user, fake_db = setup_api

    response = client.post(
        "/api/v1/study-spaces/",
        json={
            "name": "Machine Learning",
            "description": "ML study notes",
        },
    )

    assert response.status_code == 201

    data = response.json()

    assert data["name"] == "Machine Learning"
    assert data["description"] == "ML study notes"
    assert data["user_id"] == str(user.id)
    assert "id" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert len(fake_db.study_spaces) == 1


def test_create_multiple_study_spaces(setup_api):
    _, fake_db = setup_api

    names = ["Machine Learning", "Cloud Computing", "Databases"]

    for name in names:
        response = client.post(
            "/api/v1/study-spaces/",
            json={"name": name, "description": None},
        )

        assert response.status_code == 201

    assert len(fake_db.study_spaces) == 3
    assert {space.name for space in fake_db.study_spaces} == set(names)


def test_create_study_space_rejects_blank_name(setup_api):
    response = client.post(
        "/api/v1/study-spaces/",
        json={"name": "   "},
    )

    assert response.status_code == 422


# -------------------------------------------------------------------
# List and retrieve tests
# -------------------------------------------------------------------


def test_list_only_current_users_study_spaces(setup_api):
    user, fake_db = setup_api
    other_user = make_user("other@example.com")

    fake_db.study_spaces.extend(
        [
            make_study_space(user.id, "My Space"),
            make_study_space(other_user.id, "Private Space"),
        ]
    )

    response = client.get("/api/v1/study-spaces/")

    assert response.status_code == 200

    data = response.json()

    assert len(data) == 1
    assert data[0]["name"] == "My Space"
    assert data[0]["user_id"] == str(user.id)


def test_get_study_space(setup_api):
    user, fake_db = setup_api
    space = make_study_space(user.id)

    fake_db.study_spaces.append(space)

    response = client.get(f"/api/v1/study-spaces/{space.id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(space.id)
    assert response.json()["name"] == space.name


def test_get_another_users_study_space_returns_404(setup_api):
    _, fake_db = setup_api
    other_user = make_user("other@example.com")
    space = make_study_space(other_user.id)

    fake_db.study_spaces.append(space)

    response = client.get(f"/api/v1/study-spaces/{space.id}")

    assert response.status_code == 404


# -------------------------------------------------------------------
# Update tests
# -------------------------------------------------------------------


def test_update_study_space(setup_api):
    user, fake_db = setup_api
    space = make_study_space(user.id)

    fake_db.study_spaces.append(space)

    response = client.patch(
        f"/api/v1/study-spaces/{space.id}",
        json={
            "name": "Deep Learning",
            "description": "Updated description",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["name"] == "Deep Learning"
    assert data["description"] == "Updated description"
    assert space.name == "Deep Learning"
    assert space.description == "Updated description"


def test_update_only_description(setup_api):
    user, fake_db = setup_api
    space = make_study_space(user.id)

    fake_db.study_spaces.append(space)

    response = client.patch(
        f"/api/v1/study-spaces/{space.id}",
        json={"description": "New description"},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "Machine Learning"
    assert response.json()["description"] == "New description"


def test_update_rejects_empty_payload(setup_api):
    user, fake_db = setup_api
    space = make_study_space(user.id)

    fake_db.study_spaces.append(space)

    response = client.patch(
        f"/api/v1/study-spaces/{space.id}",
        json={},
    )

    assert response.status_code == 422


def test_update_rejects_blank_name(setup_api):
    user, fake_db = setup_api
    space = make_study_space(user.id)

    fake_db.study_spaces.append(space)

    response = client.patch(
        f"/api/v1/study-spaces/{space.id}",
        json={"name": "   "},
    )

    assert response.status_code == 422


def test_update_another_users_space_returns_404(setup_api):
    _, fake_db = setup_api
    other_user = make_user("other@example.com")
    space = make_study_space(other_user.id)

    fake_db.study_spaces.append(space)

    response = client.patch(
        f"/api/v1/study-spaces/{space.id}",
        json={"name": "Unauthorized change"},
    )

    assert response.status_code == 404
    assert space.name == "Machine Learning"


# -------------------------------------------------------------------
# Delete tests
# -------------------------------------------------------------------


def test_delete_study_space(setup_api):
    user, fake_db = setup_api
    space = make_study_space(user.id)

    fake_db.study_spaces.append(space)

    response = client.delete(f"/api/v1/study-spaces/{space.id}")

    assert response.status_code == 204
    assert space not in fake_db.study_spaces
    assert space in fake_db.deleted_spaces

    get_response = client.get(f"/api/v1/study-spaces/{space.id}")

    assert get_response.status_code == 404


def test_delete_another_users_space_returns_404(setup_api):
    _, fake_db = setup_api
    other_user = make_user("other@example.com")
    space = make_study_space(other_user.id)

    fake_db.study_spaces.append(space)

    response = client.delete(f"/api/v1/study-spaces/{space.id}")

    assert response.status_code == 404
    assert space in fake_db.study_spaces


# -------------------------------------------------------------------
# Authentication and validation tests
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/v1/study-spaces/"),
        ("post", "/api/v1/study-spaces/"),
    ],
)
def test_study_spaces_requires_authentication(method, path):
    if method == "post":
        response = client.post(
            path,
            json={"name": "Unauthorized"},
        )
    else:
        response = client.get(path)

    assert response.status_code == 401


def test_invalid_study_space_id_returns_422(setup_api):
    response = client.get("/api/v1/study-spaces/not-a-uuid")

    assert response.status_code == 422
