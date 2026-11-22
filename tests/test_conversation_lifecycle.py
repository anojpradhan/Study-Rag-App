import uuid

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import conversations as conversations_endpoint
from app.api.v1.endpoints.conversations import get_owned_conversation
from app.db.session import SessionLocal
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.study_space import StudySpace
from app.models.user import User
from app.services.memory_service import MemoryService
from app.services.message_service import MessageService


@pytest.fixture
def lifecycle_records():
    db = SessionLocal()
    memory_service = MemoryService()

    user = None
    study_space = None
    conversation = None

    try:
        # Create an isolated user.
        user = User(
            email=f"lifecycle-{uuid.uuid4()}@example.com",
            password_hash="test-password-hash",
        )
        db.add(user)
        db.flush()

        # Create a Study Space owned by this user.
        study_space = StudySpace(
            user_id=user.id,
            name="Lifecycle Test Space",
            description="Temporary integration test data",
        )
        db.add(study_space)
        db.flush()

        # Create a conversation inside the Study Space.
        conversation = Conversation(
            user_id=user.id,
            study_space_id=study_space.id,
            title="Lifecycle Test Conversation",
        )
        db.add(conversation)
        db.commit()

        yield db, user, study_space, conversation, memory_service

    finally:
        # Clear Redis history if the test created a conversation.
        if conversation is not None and study_space is not None:
            memory_service.clear_history(
                study_space_id=str(study_space.id),
                session_id=str(conversation.id),
            )

        # Delete the test user if it still exists.
        if user is not None:
            existing_user = db.get(User, user.id)

            if existing_user is not None:
                db.delete(existing_user)
                db.commit()

        db.close()


def save_test_exchange(db, conversation_id):
    """Create a test exchange in PostgreSQL."""

    message_service = MessageService()

    message_service.save_exchange(
        db=db,
        conversation_id=conversation_id,
        user_message="What is machine learning?",
        assistant_message="Machine learning is a subset of AI.",
        sources=[],
        model_used="test-model",
        fallback_used=False,
    )


def test_message_persistence_and_redis_restoration(lifecycle_records):
    db, user, study_space, conversation, memory_service = lifecycle_records

    message_service = MessageService()

    user_message = "What is machine learning?"
    assistant_message = "Machine learning is a subset of AI."

    # 1. Save the exchange to PostgreSQL.
    message_service.save_exchange(
        db=db,
        conversation_id=conversation.id,
        user_message=user_message,
        assistant_message=assistant_message,
        sources=[],
        model_used="test-model",
        fallback_used=False,
    )

    # 2. Verify that both messages exist in PostgreSQL.
    saved_messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.created_at, Message.id)
        .all()
    )

    assert len(saved_messages) == 2

    messages_by_role = {message.role: message.content for message in saved_messages}

    assert messages_by_role == {
        "user": user_message,
        "assistant": assistant_message,
    }

    # 3. Read conversation history from PostgreSQL.
    history = message_service.get_history(
        db=db,
        conversation_id=conversation.id,
        limit=12,
    )

    assert history == [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]

    # 4. Clear Redis to simulate an expired or missing cache.
    memory_service.clear_history(
        study_space_id=str(study_space.id),
        session_id=str(conversation.id),
    )

    assert (
        memory_service.get_history(
            study_space_id=str(study_space.id),
            session_id=str(conversation.id),
        )
        == []
    )

    # 5. Restore Redis history from PostgreSQL.
    memory_service.restore_history(
        study_space_id=str(study_space.id),
        session_id=str(conversation.id),
        messages=history,
    )

    restored_history = memory_service.get_history(
        study_space_id=str(study_space.id),
        session_id=str(conversation.id),
    )

    assert restored_history == history


def test_conversation_ownership_isolation(lifecycle_records):
    db, owner, study_space, conversation, memory_service = lifecycle_records

    # Create another user.
    another_user = User(
        email=f"another-{uuid.uuid4()}@example.com",
        password_hash="test-password-hash",
    )

    try:
        db.add(another_user)
        db.commit()
        db.refresh(another_user)

        # 1. The owner can access their conversation.
        owned_conversation = get_owned_conversation(
            conversation_id=conversation.id,
            user_id=owner.id,
            db=db,
        )

        assert owned_conversation.id == conversation.id

        # 2. Another user cannot access the conversation.
        with pytest.raises(HTTPException) as exc_info:
            get_owned_conversation(
                conversation_id=conversation.id,
                user_id=another_user.id,
                db=db,
            )

        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Conversation not found."

        # 3. A nonexistent conversation also returns 404.
        with pytest.raises(HTTPException) as exc_info:
            get_owned_conversation(
                conversation_id=uuid.uuid4(),
                user_id=owner.id,
                db=db,
            )

        assert exc_info.value.status_code == 404

    finally:
        db.delete(another_user)
        db.commit()


def test_conversation_deletion_cascades_messages_and_clears_redis(
    lifecycle_records,
    monkeypatch,
):
    db, user, study_space, conversation, memory_service = lifecycle_records

    conversation_id = conversation.id
    study_space_id = str(study_space.id)
    conversation_key = str(conversation_id)

    # 1. Save messages to PostgreSQL.
    save_test_exchange(db, conversation_id)

    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count()
        == 2
    )

    # 2. Save conversation history to Redis.
    memory_service.save_exchange(
        study_space_id=study_space_id,
        session_id=conversation_key,
        user_message="What is machine learning?",
        assistant_message="Machine learning is a subset of AI.",
    )

    assert (
        len(
            memory_service.get_history(
                study_space_id=study_space_id,
                session_id=conversation_key,
            )
        )
        == 2
    )

    # 3. Use the actual endpoint with the test Redis service.
    monkeypatch.setattr(
        conversations_endpoint,
        "memory_service",
        memory_service,
    )

    response = conversations_endpoint.delete_conversation(
        conversation_id=conversation_id,
        db=db,
        current_user=user,
    )

    assert response.status_code == 204

    # 4. Verify that the conversation is deleted.
    db.expire_all()

    assert db.get(Conversation, conversation_id) is None

    # 5. Verify that its messages were deleted through cascade.
    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count()
        == 0
    )

    # 6. Verify that Redis history was cleared.
    assert (
        memory_service.get_history(
            study_space_id=study_space_id,
            session_id=conversation_key,
        )
        == []
    )


def test_study_space_deletion_cascades_conversations_and_messages(
    lifecycle_records,
):
    db, user, study_space, conversation, memory_service = lifecycle_records

    study_space_id = study_space.id
    conversation_id = conversation.id

    # 1. Save messages to PostgreSQL.
    save_test_exchange(db, conversation_id)

    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count()
        == 2
    )

    # 2. Delete the Study Space.
    db.delete(study_space)
    db.commit()

    db.expire_all()

    # 3. Verify the Study Space was deleted.
    assert db.get(StudySpace, study_space_id) is None

    # 4. Verify its conversations were deleted.
    assert db.get(Conversation, conversation_id) is None

    # 5. Verify its messages were deleted.
    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count()
        == 0
    )


def test_user_deletion_cascades_study_spaces_conversations_and_messages(
    lifecycle_records,
):
    db, user, study_space, conversation, memory_service = lifecycle_records

    user_id = user.id
    study_space_id = study_space.id
    conversation_id = conversation.id

    # 1. Save messages to PostgreSQL.
    save_test_exchange(db, conversation_id)

    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count()
        == 2
    )

    # 2. Delete the user.
    db.delete(user)
    db.commit()

    db.expire_all()

    # 3. Verify the user was deleted.
    assert db.get(User, user_id) is None

    # 4. Verify the user's Study Spaces were deleted.
    assert db.get(StudySpace, study_space_id) is None

    # 5. Verify the user's conversations were deleted.
    assert db.get(Conversation, conversation_id) is None

    # 6. Verify the conversation's messages were deleted.
    assert (
        db.query(Message).filter(Message.conversation_id == conversation_id).count()
        == 0
    )
