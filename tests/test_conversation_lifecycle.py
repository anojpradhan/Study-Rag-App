import uuid

import pytest

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
        if conversation is not None and study_space is not None:
            memory_service.clear_history(
                study_space_id=str(study_space.id),
                session_id=str(conversation.id),
            )

        if user is not None:
            db.delete(user)
            db.commit()

        db.close()


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

    # 3. Read the conversation history from PostgreSQL.
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
