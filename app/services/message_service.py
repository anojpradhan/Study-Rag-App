from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.conversation import Conversation
from app.models.message import Message


class MessageService:
    """Manage persistent conversation messages in PostgreSQL."""

    def get_history(
        self,
        db: Session,
        conversation_id: UUID,
        limit: int = 12,
    ) -> list[dict[str, str]]:
        """Return recent messages in chronological order."""

        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
            .all()
        )

        # Convert newest-first results into chronological order.
        messages.reverse()

        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in messages
        ]

    def save_exchange(
        self,
        db: Session,
        conversation_id: UUID,
        user_message: str,
        assistant_message: str,
        sources: list[dict] | None = None,
        model_used: str | None = None,
        fallback_used: bool = False,
    ) -> None:
        """Persist a complete user-assistant exchange."""

        conversation = db.get(Conversation, conversation_id)

        if conversation is None:
            raise ValueError("Conversation not found.")

        # Assign distinct timestamps to preserve message order.
        exchange_time = datetime.now(UTC)

        messages = [
            Message(
                conversation_id=conversation_id,
                role="user",
                content=user_message,
                created_at=exchange_time,
            ),
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_message,
                sources=sources,
                model_used=model_used,
                fallback_used=fallback_used,
                created_at=exchange_time + timedelta(microseconds=1),
            ),
        ]

        try:
            db.add_all(messages)

            conversation.updated_at = func.now()

            db.commit()

        except Exception:
            db.rollback()
            raise

    def delete_history(
        self,
        db: Session,
        conversation_id: UUID,
    ) -> None:
        """Delete all messages belonging to a conversation."""

        try:
            db.query(Message).filter(Message.conversation_id == conversation_id).delete(
                synchronize_session=False
            )

            db.commit()

        except Exception:
            db.rollback()
            raise
