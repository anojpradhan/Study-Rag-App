import json

import redis

from app.core.config import settings


class MemoryService:
    """Store and retrieve recent conversation messages using Redis"""

    def __init__(
        self,
        ttl_seconds: int = 86400,
        max_messages: int = 12,
    ) -> None:
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def _get_key(self, study_space_id: str, session_id: str) -> str:
        return f"study_rag:memory:{study_space_id}: {session_id}"

    def get_history(self, study_space_id: str, session_id: str) -> list[dict[str, str]]:
        """Return recent messages in chronological order."""
        key = self._get_key(study_space_id, session_id)
        stored_messages = self.client.lrange(key, 0, -1)
        return [json.load(message) for message in stored_messages]

    def save_exchange(
        self,
        study_space_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Save one user/assistant exchange and limit stored history."""
        key = self._get_key(study_space_id, session_id)
        messages = [
            {
                "role": "user",
                "content": user_message,
            },
            {
                "role": "assistant",
                "content": assistant_message,
            },
        ]
        with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *[json.dumps(message) for message in messages])
            pipe.ltrim(key, -self.max_messages, -1)
            pipe.expire(key, self.ttl_seconds)
            pipe.execute()

    def clear_history(self, study_space_id: str, session_id: str) -> None:
        """Clear the history for one conversation"""

        key = self._get_key(study_space_id, session_id)
        self.client.delete(key)
