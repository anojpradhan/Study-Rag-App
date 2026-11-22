import sys
from pathlib import Path

import redis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import settings


def reset_redis():
    client = redis.Redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )

    pattern = "study_rag:memory:*"
    deleted_count = 0

    for key in client.scan_iter(match=pattern):
        client.delete(key)
        deleted_count += 1

    print(f"Deleted {deleted_count} Redis conversation memory keys.")

    client.close()


if __name__ == "__main__":
    reset_redis()
