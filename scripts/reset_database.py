import sys
from pathlib import Path

from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.db.session import engine

TABLES = [
    "messages",
    "conversations",
    "study_documents",
    "study_spaces",
    "users",
]


def reset_database():
    with engine.begin() as connection:
        existing_tables = set(
            connection.execute(
                text(
                    """
                    SELECT tablename
                    FROM pg_tables
                    WHERE schemaname = 'public'
                    """
                )
            ).scalars()
        )

        tables_to_reset = [table for table in TABLES if table in existing_tables]

        if not tables_to_reset:
            print("No application tables found.")
            return

        table_names = ", ".join(f'"{table}"' for table in tables_to_reset)

        connection.execute(
            text(f"TRUNCATE TABLE {table_names} RESTART IDENTITY CASCADE")
        )

        print("Database reset completed.")
        print("Tables cleared:", ", ".join(tables_to_reset))


if __name__ == "__main__":
    reset_database()
