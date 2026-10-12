from sqlalchemy import create_engine, text

from app.core.config import settings


def test_database_connection():
    engine = create_engine(settings.database_url)

    try:
        with engine.connect() as connection:
            database, user = connection.execute(
                text("SELECT current_database(), current_user")
            ).one()

        assert database == "study_rag"

        print("\nDatabase connection successful!")
        print("Database:", database)
        print("User:", user)

    finally:
        engine.dispose()
