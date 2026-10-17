from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "Study Rag"
    app_version: str = "1.0.0"
    debug: bool = True
    gemini_api_key: SecretStr | None = None
    gemini_primary_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    redis_url: str = "redis://localhost:6379/0"
    database_url: str
    jwt_secret_key: SecretStr
    access_token_expire_minutes:int= 30


settings = Settings()
