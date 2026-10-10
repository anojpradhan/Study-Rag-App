from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Study Rag"
    app_version: str = "1.0.0"
    debug: bool = True
    gemini_api_key: SecretStr | None = None
    gemini_primary_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"

    class Config:
        env_file = ".env"


settings = Settings()
