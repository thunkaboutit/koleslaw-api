from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/prompt_enhance"
    redis_url: str = "redis://localhost:6379"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "ollama_chat/prompt-enhance-14b-think"
    environment: Literal["development", "staging", "production"] = "development"
    api_key_hash_secret: SecretStr = SecretStr("api_key_hash_secret_change_me_development")
    session_secret: SecretStr = SecretStr("session_secret_change_me_development")
    cors_origins: list[str] = ["*"]
    rate_limit_requests_per_minute: int = 60
    cache_ttl_seconds: int = 86400
    github_client_id: str = ""
    github_client_secret: SecretStr | None = None
    google_client_id: str = ""
    google_client_secret: SecretStr | None = None
    facebook_client_id: str = ""
    facebook_client_secret: SecretStr | None = None
    instagram_client_id: str = ""
    instagram_client_secret: SecretStr | None = None
    frontend_url: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
