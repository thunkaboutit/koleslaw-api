from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    redis_url: str
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    default_model: str = "claude-haiku-4-5-20250929"
    environment: Literal["development", "staging", "production"] = "development"
    api_key_hash_secret: SecretStr
    cors_origins: list[str] = ["*"]
    rate_limit_requests_per_minute: int = 60
    cache_ttl_seconds: int = 86400


@lru_cache
def get_settings() -> Settings:
    return Settings()
