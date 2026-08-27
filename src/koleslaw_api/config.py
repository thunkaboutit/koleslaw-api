from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/koleslaw_api"
    redis_url: str = "redis://localhost:6379"
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "ollama_chat/prompt-enhance-14b-think"
    fallback_model: str | None = None
    # Models a caller may select via the request `model` field. default_model
    # is always allowed; this adds any extra *served* local tags (e.g. during a
    # model-roll overlap). Cloud/fallback models are deliberately absent — they
    # are server-internal fallback only, never caller-selectable, so a caller
    # can't route requests to a paid provider on our keys (security review T1-A).
    allowed_models: list[str] = []
    # Queue shaping: explicit ceilings on LLM calls so a request queued behind
    # a busy GPU spills to the fallback model instead of hanging (LiteLLM's
    # default request timeout is 6000s). Both are read-based, not total-duration:
    # the stream value caps time-to-first-token (an active stream is never cut
    # mid-generation), the sync value caps queue wait + the whole generation —
    # keep it under the 100s proxy idle wall the sync endpoint lives behind.
    llm_stream_timeout_seconds: float = 30.0
    llm_sync_timeout_seconds: float = 90.0
    environment: Literal["development", "staging", "production"] = "development"
    api_key_hash_secret: SecretStr = SecretStr("api_key_hash_secret_change_me_development")
    session_secret: SecretStr = SecretStr("session_secret_change_me_development")
    # Explicit CORS allowlist — never a wildcard with credentials (T1-C). The
    # portal is same-origin so needs no CORS; these cover any cross-origin
    # browser client. Override via CORS_ORIGINS (JSON list) per environment.
    cors_origins: list[str] = ["https://koleslaw.ai", "https://www.koleslaw.ai"]
    rate_limit_requests_per_minute: int = 60
    # Per-client-IP daily cap on LLM endpoints for anonymous keys (0 disables).
    # Keys are matched by sha256 hash; the default is the portal playground key
    # baked into the web bundle (seeded by kowpen scripts/seed-db.sql).
    anon_daily_limit: int = 10
    anon_key_hashes: list[str] = ["cce87702a65c5cc49688ce812951ba585418eadfd7d2d93941ab37ac9b6b48a0"]
    # Per-key daily caps on LLM endpoints by plan (0 disables); see
    # docs stripe-billing-plan.md in the workspace for the tier design
    free_daily_limit: int = 50
    pro_daily_limit: int = 500
    # Billing (stripe-billing-plan.md). Empty/None = webhook disabled (503)
    stripe_secret_key: SecretStr | None = None
    stripe_webhook_secret: SecretStr | None = None
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
    # Host header allowlist for the MCP endpoint's DNS-rebinding protection.
    # MCP clients are not browsers and send no Origin, so an Origin allowlist
    # stays empty on purpose: absent passes, anything present is rejected.
    # `host:*` matches any port.
    mcp_allowed_hosts: list[str] = ["api.koleslaw.ai", "localhost:*", "127.0.0.1:*"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
