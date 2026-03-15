import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost:5432/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("API_KEY_HASH_SECRET", "test-secret")
os.environ.setdefault("SESSION_SECRET", "test-session-secret")
os.environ.setdefault("ENVIRONMENT", "development")

from prompt_enhance.config import Settings, get_settings  # noqa: E402
from prompt_enhance.services.cache import PromptCache  # noqa: E402
from prompt_enhance.services.llm import LLMService  # noqa: E402


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="postgresql+asyncpg://localhost:5432/test",
        redis_url="redis://localhost:6379/0",
        api_key_hash_secret=SecretStr("test-secret"),
        session_secret=SecretStr("test-session-secret"),
        environment="development",
        default_model="ollama_chat/prompt-enhance-14b-think",
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.ping = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.sismember = AsyncMock(return_value=False)
    redis.aclose = AsyncMock()
    return redis


@pytest.fixture
def mock_db_session_factory() -> MagicMock:
    session = AsyncMock()
    session.execute = AsyncMock(return_value=None)

    @asynccontextmanager
    async def factory():
        yield session

    return factory


@pytest.fixture
def mock_llm_service(settings: Settings) -> LLMService:
    return LLMService(settings)


@pytest.fixture
async def client(
    settings: Settings,
    mock_redis: AsyncMock,
    mock_db_session_factory: MagicMock,
    mock_llm_service: LLMService,
) -> AsyncIterator[AsyncClient]:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from prompt_enhance.routes import enhance, health, models
    from prompt_enhance.schemas.enhance import ErrorResponse
    from prompt_enhance.services.auth import get_current_api_key
    from prompt_enhance.services.llm import LLMServiceError

    @asynccontextmanager
    async def noop_lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield

    app = FastAPI(lifespan=noop_lifespan)

    app.state.settings = settings
    app.state.redis = mock_redis
    app.state.db_session_factory = mock_db_session_factory
    app.state.db_engine = AsyncMock()
    app.state.llm_service = mock_llm_service
    app.state.cache = PromptCache(mock_redis)

    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(enhance.router)

    app.dependency_overrides[get_current_api_key] = lambda: None
    app.dependency_overrides[get_settings] = lambda: settings

    @app.exception_handler(LLMServiceError)
    async def llm_error_handler(_request, exc: LLMServiceError) -> JSONResponse:
        status_map = {
            "authentication_error": 502,
            "rate_limit_error": 502,
            "bad_request": 400,
            "model_not_found": 404,
            "connection_error": 502,
            "service_unavailable": 503,
        }
        status_code = status_map.get(exc.code, 500)
        body = ErrorResponse(error=str(exc), code=exc.code, detail=exc.detail)
        return JSONResponse(status_code=status_code, content=body.model_dump())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
