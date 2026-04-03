import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from prompt_enhance.services.usage import log_usage


@pytest.fixture
def api_key_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def mock_db_session_factory_with_add():
    session = AsyncMock()
    added_objects = []

    def track_add(obj):
        added_objects.append(obj)

    session.add = track_add
    session.commit = AsyncMock()

    @asynccontextmanager
    async def factory():
        yield session

    factory.added_objects = added_objects
    factory.session = session
    return factory


async def test_log_usage_inserts_row(api_key_id, mock_db_session_factory_with_add):
    await log_usage(
        mock_db_session_factory_with_add,
        api_key_id,
        model="ollama_chat/test",
        enhancement_type="expand",
        input_tokens=100,
        output_tokens=200,
        cached=False,
        latency_ms=500,
    )

    added = mock_db_session_factory_with_add.added_objects
    assert len(added) == 1

    row = added[0]
    assert row.api_key_id == api_key_id
    assert row.model == "ollama_chat/test"
    assert row.enhancement_type == "expand"
    assert row.input_tokens == 100
    assert row.output_tokens == 200
    assert row.cached is False
    assert row.latency_ms == 500

    mock_db_session_factory_with_add.session.commit.assert_awaited_once()


async def test_log_usage_skips_when_no_api_key(mock_db_session_factory_with_add):
    await log_usage(
        mock_db_session_factory_with_add,
        None,
        model="ollama_chat/test",
        enhancement_type="chat",
        input_tokens=0,
        output_tokens=0,
        cached=False,
        latency_ms=0,
    )

    assert len(mock_db_session_factory_with_add.added_objects) == 0


async def test_log_usage_swallows_exceptions():
    @asynccontextmanager
    async def failing_factory():
        raise RuntimeError("DB is down")
        yield

    # Should not raise
    await log_usage(
        failing_factory,
        uuid.uuid4(),
        model="ollama_chat/test",
        enhancement_type="chat",
        input_tokens=0,
        output_tokens=0,
        cached=False,
        latency_ms=0,
    )
