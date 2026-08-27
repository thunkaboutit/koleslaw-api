"""The remote MCP endpoint — the second front door onto /v1/enhance/sync.

These tests drive the mounted ASGI app with raw JSON-RPC rather than an MCP
client, so a protocol-level regression (a tool that stops being listed, a
schema that grows a field) fails here rather than in someone's editor.

Unlike the REST tests, nothing overrides `get_current_api_key` — the MCP door
runs the real auth chain, which is exactly what these assert.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from koleslaw_api.config import Settings
from koleslaw_api.mcp_server import MCP_PATH, TOOL_NAME, mount_mcp_server
from koleslaw_api.services.auth import hash_api_key

# The portal/extension public key, same string test_anon_cap.py pins. Its
# sha256 is the default entry in Settings.anon_key_hashes.
PUBLIC_KEY = "pk_live_fNuYW07Bf16zLFLbghxSPERYV4QS1SWF4-OuLJ5JcCE"

RPC_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _fake_completion() -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "Enhanced prompt result"
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 20
    return resp


def _arm_pipeline(mock_redis: AsyncMock, *, daily: int = 1, rate: int = 1) -> MagicMock:
    """Arm redis.pipeline() for both counters the auth chain runs.

    check_daily_cap reads results[0] (its INCR); check_rate_limit reads
    results[2] (its ZCARD), so one return value serves both.
    """
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[daily, True, rate, True])
    mock_redis.pipeline = MagicMock(return_value=pipe)
    return pipe


def _key_row(key: str, *, plan: str) -> MagicMock:
    """A (ApiKey, plan) row shaped like the auth chain's SELECT returns."""
    api_key = MagicMock(id=uuid.uuid4(), key_hash=hash_api_key(key), is_active=True, rate_limit=60)
    result = MagicMock()
    result.one_or_none.return_value = (api_key, plan)
    return result


@asynccontextmanager
async def _harness(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Mount the MCP endpoint on the test app and drive it over ASGI.

    ASGITransport does not run lifespan, so the session manager — which the
    mount needs and which main.py starts from the app lifespan — is started
    here by hand.
    """
    server = mount_mcp_server(app)
    async with server.session_manager.run():
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def _rpc(client: AsyncClient, method: str, params: dict | None = None, **headers: str) -> Response:
    body: dict = {"jsonrpc": "2.0", "id": 1, "method": method}
    if params is not None:
        body["params"] = params
    return await client.post(f"{MCP_PATH}/", json=body, headers={**RPC_HEADERS, **headers})


async def _call_tool(client: AsyncClient, arguments: dict, **headers: str) -> dict:
    response = await _rpc(client, "tools/call", {"name": TOOL_NAME, "arguments": arguments}, **headers)
    assert response.status_code == 200, response.text
    return response.json()["result"]


def _error_text(result: dict) -> str:
    assert result["isError"] is True, result
    return " ".join(block["text"] for block in result["content"])


# --- surface -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_lists_exactly_one_tool(app: FastAPI, settings: Settings) -> None:
    """Every tool definition is permanent context tax on every session, so the
    endpoint exposes enhance_prompt and nothing else."""
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        response = await _rpc(client, "tools/list", {})

    tools = response.json()["result"]["tools"]
    assert [tool["name"] for tool in tools] == [TOOL_NAME]


@pytest.mark.asyncio
async def test_tool_does_not_expose_model_selection(app: FastAPI, settings: Settings) -> None:
    """T1-A: callers may not steer the model. The router picks, and a cloud
    model is server-internal fallback only."""
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        response = await _rpc(client, "tools/list", {})

    (tool,) = response.json()["result"]["tools"]
    assert set(tool["inputSchema"]["properties"]) == {"prompt", "context", "enhancement_type"}


# --- happy path --------------------------------------------------------------


@pytest.mark.asyncio
async def test_enhances_a_prompt(app: FastAPI, settings: Settings) -> None:
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_completion()
            result = await _call_tool(client, {"prompt": "Write a poem", "enhancement_type": "refine"})

    assert result["isError"] is False
    assert result["structuredContent"]["enhanced_prompt"] == "Enhanced prompt result"
    assert result["structuredContent"]["model_used"] == "ollama_chat/prompt-enhance-14b-think"
    assert result["structuredContent"]["cached"] is False


@pytest.mark.asyncio
async def test_serves_a_cache_hit_without_calling_the_model(
    app: FastAPI, settings: Settings, mock_redis: AsyncMock
) -> None:
    """The MCP door shares PromptCache with the REST door — same key, same hit."""
    settings.mcp_allowed_hosts = ["test"]
    mock_redis.get = AsyncMock(return_value="cached result")
    async with _harness(app) as client:
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            result = await _call_tool(client, {"prompt": "Write a poem"})

    mock_llm.assert_not_called()
    assert result["structuredContent"]["enhanced_prompt"] == "cached result"
    assert result["structuredContent"]["cached"] is True


@pytest.mark.asyncio
async def test_context_reaches_the_model(app: FastAPI, settings: Settings) -> None:
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = _fake_completion()
            await _call_tool(client, {"prompt": "Write a poem", "context": "for a wedding toast"})

    (message,) = mock_llm.call_args.kwargs["messages"]
    assert "for a wedding toast" in message["content"]


# --- error taxonomy ----------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_key_is_an_auth_error(app: FastAPI, settings: Settings) -> None:
    settings.mcp_allowed_hosts = ["test"]
    settings.environment = "production"
    async with _harness(app) as client:
        result = await _call_tool(client, {"prompt": "Write a poem"})

    assert "auth:" in _error_text(result)


@pytest.mark.asyncio
async def test_anon_key_over_daily_cap_is_a_rate_limit_error(
    app: FastAPI, settings: Settings, mock_redis: AsyncMock, mock_db_session_factory: MagicMock
) -> None:
    """Anonymous keys stay IP-capped through this door too — no new limiter."""
    settings.mcp_allowed_hosts = ["test"]
    _arm_pipeline(mock_redis, daily=settings.anon_daily_limit + 1)
    async with mock_db_session_factory() as session:
        session.execute = AsyncMock(return_value=_key_row(PUBLIC_KEY, plan="anonymous"))

    async with _harness(app) as client:
        result = await _call_tool(
            client,
            {"prompt": "Write a poem"},
            Authorization=f"Bearer {PUBLIC_KEY}",
        )

    text = _error_text(result)
    assert "rate_limit:" in text
    assert "Daily free limit" in text


@pytest.mark.asyncio
async def test_backend_failure_is_a_backend_unavailable_error(app: FastAPI, settings: Settings) -> None:
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        with patch("koleslaw_api.services.llm.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = RuntimeError("GPU is gone")
            result = await _call_tool(client, {"prompt": "Write a poem"})

    assert "backend_unavailable:" in _error_text(result)


@pytest.mark.asyncio
async def test_empty_prompt_is_an_invalid_request_error(app: FastAPI, settings: Settings) -> None:
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        result = await _call_tool(client, {"prompt": ""})

    assert "invalid_request:" in _error_text(result)


@pytest.mark.asyncio
async def test_oversized_prompt_is_an_invalid_request_error(app: FastAPI, settings: Settings) -> None:
    """The same Pydantic bounds the REST body enforces (T2-E), not a second set."""
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        result = await _call_tool(client, {"prompt": "x" * 50_001})

    assert "invalid_request:" in _error_text(result)


@pytest.mark.asyncio
async def test_unexpected_failure_is_reported_as_backend_unavailable(app: FastAPI, settings: Settings) -> None:
    """An unreachable Redis or Postgres still gets the caller an actionable
    code; the detail stays in the logs (T3-I)."""
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        with patch("koleslaw_api.mcp_server.enhance_once", new_callable=AsyncMock) as mock_enhance:
            mock_enhance.side_effect = RuntimeError("connection string postgres://user:pw@host")
            result = await _call_tool(client, {"prompt": "Write a poem"})

    text = _error_text(result)
    assert "backend_unavailable:" in text
    assert "postgres://" not in text
    assert "pw" not in text


# --- transport security ------------------------------------------------------


@pytest.mark.asyncio
async def test_rejects_an_unlisted_host(app: FastAPI, settings: Settings) -> None:
    """DNS-rebinding protection: the endpoint answers only on hosts we serve."""
    settings.mcp_allowed_hosts = ["api.koleslaw.ai"]
    async with _harness(app) as client:
        response = await _rpc(client, "tools/list", {})

    assert response.status_code == 421


@pytest.mark.asyncio
async def test_rejects_a_browser_origin(app: FastAPI, settings: Settings) -> None:
    """MCP clients send no Origin; a page that does is not a client of ours."""
    settings.mcp_allowed_hosts = ["test"]
    async with _harness(app) as client:
        response = await _rpc(client, "tools/list", {}, Origin="https://evil.example")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_the_slashed_url_is_the_one_that_serves_directly(app: FastAPI, settings: Settings) -> None:
    """Clients are configured with the trailing slash on purpose.

    The bare path gets a 307 from Starlette's router, which a client that does
    not follow redirects reads as the response and rejects. Pinned here so the
    documented URL and the mount cannot drift apart.
    """
    settings.mcp_allowed_hosts = ["test"]
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    async with _harness(app) as client:
        bare = await client.post(MCP_PATH, json=body, headers=RPC_HEADERS)
        slashed = await client.post(f"{MCP_PATH}/", json=body, headers=RPC_HEADERS)

    assert bare.status_code == 307
    assert slashed.status_code == 200
