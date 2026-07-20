"""Security T1-C — CORS must use an explicit origin allowlist, not reflect-any.

The old config (`allow_origin_regex=r".*"` + `allow_credentials=True`) echoed
any Origin back with credentials allowed. Combined with the cookie session
that's one SameSite change away from a cross-site credential hole, and it lets
any site read API responses. Fix: an explicit allowlist from settings, and the
error-response helper must honour the same list. See
docs/security-review-2026-07-19.md (T1-C).
"""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from koleslaw_api.config import Settings
from koleslaw_api.main import _cors_headers, configure_cors


def _app(origins: list[str]) -> FastAPI:
    app = FastAPI()
    settings = Settings(cors_origins=origins)
    app.state.settings = settings
    configure_cors(app, settings)

    @app.get("/ping")
    async def ping() -> dict[str, bool]:
        return {"ok": True}

    return app


@pytest.mark.asyncio
async def test_allowed_origin_gets_cors_headers() -> None:
    app = _app(["https://koleslaw.ai"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/ping", headers={"Origin": "https://koleslaw.ai"})
    assert resp.headers.get("access-control-allow-origin") == "https://koleslaw.ai"
    assert resp.headers.get("access-control-allow-credentials") == "true"


@pytest.mark.asyncio
async def test_disallowed_origin_is_not_reflected() -> None:
    app = _app(["https://koleslaw.ai"])
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.get("/ping", headers={"Origin": "https://evil.example"})
    # Starlette omits ACAO entirely for a disallowed origin — it must never be
    # reflected back.
    assert resp.headers.get("access-control-allow-origin") != "https://evil.example"


def test_default_cors_origins_is_an_explicit_allowlist() -> None:
    default = Settings.model_fields["cors_origins"].default
    assert default, "default must not be empty"
    assert "*" not in default, "default must not be a wildcard"


def test_cors_headers_helper_reflects_only_allowlisted_origins() -> None:
    settings = Settings(cors_origins=["https://koleslaw.ai"])

    allowed = MagicMock()
    allowed.headers = {"origin": "https://koleslaw.ai"}
    allowed.app.state.settings = settings
    headers = _cors_headers(allowed)
    assert headers["Access-Control-Allow-Origin"] == "https://koleslaw.ai"
    assert headers["Access-Control-Allow-Credentials"] == "true"

    denied = MagicMock()
    denied.headers = {"origin": "https://evil.example"}
    denied.app.state.settings = settings
    assert _cors_headers(denied) == {}
