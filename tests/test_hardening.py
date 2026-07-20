"""Security T3 hygiene — log redaction, error-body minimisation, dev-bypass guard.

- T3-H: the DB URL (with password) was logged at startup → CloudWatch.
- T3-I: the global exception handler returned the exception class name.
- T3-K: ENVIRONMENT defaults to 'development', which bypasses API-key auth;
  running that against a production database must fail fast.
See docs/security-review-2026-07-19.md.
"""

import json
from unittest.mock import MagicMock

import pytest

from koleslaw_api.config import Settings
from koleslaw_api.main import _assert_safe_environment, _redact_dsn, global_error_handler

# ── T3-H ─────────────────────────────────────────────────────────────────


def test_redact_dsn_hides_password_keeps_host() -> None:
    red = _redact_dsn("postgresql+asyncpg://app:s3cr3t@db.abc.us-east-2.rds.amazonaws.com:5432/appdb")
    assert "s3cr3t" not in red
    assert "db.abc.us-east-2.rds.amazonaws.com" in red  # host stays for debugging
    assert red.startswith("postgresql+asyncpg://")


def test_redact_dsn_passwordless_url_unchanged() -> None:
    assert _redact_dsn("redis://localhost:6379") == "redis://localhost:6379"


# ── T3-I ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_global_error_handler_leaks_no_internals() -> None:
    req = MagicMock()
    req.headers = {}
    req.app.state.settings = Settings()
    resp = await global_error_handler(req, ValueError("secret internal detail"))
    body = json.loads(resp.body)
    assert resp.status_code == 500
    dumped = json.dumps(body)
    assert "ValueError" not in dumped  # no exception class name
    assert "secret internal detail" not in dumped  # no exception message


# ── T3-K ─────────────────────────────────────────────────────────────────


def _settings(env: str, host: str) -> Settings:
    return Settings(
        environment=env,
        database_url=f"postgresql+asyncpg://app:pw@{host}:5432/appdb",
    )


def test_dev_against_cloud_db_refuses_to_start() -> None:
    with pytest.raises(RuntimeError):
        _assert_safe_environment(_settings("development", "koleslaw-prod-db.abc.us-east-2.rds.amazonaws.com"))


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "postgres", "db", "host.docker.internal"])
def test_dev_against_local_db_is_allowed(host: str) -> None:
    _assert_safe_environment(_settings("development", host))  # must not raise


def test_production_is_allowed_against_cloud_db() -> None:
    _assert_safe_environment(_settings("production", "koleslaw-prod-db.abc.us-east-2.rds.amazonaws.com"))
