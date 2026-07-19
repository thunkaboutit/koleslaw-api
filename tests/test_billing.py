"""Stripe webhook (/v1/billing/webhook): signature auth + the two event handlers."""

import hashlib
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import AsyncClient
from pydantic import SecretStr

WHSEC = "whsec_test_secret"
USER_ID = uuid.uuid4()


def _signed_headers(payload: bytes, secret: str = WHSEC) -> dict[str, str]:
    ts = int(time.time())
    v1 = hmac.new(secret.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    return {"Stripe-Signature": f"t={ts},v1={v1}", "Content-Type": "application/json"}


def _event(event_type: str, obj: dict) -> bytes:
    return json.dumps(
        {
            "id": "evt_test_1",
            "object": "event",
            "api_version": "2026-01-01",
            "type": event_type,
            "data": {"object": obj},
        }
    ).encode()


def _install_db(app, user) -> AsyncMock:
    """DB factory whose queries resolve to `user`; returns the session mock."""
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=user)))

    @asynccontextmanager
    async def factory():
        yield session

    app.state.db_session_factory = factory
    return session


@pytest.fixture
def billing_app(app):
    app.state.settings.stripe_webhook_secret = SecretStr(WHSEC)
    return app


@pytest.mark.asyncio
async def test_unconfigured_returns_503(app, client: AsyncClient) -> None:
    payload = _event("checkout.session.completed", {})
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_bad_signature_returns_400(billing_app, client: AsyncClient) -> None:
    payload = _event("checkout.session.completed", {})
    response = await client.post(
        "/v1/billing/webhook", content=payload, headers=_signed_headers(payload, secret="whsec_wrong")
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_checkout_completed_upgrades_user(billing_app, client: AsyncClient) -> None:
    user = MagicMock(plan="free", stripe_customer_id=None, stripe_subscription_id=None)
    session = _install_db(billing_app, user)
    payload = _event(
        "checkout.session.completed",
        {"id": "cs_1", "client_reference_id": str(USER_ID), "customer": "cus_1", "subscription": "sub_1"},
    )
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert user.plan == "pro"
    assert user.stripe_customer_id == "cus_1"
    assert user.stripe_subscription_id == "sub_1"
    session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_checkout_without_reference_is_ignored(billing_app, client: AsyncClient) -> None:
    session = _install_db(billing_app, MagicMock())
    payload = _event("checkout.session.completed", {"id": "cs_1", "customer": "cus_1"})
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
    session.execute.assert_not_called()


@pytest.mark.asyncio
async def test_checkout_unknown_user_is_ignored(billing_app, client: AsyncClient) -> None:
    _install_db(billing_app, None)
    payload = _event(
        "checkout.session.completed",
        {"id": "cs_1", "client_reference_id": str(uuid.uuid4()), "customer": "cus_1", "subscription": "sub_1"},
    )
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}


@pytest.mark.asyncio
async def test_subscription_deleted_downgrades_user(billing_app, client: AsyncClient) -> None:
    user = MagicMock(plan="pro", stripe_subscription_id="sub_1")
    session = _install_db(billing_app, user)
    payload = _event("customer.subscription.deleted", {"id": "sub_1"})
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert user.plan == "free"
    assert user.stripe_subscription_id is None
    session.commit.assert_awaited()


@pytest.mark.asyncio
async def test_subscription_deleted_unknown_is_idempotent(billing_app, client: AsyncClient) -> None:
    """Re-delivery after a downgrade finds no matching user and stays a 200 no-op."""
    _install_db(billing_app, None)
    payload = _event("customer.subscription.deleted", {"id": "sub_gone"})
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}


@pytest.mark.asyncio
async def test_unhandled_event_is_acknowledged(billing_app, client: AsyncClient) -> None:
    payload = _event("invoice.paid", {"id": "in_1"})
    response = await client.post("/v1/billing/webhook", content=payload, headers=_signed_headers(payload))
    assert response.status_code == 200
    assert response.json() == {"status": "ignored"}
