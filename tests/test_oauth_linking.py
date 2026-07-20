"""Security T2-D — OAuth account linking must not trust an unverified email.

`_upsert_oauth_user` links a sign-in to an existing account by email when a
returning user arrives via a different provider. If the incoming email isn't
provider-verified, an attacker who sets their provider's primary email to a
victim's address (unverified) could be merged into the victim's account.

Policy (chosen 2026-07-19): link across providers ONLY when the provider
asserts the email is verified; otherwise create a separate account — and if
that unverified email already belongs to someone, refuse rather than 500 or
link. See docs/security-review-2026-07-19.md (T2-D).
"""

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from koleslaw_api.db.models import User
from koleslaw_api.routes.auth import _upsert_oauth_user


def _session_returning(*scalar_results: object) -> AsyncMock:
    """A mock session whose successive execute().scalar_one_or_none() calls
    return the given values in order."""
    session = AsyncMock()
    results = []
    for val in scalar_results:
        r = MagicMock()
        r.scalar_one_or_none = MagicMock(return_value=val)
        results.append(r)
    session.execute = AsyncMock(side_effect=results)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


def _factory(session: AsyncMock):
    @asynccontextmanager
    async def factory():
        yield session

    return factory


def _user(email: str, provider: str, oauth_id: str) -> User:
    return User(id=uuid.uuid4(), email=email, name="Existing", oauth_provider=provider, oauth_id=oauth_id)


@pytest.mark.asyncio
async def test_unverified_email_does_not_link_into_existing_account() -> None:
    """The forward takeover: an unverified email whose address already belongs
    to a victim must be refused (409), never merged, never 500."""
    victim = _user("victim@example.com", "google", "g1")
    # q1 provider match → None; email-link SKIPPED (unverified); q2 clash → victim
    session = _session_returning(None, victim)
    with pytest.raises(HTTPException) as exc:
        await _upsert_oauth_user(
            _factory(session),
            oauth_provider="github",
            oauth_id="attacker1",
            email="victim@example.com",
            name="Attacker",
            email_verified=False,
        )
    assert exc.value.status_code == 409
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_verified_email_links_across_providers() -> None:
    """A verified email may still link a returning user across providers."""
    existing = _user("user@example.com", "google", "g1")
    session = _session_returning(None, existing)  # q1 None, q2 email match → existing
    user = await _upsert_oauth_user(
        _factory(session),
        oauth_provider="github",
        oauth_id="gh1",
        email="user@example.com",
        name="User via GitHub",
        email_verified=True,
    )
    assert user is existing
    session.add.assert_not_called()


@pytest.mark.asyncio
async def test_verified_new_email_creates_account() -> None:
    session = _session_returning(None, None)  # q1 None, q2 email → None
    user = await _upsert_oauth_user(
        _factory(session),
        oauth_provider="google",
        oauth_id="g2",
        email="new@example.com",
        name="New",
        email_verified=True,
    )
    session.add.assert_called_once()
    assert user.email == "new@example.com"


@pytest.mark.asyncio
async def test_unverified_email_with_no_clash_creates_account() -> None:
    """Unverified is fine when the address isn't already taken."""
    session = _session_returning(None, None)  # q1 None, clash → None
    user = await _upsert_oauth_user(
        _factory(session),
        oauth_provider="github",
        oauth_id="gh2",
        email="brand-new@example.com",
        name="New",
        email_verified=False,
    )
    session.add.assert_called_once()
    assert user.email == "brand-new@example.com"


@pytest.mark.asyncio
async def test_returning_user_matches_by_provider_id_regardless_of_verification() -> None:
    """The provider identity (provider + oauth_id) is always trusted for its own
    account — verification gating applies only to cross-provider email links."""
    existing = _user("me@example.com", "github", "gh1")
    session = _session_returning(existing)  # q1 provider match → existing
    user = await _upsert_oauth_user(
        _factory(session),
        oauth_provider="github",
        oauth_id="gh1",
        email="me@example.com",
        name="Me Updated",
        email_verified=False,
    )
    assert user is existing
    session.add.assert_not_called()
