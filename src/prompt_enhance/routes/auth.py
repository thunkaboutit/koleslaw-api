import hashlib
import logging
import os
import secrets
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select

from prompt_enhance.config import Settings, get_settings
from prompt_enhance.db.models import User
from prompt_enhance.schemas.auth import LoginRequest, UserInfo
from prompt_enhance.services.session_auth import (
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    create_session_token,
    get_current_user_id,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_USER_URL = "https://api.github.com/user"
GITHUB_EMAILS_URL = "https://api.github.com/user/emails"

OAUTH_STATE_COOKIE = "pe_oauth_state"


# ---------------------------------------------------------------------------
# Password hashing (stdlib only — no extra dependencies)
# ---------------------------------------------------------------------------


def _hash_password(password: str) -> str:
    """Hash a password with PBKDF2-SHA256 and a random salt."""
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=600_000)
    return f"{salt.hex()}:{dk.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored salt:hash string."""
    try:
        salt_hex, hash_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations=600_000)
        return secrets.compare_digest(dk.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Dev admin seed
# ---------------------------------------------------------------------------


async def ensure_dev_admin(request: Request, settings: Settings) -> None:
    """Create a default admin/admin user in development mode if it doesn't exist."""
    if settings.environment != "development":
        return

    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.email == "admin"))
        if result.scalar_one_or_none() is not None:
            return

        admin = User(
            email="admin",
            name="Admin",
            oauth_provider="local",
            oauth_id="",
            password_hash=_hash_password("admin"),
        )
        session.add(admin)
        await session.commit()
        logger.info("Created default dev admin user (admin/admin)")


# ---------------------------------------------------------------------------
# Helper to build a session-cookie response
# ---------------------------------------------------------------------------


def _session_response(user_id: uuid.UUID, settings: Settings, *, redirect_to: str | None = None) -> Response:
    token = create_session_token(user_id, settings)
    if redirect_to:
        response = Response(status_code=307, headers={"Location": redirect_to})
    else:
        response = Response(content='{"status":"ok"}', media_type="application/json")
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=int(SESSION_MAX_AGE.total_seconds()),
        httponly=True,
        samesite="lax",
        secure=settings.environment != "development",
        path="/",
    )
    return response


# ---------------------------------------------------------------------------
# Username / password login
# ---------------------------------------------------------------------------


@router.post("/login")
async def login(
    request: Request,
    body: LoginRequest,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    # Seed the dev admin on first login attempt (idempotent)
    await ensure_dev_admin(request, settings)

    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.email == body.username, User.oauth_provider == "local"))
        user = result.scalar_one_or_none()

    if user is None or user.password_hash is None or not _verify_password(body.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return _session_response(user.id, settings)


# ---------------------------------------------------------------------------
# GitHub OAuth
# ---------------------------------------------------------------------------


@router.get("/github")
async def github_login(
    response: Response,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    if not settings.github_client_id:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")

    state = secrets.token_urlsafe(32)

    redirect_url = f"{GITHUB_AUTHORIZE_URL}?client_id={settings.github_client_id}&scope=user:email&state={state}"

    response = Response(status_code=307, headers={"Location": redirect_url})
    response.set_cookie(
        OAUTH_STATE_COOKIE,
        state,
        max_age=600,
        httponly=True,
        samesite="lax",
        secure=settings.environment != "development",
    )
    return response


@router.get("/github/callback")
async def github_callback(
    request: Request,
    code: str,
    state: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    # Verify CSRF state
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not expected_state or not secrets.compare_digest(state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    if not settings.github_client_secret:
        raise HTTPException(status_code=500, detail="GitHub OAuth not configured")

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GITHUB_TOKEN_URL,
            json={
                "client_id": settings.github_client_id,
                "client_secret": settings.github_client_secret.get_secret_value(),
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()

    access_token = token_data.get("access_token")
    if not access_token:
        logger.error("GitHub token exchange failed: %s", token_data)
        raise HTTPException(status_code=400, detail="Failed to authenticate with GitHub")

    # Fetch user profile
    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    async with httpx.AsyncClient() as client:
        user_resp = await client.get(GITHUB_USER_URL, headers=headers)
        gh_user = user_resp.json()

        # Fetch primary email if not public
        email = gh_user.get("email")
        if not email:
            emails_resp = await client.get(GITHUB_EMAILS_URL, headers=headers)
            emails = emails_resp.json()
            primary = next((e for e in emails if e.get("primary")), None)
            email = primary["email"] if primary else emails[0]["email"]

    github_id = str(gh_user["id"])
    name = gh_user.get("name") or gh_user.get("login", "")

    # Upsert user
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.oauth_provider == "github", User.oauth_id == github_id))
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                oauth_provider="github",
                oauth_id=github_id,
                email=email,
                name=name,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        else:
            user.email = email
            user.name = name
            await session.commit()

    # Set session cookie and redirect to frontend
    response = _session_response(user.id, settings, redirect_to=f"{settings.frontend_url}/dashboard")
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Session info / logout
# ---------------------------------------------------------------------------


@router.get("/me")
async def get_me(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> UserInfo:
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return UserInfo(
        id=user.id,
        email=user.email,
        name=user.name,
        oauth_provider=user.oauth_provider,
        created_at=user.created_at,
    )


@router.post("/logout")
async def logout(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> dict[str, str]:
    response = Response(content='{"status":"logged_out"}', media_type="application/json")
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=settings.environment != "development",
        samesite="lax",
    )
    return response  # type: ignore[return-value]
