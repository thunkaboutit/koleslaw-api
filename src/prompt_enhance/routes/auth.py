import logging
import secrets
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import select

from prompt_enhance.config import Settings, get_settings
from prompt_enhance.db.models import User
from prompt_enhance.schemas.auth import UserInfo
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

GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USER_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

FACEBOOK_AUTHORIZE_URL = "https://www.facebook.com/v19.0/dialog/oauth"
FACEBOOK_TOKEN_URL = "https://graph.facebook.com/v19.0/oauth/access_token"
FACEBOOK_USER_URL = "https://graph.facebook.com/v19.0/me?fields=id,name,email"

INSTAGRAM_AUTHORIZE_URL = "https://www.instagram.com/oauth/authorize"
INSTAGRAM_TOKEN_URL = "https://api.instagram.com/oauth/access_token"
INSTAGRAM_USER_URL = "https://graph.instagram.com/me?fields=id,username"

OAUTH_STATE_COOKIE = "pe_oauth_state"


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
# OAuth user upsert — shared across all social providers
# ---------------------------------------------------------------------------


async def _upsert_oauth_user(
    db_session_factory,
    *,
    oauth_provider: str,
    oauth_id: str,
    email: str,
    name: str,
) -> User:
    """Find or create a user for an OAuth sign-in.

    Lookup order:
      1. Exact match on oauth_provider + oauth_id  (returning user, same provider)
      2. Match on email                             (same person, different provider)
      3. Create a new user                          (first time)
    """
    async with db_session_factory() as session:
        # 1. Exact provider match
        result = await session.execute(
            select(User).where(User.oauth_provider == oauth_provider, User.oauth_id == oauth_id)
        )
        user = result.scalar_one_or_none()

        # 2. Cross-provider match by email
        if user is None and email:
            result = await session.execute(select(User).where(User.email == email))
            user = result.scalar_one_or_none()

        if user is None:
            # 3. Brand-new user
            user = User(
                oauth_provider=oauth_provider,
                oauth_id=oauth_id,
                email=email,
                name=name,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
        else:
            user.name = name
            await session.commit()

        return user


# ---------------------------------------------------------------------------
# Configured providers
# ---------------------------------------------------------------------------


@router.get("/providers")
async def get_providers(
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> list[str]:
    """Return the list of OAuth provider IDs that have a client_id configured."""
    providers: list[str] = []
    if settings.github_client_id:
        providers.append("github")
    if settings.google_client_id:
        providers.append("google")
    if settings.facebook_client_id:
        providers.append("facebook")
    if settings.instagram_client_id:
        providers.append("instagram")
    return providers


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

    user = await _upsert_oauth_user(
        request.app.state.db_session_factory,
        oauth_provider="github",
        oauth_id=github_id,
        email=email,
        name=name,
    )

    # Set session cookie and redirect to frontend
    response = _session_response(user.id, settings, redirect_to=f"{settings.frontend_url}/dashboard")
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------


@router.get("/google")
async def google_login(
    response: Response,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    if not settings.google_client_id:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    state = secrets.token_urlsafe(32)

    redirect_url = (
        f"{GOOGLE_AUTHORIZE_URL}"
        f"?client_id={settings.google_client_id}"
        f"&redirect_uri={settings.frontend_url}/auth/google/callback"
        f"&response_type=code"
        f"&scope=openid%20email%20profile"
        f"&state={state}"
    )

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


@router.get("/google/callback")
async def google_callback(
    request: Request,
    code: str,
    state: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not expected_state or not secrets.compare_digest(state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    if not settings.google_client_secret:
        raise HTTPException(status_code=500, detail="Google OAuth not configured")

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            GOOGLE_TOKEN_URL,
            json={
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret.get_secret_value(),
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{settings.frontend_url}/auth/google/callback",
            },
            headers={"Accept": "application/json"},
        )
        token_data = token_resp.json()

    access_token = token_data.get("access_token")
    if not access_token:
        logger.error("Google token exchange failed: %s", token_data)
        raise HTTPException(status_code=400, detail="Failed to authenticate with Google")

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            GOOGLE_USER_URL,
            headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        )
        g_user = user_resp.json()

    google_id = str(g_user["id"])
    name = g_user.get("name", "")
    email = g_user.get("email", "")

    user = await _upsert_oauth_user(
        request.app.state.db_session_factory,
        oauth_provider="google",
        oauth_id=google_id,
        email=email,
        name=name,
    )

    response = _session_response(user.id, settings, redirect_to=f"{settings.frontend_url}/dashboard")
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Facebook OAuth
# ---------------------------------------------------------------------------


@router.get("/facebook")
async def facebook_login(
    response: Response,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    if not settings.facebook_client_id:
        raise HTTPException(status_code=500, detail="Facebook OAuth not configured")

    state = secrets.token_urlsafe(32)

    redirect_url = (
        f"{FACEBOOK_AUTHORIZE_URL}"
        f"?client_id={settings.facebook_client_id}"
        f"&redirect_uri={settings.frontend_url}/auth/facebook/callback"
        f"&scope=email,public_profile"
        f"&state={state}"
    )

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


@router.get("/facebook/callback")
async def facebook_callback(
    request: Request,
    code: str,
    state: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not expected_state or not secrets.compare_digest(state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    if not settings.facebook_client_secret:
        raise HTTPException(status_code=500, detail="Facebook OAuth not configured")

    async with httpx.AsyncClient() as client:
        token_resp = await client.get(
            FACEBOOK_TOKEN_URL,
            params={
                "client_id": settings.facebook_client_id,
                "client_secret": settings.facebook_client_secret.get_secret_value(),
                "code": code,
                "redirect_uri": f"{settings.frontend_url}/auth/facebook/callback",
            },
        )
        token_data = token_resp.json()

    access_token = token_data.get("access_token")
    if not access_token:
        logger.error("Facebook token exchange failed: %s", token_data)
        raise HTTPException(status_code=400, detail="Failed to authenticate with Facebook")

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            FACEBOOK_USER_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        fb_user = user_resp.json()

    facebook_id = str(fb_user["id"])
    name = fb_user.get("name", "")
    email = fb_user.get("email", "")

    user = await _upsert_oauth_user(
        request.app.state.db_session_factory,
        oauth_provider="facebook",
        oauth_id=facebook_id,
        email=email,
        name=name,
    )

    response = _session_response(user.id, settings, redirect_to=f"{settings.frontend_url}/dashboard")
    response.delete_cookie(OAUTH_STATE_COOKIE)
    return response


# ---------------------------------------------------------------------------
# Instagram OAuth
# ---------------------------------------------------------------------------


@router.get("/instagram")
async def instagram_login(
    response: Response,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    if not settings.instagram_client_id:
        raise HTTPException(status_code=500, detail="Instagram OAuth not configured")

    state = secrets.token_urlsafe(32)

    redirect_url = (
        f"{INSTAGRAM_AUTHORIZE_URL}"
        f"?client_id={settings.instagram_client_id}"
        f"&redirect_uri={settings.frontend_url}/auth/instagram/callback"
        f"&scope=instagram_basic"
        f"&response_type=code"
        f"&state={state}"
    )

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


@router.get("/instagram/callback")
async def instagram_callback(
    request: Request,
    code: str,
    state: str,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    expected_state = request.cookies.get(OAUTH_STATE_COOKIE)
    if not expected_state or not secrets.compare_digest(state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid OAuth state")

    if not settings.instagram_client_secret:
        raise HTTPException(status_code=500, detail="Instagram OAuth not configured")

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(
            INSTAGRAM_TOKEN_URL,
            data={
                "client_id": settings.instagram_client_id,
                "client_secret": settings.instagram_client_secret.get_secret_value(),
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": f"{settings.frontend_url}/auth/instagram/callback",
            },
        )
        token_data = token_resp.json()

    access_token = token_data.get("access_token")
    if not access_token:
        logger.error("Instagram token exchange failed: %s", token_data)
        raise HTTPException(status_code=400, detail="Failed to authenticate with Instagram")

    user_id_from_token = str(token_data.get("user_id", ""))

    async with httpx.AsyncClient() as client:
        user_resp = await client.get(
            INSTAGRAM_USER_URL,
            params={"access_token": access_token},
        )
        ig_user = user_resp.json()

    instagram_id = str(ig_user.get("id", user_id_from_token))
    username = ig_user.get("username", "")
    name = username
    email = f"{username}@instagram.oauth"

    user = await _upsert_oauth_user(
        request.app.state.db_session_factory,
        oauth_provider="instagram",
        oauth_id=instagram_id,
        email=email,
        name=name,
    )

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
