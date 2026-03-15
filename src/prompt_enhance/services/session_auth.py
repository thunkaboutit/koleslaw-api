import logging
import uuid
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, Request

from prompt_enhance.config import Settings, get_settings

logger = logging.getLogger(__name__)

SESSION_COOKIE = "pe_session"
SESSION_MAX_AGE = timedelta(days=7)


def create_session_token(user_id: uuid.UUID, settings: Settings) -> str:
    payload = {
        "sub": str(user_id),
        "iat": datetime.now(UTC),
        "exp": datetime.now(UTC) + SESSION_MAX_AGE,
    }
    return jwt.encode(payload, settings.session_secret.get_secret_value(), algorithm="HS256")


def decode_session_token(token: str, settings: Settings) -> uuid.UUID:
    try:
        payload = jwt.decode(token, settings.session_secret.get_secret_value(), algorithms=["HS256"])
        return uuid.UUID(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Invalid session") from exc


async def get_current_user_id(
    request: Request,
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> uuid.UUID:
    token = request.cookies.get(SESSION_COOKIE)
    if token is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return decode_session_token(token, settings)
