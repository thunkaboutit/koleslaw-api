import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from koleslaw_api.config import Settings, get_settings
from koleslaw_api.db.models import ApiKey, UsageLog, User
from koleslaw_api.schemas.account import DeleteAccountRequest, UpdateProfileRequest
from koleslaw_api.schemas.auth import UserInfo
from koleslaw_api.services.session_auth import SESSION_COOKIE, get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/account", tags=["account"])


# ---------------------------------------------------------------------------
# Update profile
# ---------------------------------------------------------------------------


@router.put("/profile")
async def update_profile(
    request: Request,
    body: UpdateProfileRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> UserInfo:
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        user.name = body.name
        await session.commit()
        await session.refresh(user)

    return UserInfo(
        id=user.id,
        email=user.email,
        name=user.name,
        oauth_provider=user.oauth_provider,
        created_at=user.created_at,
        plan=user.plan,
    )


# ---------------------------------------------------------------------------
# Data export (GDPR Art. 20 — right to data portability)
# ---------------------------------------------------------------------------


class ExportApiKey(BaseModel):
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    is_active: bool


class ExportUsageEntry(BaseModel):
    timestamp: datetime
    input_tokens: int
    output_tokens: int


class ExportProfile(BaseModel):
    email: str
    name: str
    oauth_provider: str
    created_at: datetime


class DataExportResponse(BaseModel):
    exported_at: datetime
    profile: ExportProfile
    api_keys: list[ExportApiKey]
    usage_logs: list[ExportUsageEntry]


@router.get("/export")
async def export_data(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> DataExportResponse:
    """Export all user data in a portable format.

    Excludes sensitive internal fields: password hashes, internal IDs,
    key hashes, and OAuth provider IDs.
    """
    db_session_factory = request.app.state.db_session_factory

    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id).options(selectinload(User.api_keys)))
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        key_ids = [k.id for k in user.api_keys]
        api_keys = [
            ExportApiKey(
                name=k.name,
                key_prefix=k.key_prefix,
                created_at=k.created_at,
                last_used_at=k.last_used_at,
                is_active=k.is_active,
            )
            for k in user.api_keys
        ]

        usage_logs: list[ExportUsageEntry] = []
        if key_ids:
            usage_result = await session.execute(
                select(UsageLog).where(UsageLog.api_key_id.in_(key_ids)).order_by(UsageLog.timestamp.desc())
            )
            usage_logs = [
                ExportUsageEntry(
                    timestamp=log.timestamp,
                    input_tokens=log.input_tokens,
                    output_tokens=log.output_tokens,
                )
                for log in usage_result.scalars().all()
            ]

    from datetime import UTC

    return DataExportResponse(
        exported_at=datetime.now(UTC),
        profile=ExportProfile(
            email=user.email,
            name=user.name,
            oauth_provider=user.oauth_provider,
            created_at=user.created_at,
        ),
        api_keys=api_keys,
        usage_logs=usage_logs,
    )


# ---------------------------------------------------------------------------
# Delete account (GDPR Art. 17 — right to erasure)
# ---------------------------------------------------------------------------


@router.delete("")
async def delete_account(
    request: Request,
    body: DeleteAccountRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Response:
    """Permanently delete the user account and all associated data.

    The caller must supply their email address as the ``confirmation``
    field to prove intent.  This action is irreversible: the user row,
    all API keys, and all usage logs are hard-deleted.
    """
    db_session_factory = request.app.state.db_session_factory

    async with db_session_factory() as session:
        result = await session.execute(select(User).where(User.id == user_id))
        user = result.scalar_one_or_none()

        if user is None:
            raise HTTPException(status_code=404, detail="User not found")

        if body.confirmation != user.email:
            raise HTTPException(
                status_code=400,
                detail="Confirmation text does not match your email address.",
            )

        # Delete usage logs for all of this user's API keys
        key_result = await session.execute(select(ApiKey.id).where(ApiKey.user_id == user_id))
        key_ids = [row[0] for row in key_result.all()]

        if key_ids:
            usage_result = await session.execute(select(UsageLog).where(UsageLog.api_key_id.in_(key_ids)))
            for log in usage_result.scalars().all():
                await session.delete(log)

        # Delete API keys
        key_rows = await session.execute(select(ApiKey).where(ApiKey.user_id == user_id))
        for key in key_rows.scalars().all():
            await session.delete(key)

        # Delete the user
        await session.delete(user)
        await session.commit()

    logger.info("Account deleted: user_id=%s", user_id)

    response = Response(
        content='{"status":"deleted"}',
        media_type="application/json",
    )
    response.delete_cookie(
        SESSION_COOKIE,
        path="/",
        secure=settings.environment != "development",
        samesite="lax",
    )
    return response
