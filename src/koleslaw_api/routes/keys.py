import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import select

from koleslaw_api.db.models import ApiKey
from koleslaw_api.schemas.keys import CreateKeyRequest, CreateKeyResponse, KeyInfo, KeyListResponse
from koleslaw_api.services.auth import generate_api_key, hash_api_key
from koleslaw_api.services.session_auth import get_current_user_id

router = APIRouter(prefix="/v1/keys", tags=["keys"])


@router.get("")
async def list_keys(
    request: Request,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> KeyListResponse:
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.user_id == user_id).order_by(ApiKey.created_at.desc())
        )
        rows = result.scalars().all()

    keys = [
        KeyInfo(
            id=row.id,
            name=row.name,
            key_prefix=row.key_prefix,
            created_at=row.created_at,
            last_used_at=row.last_used_at,
            is_active=row.is_active,
            rate_limit=row.rate_limit,
        )
        for row in rows
    ]
    return KeyListResponse(keys=keys)


@router.post("")
async def create_key(
    request: Request,
    body: CreateKeyRequest,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> CreateKeyResponse:
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    key_prefix = raw_key[:12]

    api_key = ApiKey(
        user_id=user_id,
        key_hash=key_hash,
        name=body.name,
        key_prefix=key_prefix,
    )

    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        session.add(api_key)
        await session.commit()
        await session.refresh(api_key)

    return CreateKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,
        key_prefix=key_prefix,
        created_at=api_key.created_at,
    )


@router.delete("/{key_id}")
async def revoke_key(
    request: Request,
    key_id: uuid.UUID,
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> dict[str, str]:
    db_session_factory = request.app.state.db_session_factory
    async with db_session_factory() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id, ApiKey.user_id == user_id))
        api_key = result.scalar_one_or_none()

        if api_key is None:
            raise HTTPException(status_code=404, detail="API key not found")

        api_key.is_active = False
        await session.commit()

    return {"status": "revoked"}
