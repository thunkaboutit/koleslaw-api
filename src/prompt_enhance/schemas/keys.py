import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class CreateKeyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)


class CreateKeyResponse(BaseModel):
    id: uuid.UUID
    name: str
    key: str
    key_prefix: str
    created_at: datetime


class KeyInfo(BaseModel):
    id: uuid.UUID
    name: str
    key_prefix: str
    created_at: datetime
    last_used_at: datetime | None
    is_active: bool
    rate_limit: int


class KeyListResponse(BaseModel):
    keys: list[KeyInfo]
