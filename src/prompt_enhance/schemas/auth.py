import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class UserInfo(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    oauth_provider: str
    created_at: datetime


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=255)
    password: str = Field(min_length=1, max_length=255)
