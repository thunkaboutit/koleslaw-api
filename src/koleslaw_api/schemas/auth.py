import uuid
from datetime import datetime

from pydantic import BaseModel


class UserInfo(BaseModel):
    id: uuid.UUID
    email: str
    name: str
    oauth_provider: str
    created_at: datetime
    plan: str = "free"
