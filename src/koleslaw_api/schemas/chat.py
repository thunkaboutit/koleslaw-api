from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str = Field(pattern=r"^(user|assistant)$")
    content: str = Field(min_length=1, max_length=50000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=100)
    model: str | None = None
    # Bounded like message content — appended into the model input (T2-E).
    context: str | None = Field(default=None, max_length=50000)


class ChatChunk(BaseModel):
    delta: str
    finish_reason: str | None = None
    type: Literal["text", "thinking"] = "text"


class ChatResponse(BaseModel):
    message: ChatMessage
    model_used: str
    usage: dict[str, int]
