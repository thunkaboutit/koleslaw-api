from typing import Literal

from pydantic import BaseModel, Field


class EnhanceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50000)
    context: str | None = None
    model: str | None = None
    enhancement_type: Literal["expand", "refine", "structure"] = "refine"
    stream: bool = True


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class EnhanceResponse(BaseModel):
    enhanced_prompt: str
    model_used: str
    usage: Usage
    cached: bool = False


class EnhanceChunk(BaseModel):
    delta: str
    finish_reason: Literal["stop", "length"] | None = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: str | None = None


class ModelInfo(BaseModel):
    id: str
    name: str
    provider: str
    tier: Literal["fast", "standard", "premium"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, bool]
