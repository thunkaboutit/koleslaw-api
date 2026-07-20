from typing import Literal

from pydantic import BaseModel, Field


class EnhanceRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=50000)
    # Bounded like prompt — it's appended into the model input, so leaving it
    # open is a cost/token-amplification lever past the prompt cap (T2-E).
    context: str | None = Field(default=None, max_length=50000)
    model: str | None = None
    enhancement_type: Literal["auto", "expand", "refine", "structure"] = "auto"
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
    tier: Literal["local", "fast", "standard", "premium"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, bool]
