from datetime import datetime

from pydantic import BaseModel


class UsageSummary(BaseModel):
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    cached_requests: int


class DailyUsage(BaseModel):
    date: str
    requests: int
    input_tokens: int
    output_tokens: int


class UsageResponse(BaseModel):
    summary: UsageSummary
    daily: list[DailyUsage]
    period_start: datetime
    period_end: datetime
