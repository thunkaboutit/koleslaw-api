import uuid
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import Date, Integer, cast, func, select

from koleslaw_api.db.models import ApiKey, UsageLog
from koleslaw_api.schemas.usage import DailyUsage, UsageResponse, UsageSummary
from koleslaw_api.services.session_auth import get_current_user_id

router = APIRouter(prefix="/v1/usage", tags=["usage"])


@router.get("")
async def get_usage(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
    user_id: uuid.UUID = Depends(get_current_user_id),  # noqa: B008
) -> UsageResponse:
    period_end = datetime.now(UTC)
    period_start = period_end - timedelta(days=days)

    db_session_factory = request.app.state.db_session_factory

    # Get all key IDs owned by this user
    async with db_session_factory() as session:
        key_result = await session.execute(select(ApiKey.id).where(ApiKey.user_id == user_id))
        key_ids = [row[0] for row in key_result.all()]

    if not key_ids:
        return UsageResponse(
            summary=UsageSummary(total_requests=0, total_input_tokens=0, total_output_tokens=0, cached_requests=0),
            daily=[],
            period_start=period_start,
            period_end=period_end,
        )

    async with db_session_factory() as session:
        # Summary query
        summary_result = await session.execute(
            select(
                func.count().label("total_requests"),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("total_input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("total_output_tokens"),
                func.coalesce(func.sum(func.cast(UsageLog.cached, Integer)), 0).label("cached_requests"),
            ).where(UsageLog.api_key_id.in_(key_ids), UsageLog.timestamp >= period_start)
        )
        summary_row = summary_result.one()

        # Daily breakdown
        day_col = cast(UsageLog.timestamp, Date).label("day")
        daily_result = await session.execute(
            select(
                day_col,
                func.count().label("requests"),
                func.coalesce(func.sum(UsageLog.input_tokens), 0).label("input_tokens"),
                func.coalesce(func.sum(UsageLog.output_tokens), 0).label("output_tokens"),
            )
            .where(UsageLog.api_key_id.in_(key_ids), UsageLog.timestamp >= period_start)
            .group_by(day_col)
            .order_by(day_col)
        )
        daily_rows = daily_result.all()

    summary = UsageSummary(
        total_requests=summary_row.total_requests,
        total_input_tokens=summary_row.total_input_tokens,
        total_output_tokens=summary_row.total_output_tokens,
        cached_requests=summary_row.cached_requests,
    )

    daily = [
        DailyUsage(
            date=str(row.day),
            requests=row.requests,
            input_tokens=row.input_tokens,
            output_tokens=row.output_tokens,
        )
        for row in daily_rows
    ]

    return UsageResponse(summary=summary, daily=daily, period_start=period_start, period_end=period_end)
