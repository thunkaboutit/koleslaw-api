from fastapi import APIRouter, Request

from koleslaw_api.schemas.enhance import ModelInfo

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> list[ModelInfo]:
    return await request.app.state.llm_service.get_available_models()
