from fastapi import APIRouter, Request

from prompt_enhance.schemas.enhance import ModelInfo

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models")
async def list_models(request: Request) -> list[ModelInfo]:
    return request.app.state.llm_service.get_available_models()
