import os
from collections.abc import AsyncGenerator

import litellm
from litellm import acompletion

from prompt_enhance.config import Settings
from prompt_enhance.prompts.system import classify_enhancement_type, get_system_prompt
from prompt_enhance.schemas.enhance import (
    EnhanceChunk,
    EnhanceRequest,
    EnhanceResponse,
    ModelInfo,
    Usage,
)


class LLMServiceError(Exception):
    def __init__(self, message: str, code: str, detail: str | None = None):
        super().__init__(message)
        self.code = code
        self.detail = detail


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key.get_secret_value()
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()

    def _resolve_model(self, request: EnhanceRequest) -> str:
        return request.model or self._settings.default_model

    @staticmethod
    def resolve_enhancement_type(request: EnhanceRequest) -> str:
        if request.enhancement_type == "auto":
            return classify_enhancement_type(request.prompt)
        return request.enhancement_type

    def _build_messages(self, request: EnhanceRequest, resolved_type: str) -> list[dict[str, str]]:
        system_prompt = get_system_prompt(resolved_type, request.context)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt},
        ]

    def _map_exception(self, exc: Exception) -> LLMServiceError:
        if isinstance(exc, litellm.AuthenticationError):
            return LLMServiceError(str(exc), "authentication_error")
        if isinstance(exc, litellm.RateLimitError):
            return LLMServiceError(str(exc), "rate_limit_error")
        if isinstance(exc, litellm.BadRequestError):
            return LLMServiceError(str(exc), "bad_request")
        if isinstance(exc, litellm.NotFoundError):
            return LLMServiceError(str(exc), "model_not_found")
        if isinstance(exc, (litellm.Timeout, litellm.APIConnectionError)):
            return LLMServiceError(str(exc), "connection_error")
        if isinstance(exc, litellm.ServiceUnavailableError):
            return LLMServiceError(str(exc), "service_unavailable")
        return LLMServiceError(str(exc), "llm_error", detail=type(exc).__name__)

    async def enhance(self, request: EnhanceRequest, resolved_type: str | None = None) -> EnhanceResponse:
        model = self._resolve_model(request)
        resolved = resolved_type or self.resolve_enhancement_type(request)
        messages = self._build_messages(request, resolved)
        try:
            response = await acompletion(
                model=model,
                messages=messages,
                num_retries=3,
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc

        return EnhanceResponse(
            enhanced_prompt=response.choices[0].message.content,
            model_used=model,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    async def enhance_stream(
        self, request: EnhanceRequest, resolved_type: str | None = None
    ) -> AsyncGenerator[EnhanceChunk, None]:
        model = self._resolve_model(request)
        resolved = resolved_type or self.resolve_enhancement_type(request)
        messages = self._build_messages(request, resolved)
        try:
            response = await acompletion(
                model=model,
                messages=messages,
                stream=True,
                num_retries=3,
            )
        except Exception as exc:
            raise self._map_exception(exc) from exc

        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            finish_reason = chunk.choices[0].finish_reason
            if delta or finish_reason:
                yield EnhanceChunk(delta=delta, finish_reason=finish_reason)

    def get_available_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(
                id="anthropic/claude-haiku-4-5",
                name="Claude Haiku 4.5",
                provider="anthropic",
                tier="fast",
            ),
            ModelInfo(
                id="anthropic/claude-sonnet-4-5",
                name="Claude Sonnet 4",
                provider="anthropic",
                tier="standard",
            ),
            ModelInfo(
                id="openai/gpt-4o-mini",
                name="GPT-4o Mini",
                provider="openai",
                tier="fast",
            ),
            ModelInfo(
                id="openai/gpt-4o",
                name="GPT-4o",
                provider="openai",
                tier="premium",
            ),
        ]
