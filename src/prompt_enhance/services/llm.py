import logging
import os
from collections.abc import AsyncGenerator

import httpx
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


# Failures that mean the backend serving the model is unreachable or not
# serving it right now — worth retrying on another tier, unlike auth or
# bad-request errors, which would fail identically anywhere. NotFoundError
# counts: a GPU instance that lost its model registration answers 404, which
# says nothing about whether a cloud model can take the request.
_AVAILABILITY_ERRORS = (
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.ServiceUnavailableError,
    litellm.InternalServerError,
    litellm.NotFoundError,
)


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        if settings.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = settings.anthropic_api_key.get_secret_value()
        if settings.openai_api_key:
            os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value()
        os.environ["OLLAMA_API_BASE"] = settings.ollama_base_url

    def _resolve_model(self, request: EnhanceRequest) -> str:
        return request.model or self._settings.default_model

    @staticmethod
    def resolve_enhancement_type(request: EnhanceRequest) -> str:
        if request.enhancement_type == "auto":
            return classify_enhancement_type(request.prompt)
        return request.enhancement_type

    def _build_messages(self, request: EnhanceRequest, resolved_type: str, model: str) -> list[dict[str, str]]:
        # Ollama-served models define their SYSTEM prompt in the Modelfile
        # (kept next to the weights in the model repo / S3). Sending no
        # system message lets it apply — a request-level one would override
        # it. Cloud fallback models have no Modelfile, so they still get the
        # per-type prompt; without it they'd answer the prompt instead of
        # enhancing it.
        if model.startswith("ollama_chat/"):
            content = request.prompt
            if request.context:
                content += (
                    "\n\n--- Additional Context ---\n"
                    "Context about the use case, to inform enhancement "
                    f"decisions:\n\n{request.context}"
                )
            return [{"role": "user", "content": content}]
        system_prompt = get_system_prompt(resolved_type, request.context)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt},
        ]

    @staticmethod
    def _log_llm_error(exc: Exception, model: str) -> None:
        logging.getLogger(__name__).error(
            "LLM call failed: model=%s, type=%s, error=%s", model, type(exc).__name__, exc
        )

    def _fall_back_or_raise(self, model: str, exc: Exception) -> str:
        """Return the fallback model for an availability failure on the local
        tier, or raise the mapped error.

        A GPU instance replacement (deploy, future spot reclaim) is a ~4 min
        serving gap; requests degrade to the configured cloud model instead
        of surfacing it. Only ollama_chat/* primaries qualify — a cloud
        model failing is not an availability problem another cloud model
        is likely to dodge, and its errors should stay visible.
        """
        self._log_llm_error(exc, model)
        fallback = self._settings.fallback_model
        if (
            not fallback
            or fallback == model
            or not model.startswith("ollama_chat/")
            or not isinstance(exc, _AVAILABILITY_ERRORS)
        ):
            raise self._map_exception(exc) from exc
        logging.getLogger(__name__).warning(
            "Model %s unavailable (%s); falling back to %s", model, type(exc).__name__, fallback
        )
        return fallback

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
        try:
            response = await acompletion(
                model=model,
                messages=self._build_messages(request, resolved, model),
                num_retries=3,
            )
        except Exception as exc:
            model = self._fall_back_or_raise(model, exc)
            try:
                response = await acompletion(
                    model=model,
                    messages=self._build_messages(request, resolved, model),
                    num_retries=3,
                )
            except Exception as fallback_exc:
                self._log_llm_error(fallback_exc, model)
                raise self._map_exception(fallback_exc) from fallback_exc

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
        try:
            response = await acompletion(
                model=model,
                messages=self._build_messages(request, resolved, model),
                stream=True,
                num_retries=3,
            )
        except Exception as exc:
            # Fallback only engages before the first byte reaches the
            # client; a mid-stream failure can't be retried without
            # duplicating output, so it propagates as before.
            model = self._fall_back_or_raise(model, exc)
            try:
                response = await acompletion(
                    model=model,
                    messages=self._build_messages(request, resolved, model),
                    stream=True,
                    num_retries=3,
                )
            except Exception as fallback_exc:
                self._log_llm_error(fallback_exc, model)
                raise self._map_exception(fallback_exc) from fallback_exc

        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            finish_reason = chunk.choices[0].finish_reason
            if delta or finish_reason:
                yield EnhanceChunk(delta=delta, finish_reason=finish_reason)

    async def get_available_models(self) -> list[ModelInfo]:
        models: list[ModelInfo] = []

        # Query Ollama for locally available models
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._settings.ollama_base_url}/api/tags")
                resp.raise_for_status()
                for m in resp.json().get("models", []):
                    name = m.get("name", "")
                    if not name:
                        continue
                    models.append(
                        ModelInfo(
                            id=f"ollama_chat/{name}",
                            name=name,
                            provider="ollama",
                            tier="local",
                        )
                    )
        except Exception:
            logging.getLogger(__name__).warning("Failed to fetch models from Ollama")

        # Add cloud provider models if API keys are configured
        if self._settings.anthropic_api_key:
            models.extend(
                [
                    ModelInfo(
                        id="anthropic/claude-haiku-4-5", name="Claude Haiku 4.5", provider="anthropic", tier="fast"
                    ),
                    ModelInfo(
                        id="anthropic/claude-sonnet-4-5", name="Claude Sonnet 4", provider="anthropic", tier="standard"
                    ),
                ]
            )
        if self._settings.openai_api_key:
            models.extend(
                [
                    ModelInfo(id="openai/gpt-4o-mini", name="GPT-4o Mini", provider="openai", tier="fast"),
                    ModelInfo(id="openai/gpt-4o", name="GPT-4o", provider="openai", tier="premium"),
                ]
            )

        return models
