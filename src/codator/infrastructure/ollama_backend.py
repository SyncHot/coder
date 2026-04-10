"""Ollama backend — local LLM inference via Ollama's OpenAI-compatible API."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from codator.config import AppSettings, get_settings
from codator.domain.interfaces import InferenceBackend
from codator.domain.models import GenerationResult, Message, Role
from codator.infrastructure.tokenizer import count_tokens_tiktoken

logger = logging.getLogger(__name__)


class OllamaBackend(InferenceBackend):
    """Inference backend using Ollama's local API (OpenAI-compatible)."""

    def __init__(self, settings: AppSettings | None = None, model: str = ""):
        self._settings = settings or get_settings()
        self._model = model or self._settings.ollama.model
        self._base_url = self._settings.ollama.base_url.rstrip("/")
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )

    @staticmethod
    def _prepare_messages(messages: list[Message]) -> list[dict[str, str]]:
        """Convert messages for Ollama (OpenAI-compatible), routing SUMMARY → system."""
        result: list[dict[str, str]] = []
        for m in messages:
            d = m.to_llm_dict()
            if m.role == Role.SUMMARY:
                d["role"] = "system"
            result.append(d)
        return result

    async def generate(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3, stream=False,
    ) -> GenerationResult:
        self._ensure_client()
        prepared = self._prepare_messages(messages)

        t0 = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=prepared,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - t0

        choice = response.choices[0]
        return GenerationResult(
            text=choice.message.content or "",
            tokens_generated=response.usage.completion_tokens if response.usage else 0,
            tokens_prompt=response.usage.prompt_tokens if response.usage else 0,
            time_seconds=elapsed,
            model_name=self._model,
            stopped_by=choice.finish_reason or "stop",
        )

    async def generate_stream(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3,
    ) -> AsyncIterator[str]:
        self._ensure_client()
        prepared = self._prepare_messages(messages)

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=prepared,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def count_tokens(self, text: str) -> int:
        return count_tokens_tiktoken(text)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    # ------------------------------------------------------------------
    # Ollama-specific helpers
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict]:
        """List models available in the local Ollama instance."""
        import httpx
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self._base_url}/api/tags", timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("models", [])
        except Exception as exc:
            logger.warning("Failed to list Ollama models: %s", exc)
            return []

    @property
    def model_name(self) -> str:
        return self._model
