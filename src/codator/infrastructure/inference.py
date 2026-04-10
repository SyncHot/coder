"""Local inference engine — llama-cpp-python with ROCm/HIP backend."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from codator.config import AppSettings, get_settings
from codator.domain.interfaces import InferenceBackend
from codator.domain.models import GenerationResult, Message, Role
from codator.infrastructure.tokenizer import count_tokens_llama, count_tokens_tiktoken

logger = logging.getLogger(__name__)


class LlamaCppBackend(InferenceBackend):
    """Inference via llama-cpp-python compiled with ROCm support."""

    def __init__(self, model_path: str | None = None, settings: AppSettings | None = None):
        self._settings = settings or get_settings()
        self._model_path = model_path or self._settings.inference.model_path
        self._model: Any = None
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self._model_path or not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self._model_path!r}. "
                "Set inference.model_path in config or pass --model to CLI."
            )
        # Load in executor to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        self._loaded = True

    def _load_model(self):
        from llama_cpp import Llama

        cfg = self._settings.inference
        logger.info("Loading model: %s (n_gpu_layers=%d)", self._model_path, cfg.n_gpu_layers)
        return Llama(
            model_path=self._model_path,
            n_ctx=cfg.context_size,
            n_gpu_layers=cfg.n_gpu_layers,
            n_threads=cfg.n_threads,
            verbose=False,
        )

    def _build_prompt(self, messages: list[Message]) -> list[dict[str, str]]:
        return [m.to_llm_dict() for m in messages]

    async def generate(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
        stream: bool = False,
    ) -> GenerationResult:
        await self._ensure_loaded()
        prompt = self._build_prompt(messages)
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()

        response = await loop.run_in_executor(
            None,
            lambda: self._model.create_chat_completion(
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=False,
            ),
        )
        elapsed = time.perf_counter() - t0

        choice = response["choices"][0]
        return GenerationResult(
            text=choice["message"]["content"],
            tokens_generated=response.get("usage", {}).get("completion_tokens", 0),
            tokens_prompt=response.get("usage", {}).get("prompt_tokens", 0),
            time_seconds=elapsed,
            model_name=Path(self._model_path).stem,
            stopped_by=choice.get("finish_reason", "unknown"),
        )

    async def generate_stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        await self._ensure_loaded()
        prompt = self._build_prompt(messages)
        loop = asyncio.get_event_loop()

        # Create streaming completion in executor, then iterate
        stream = await loop.run_in_executor(
            None,
            lambda: self._model.create_chat_completion(
                messages=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ),
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")
            if token:
                yield token

    def count_tokens(self, text: str) -> int:
        if self._model is not None:
            return count_tokens_llama(text, self._model)
        return count_tokens_tiktoken(text)

    async def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded = False


class DummyBackend(InferenceBackend):
    """Fallback backend when no model is loaded (for testing / API-only mode)."""

    async def generate(self, messages, *, max_tokens=2048, temperature=0.3, stream=False):
        return GenerationResult(
            text="[No local model loaded. Use /model to load one, or /api to switch to cloud.]",
            model_name="none",
            stopped_by="no_model",
        )

    async def generate_stream(self, messages, *, max_tokens=2048, temperature=0.3):
        yield "[No local model loaded]"

    def count_tokens(self, text: str) -> int:
        return count_tokens_tiktoken(text)

    async def close(self) -> None:
        pass
