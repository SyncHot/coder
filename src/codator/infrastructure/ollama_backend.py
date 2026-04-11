"""Ollama backend — local LLM inference via Ollama's OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from codator.config import AppSettings, get_settings
from codator.domain.interfaces import InferenceBackend
from codator.domain.models import GenerationResult, Message, Role, ToolCall
from codator.infrastructure.tokenizer import count_tokens_tiktoken

logger = logging.getLogger(__name__)


class OllamaBackend(InferenceBackend):
    """Inference backend using Ollama's local API (OpenAI-compatible)."""

    def __init__(
        self, settings: AppSettings | None = None, model: str = "", num_ctx: int = 0,
    ):
        self._settings = settings or get_settings()
        self._model = model or self._settings.ollama.model
        self._base_url = self._settings.ollama.base_url.rstrip("/")
        self._num_ctx = num_ctx
        self._client = None
        self._supports_tools = True  # assume yes; auto-detected on first call

    def _ensure_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                base_url=f"{self._base_url}/v1",
                api_key="ollama",  # Ollama doesn't need a real key
            )

    @staticmethod
    def _prepare_messages(messages: list[Message]) -> list[dict[str, Any]]:
        """Convert messages for Ollama, handling SUMMARY, TOOL, and tool_calls."""
        result: list[dict[str, Any]] = []
        for m in messages:
            if m.role == Role.TOOL:
                # Tool result message
                result.append({
                    "role": "tool",
                    "tool_call_id": m.metadata.get("tool_call_id", ""),
                    "content": m.content,
                })
            elif m.role == Role.ASSISTANT and m.metadata.get("tool_calls"):
                # Assistant message that requested tool calls
                msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": m.metadata["tool_calls"],
                }
                result.append(msg)
            elif m.role == Role.SUMMARY:
                result.append({"role": "system", "content": m.content})
            else:
                result.append(m.to_llm_dict())
        return result

    async def generate(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3,
        stream=False, tools: list[dict] | None = None,
    ) -> GenerationResult:
        self._ensure_client()
        prepared = self._prepare_messages(messages)

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": prepared,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools and self._supports_tools:
            kwargs["tools"] = tools
        if self._num_ctx:
            kwargs["extra_body"] = {"num_ctx": self._num_ctx}

        t0 = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some models (e.g. deepseek-r1) don't support native tool calling.
            # Retry without tools — text-based tool parsing will handle it.
            if "does not support tools" in str(e) and "tools" in kwargs:
                logger.warning(
                    "Model %s does not support tools, falling back to text mode",
                    self._model,
                )
                self._supports_tools = False
                del kwargs["tools"]
                response = await self._client.chat.completions.create(**kwargs)
            # Ollama returns 500 when num_ctx is too large for available memory.
            # Retry with progressively smaller context until it works.
            elif "500" in str(e) and self._num_ctx and self._num_ctx > 4096:
                original_ctx = self._num_ctx
                while self._num_ctx > 4096:
                    self._num_ctx = max(4096, self._num_ctx // 2)
                    kwargs["extra_body"] = {"num_ctx": self._num_ctx}
                    logger.warning(
                        "Ollama OOM with num_ctx=%d, retrying with %d",
                        original_ctx, self._num_ctx,
                    )
                    try:
                        response = await self._client.chat.completions.create(**kwargs)
                        break
                    except Exception:
                        original_ctx = self._num_ctx
                        continue
                else:
                    raise
            else:
                raise
        elapsed = time.perf_counter() - t0

        choice = response.choices[0]

        # Parse tool calls if present
        tool_calls: list[ToolCall] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Malformed tool call arguments for '%s': %s",
                        tc.function.name, tc.function.arguments,
                    )
                    args = {}
                tool_calls.append(ToolCall(
                    tool_name=tc.function.name,
                    parameters=args,
                    call_id=tc.id or "",
                ))

        return GenerationResult(
            text=choice.message.content or "",
            tokens_generated=response.usage.completion_tokens if response.usage else 0,
            tokens_prompt=response.usage.prompt_tokens if response.usage else 0,
            time_seconds=elapsed,
            model_name=self._model,
            stopped_by=choice.finish_reason or "stop",
            tool_calls=tool_calls,
        )

    async def generate_stream(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3,
    ) -> AsyncIterator[str]:
        self._ensure_client()
        prepared = self._prepare_messages(messages)

        stream_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": prepared,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if self._num_ctx:
            stream_kwargs["extra_body"] = {"num_ctx": self._num_ctx}

        stream = await self._client.chat.completions.create(**stream_kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def count_tokens(self, text: str) -> int:
        # tiktoken cl100k_base underestimates tokens for non-OpenAI models
        # (Qwen, Llama, Mistral, etc.) by 15-25%. Apply 20% safety margin
        # to trigger context compaction at the right time.
        base = count_tokens_tiktoken(text)
        return int(base * 1.20)

    def switch_model(self, model: str, num_ctx: int = 0) -> None:
        """Switch to a different model without recreating the HTTP client."""
        self._model = model
        self._num_ctx = num_ctx
        self._supports_tools = True  # reset — new model may support tools

    @property
    def num_ctx(self) -> int:
        return self._num_ctx

    @property
    def supports_tools(self) -> bool:
        return self._supports_tools

    async def get_model_context_length(self, model: str | None = None) -> int:
        """Query Ollama /api/show for the model's native context length."""
        import httpx
        model = model or self._model
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model},
                    timeout=10,
                )
                resp.raise_for_status()
                info = resp.json().get("model_info", {})
                for key, val in info.items():
                    if "context_length" in key.lower():
                        return int(val)
        except Exception as exc:
            logger.warning("Failed to query model context for %s: %s", model, exc)
        return 0

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

    async def get_model_capabilities(self, model: str | None = None) -> dict[str, Any]:
        """Query Ollama /api/show and return detected model capabilities.

        Returns a dict with:
          - tools: bool — native function/tool calling
          - thinking: bool — chain-of-thought / reasoning mode
          - system: bool — supports system prompts
          - fim: bool — fill-in-middle code completion
          - family: str — model family (qwen2, gemma4, llama, etc.)
          - architecture: str — model architecture
          - quantization: str — quantization level
          - suggested_role: str — "architect", "editor", "general", or "both"
        """
        import httpx
        model = model or self._model
        caps: dict[str, Any] = {
            "tools": False,
            "thinking": False,
            "system": False,
            "fim": False,
            "family": "unknown",
            "architecture": "unknown",
            "quantization": "unknown",
            "suggested_role": "general",
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{self._base_url}/api/show",
                    json={"name": model},
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()

            tmpl = data.get("template", "")
            details = data.get("details", {})
            info = data.get("model_info", {})

            caps["tools"] = ".Tools" in tmpl or "tool_call" in tmpl
            caps["thinking"] = (
                "think" in tmpl.lower()
                or "/think" in tmpl.lower()
                or "r1" in model.lower()
                or "deepseek-r1" in model.lower()
            )
            caps["system"] = ".System" in tmpl or "system" in tmpl.lower()
            caps["fim"] = "fim_prefix" in tmpl or "fim_middle" in tmpl
            caps["family"] = details.get("family", "unknown")
            caps["architecture"] = info.get("general.architecture", "unknown")
            caps["quantization"] = details.get("quantization_level", "unknown")

            # Suggest role based on capabilities
            if caps["tools"] and not caps["thinking"]:
                caps["suggested_role"] = "editor"
            elif caps["thinking"] and not caps["tools"]:
                caps["suggested_role"] = "architect"
            elif caps["tools"] and caps["thinking"]:
                caps["suggested_role"] = "both"
            else:
                caps["suggested_role"] = "general"

        except Exception as exc:
            logger.warning("Failed to get capabilities for %s: %s", model, exc)

        return caps
