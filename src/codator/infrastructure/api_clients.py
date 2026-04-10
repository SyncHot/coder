"""Cloud API clients — Claude (Anthropic) and OpenAI, behind the InferenceBackend interface."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from codator.config import AppSettings, get_settings
from codator.domain.interfaces import InferenceBackend
from codator.domain.models import GenerationResult, Message, Role
from codator.infrastructure.tokenizer import count_tokens_tiktoken

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Claude (Anthropic)
# ---------------------------------------------------------------------------

class ClaudeBackend(InferenceBackend):
    def __init__(self, settings: AppSettings | None = None):
        self._settings = settings or get_settings()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(
                api_key=self._settings.api.claude_api_key
            )

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, list[dict[str, str]]]:
        """Separate system/summary messages from chat messages for Claude API."""
        system_parts: list[str] = []
        chat_msgs: list[dict[str, str]] = []
        for m in messages:
            if m.role in (Role.SYSTEM, Role.SUMMARY):
                system_parts.append(m.content)
            else:
                chat_msgs.append(m.to_llm_dict())
        system_text = "\n\n".join(system_parts) if system_parts else ""
        return system_text, chat_msgs

    async def generate(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3, stream=False,
    ) -> GenerationResult:
        self._ensure_client()
        system_text, chat_msgs = self._split_messages(messages)

        t0 = time.perf_counter()
        response = await self._client.messages.create(
            model=self._settings.api.claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_text or "You are a senior software engineer.",
            messages=chat_msgs,
        )
        elapsed = time.perf_counter() - t0

        text = response.content[0].text if response.content else ""
        return GenerationResult(
            text=text,
            tokens_generated=response.usage.output_tokens,
            tokens_prompt=response.usage.input_tokens,
            time_seconds=elapsed,
            model_name=self._settings.api.claude_model,
            stopped_by=response.stop_reason or "end_turn",
        )

    async def generate_stream(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3,
    ) -> AsyncIterator[str]:
        self._ensure_client()
        system_text, chat_msgs = self._split_messages(messages)

        async with self._client.messages.stream(
            model=self._settings.api.claude_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_text or "You are a senior software engineer.",
            messages=chat_msgs,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def count_tokens(self, text: str) -> int:
        return count_tokens_tiktoken(text)

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

class OpenAIBackend(InferenceBackend):
    def __init__(self, settings: AppSettings | None = None):
        self._settings = settings or get_settings()
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(
                api_key=self._settings.api.openai_api_key
            )

    @staticmethod
    def _prepare_messages(messages: list[Message]) -> list[dict[str, str]]:
        """Convert messages for OpenAI API, merging SUMMARY into system role."""
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
            model=self._settings.api.openai_model,
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
            model_name=self._settings.api.openai_model,
            stopped_by=choice.finish_reason or "stop",
        )

    async def generate_stream(
        self, messages: list[Message], *, max_tokens=2048, temperature=0.3,
    ) -> AsyncIterator[str]:
        self._ensure_client()
        prepared = self._prepare_messages(messages)
        stream = await self._client.chat.completions.create(
            model=self._settings.api.openai_model,
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
