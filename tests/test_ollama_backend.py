"""Tests for the Ollama inference backend."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codator.domain.models import Message, Role


class TestOllamaBackend:
    """OllamaBackend tests with mocked openai client."""

    @pytest.fixture
    def backend(self):
        from codator.infrastructure.ollama_backend import OllamaBackend

        settings = MagicMock()
        settings.ollama.base_url = "http://localhost:11434"
        settings.ollama.model = "qwen2.5-coder:7b"
        return OllamaBackend(settings=settings)

    def test_model_name(self, backend):
        assert backend.model_name == "qwen2.5-coder:7b"

    def test_prepare_messages_routes_summary(self, backend):
        msgs = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.SUMMARY, content="summary text"),
            Message(role=Role.USER, content="hello"),
        ]
        prepared = backend._prepare_messages(msgs)
        assert prepared[0]["role"] == "system"
        assert prepared[1]["role"] == "system"  # SUMMARY → system
        assert prepared[2]["role"] == "user"

    async def test_generate_calls_openai(self, backend):
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello world"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.usage.completion_tokens = 5
        mock_response.usage.prompt_tokens = 10
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        backend._client = mock_client

        msgs = [Message(role=Role.USER, content="hi")]
        result = await backend.generate(msgs)
        assert result.text == "Hello world"
        assert result.tokens_generated == 5
        assert result.model_name == "qwen2.5-coder:7b"

    async def test_list_models_success(self, backend):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5-coder:7b", "size": 4_700_000_000},
                {"name": "llama2:latest", "size": 3_800_000_000},
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_httpx:
            mock_http_client = AsyncMock()
            mock_http_client.get = AsyncMock(return_value=mock_resp)
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_http_client

            models = await backend.list_models()
            assert len(models) == 2
            assert models[0]["name"] == "qwen2.5-coder:7b"

    async def test_list_models_failure(self, backend):
        with patch("httpx.AsyncClient") as mock_httpx:
            mock_http_client = AsyncMock()
            mock_http_client.get = AsyncMock(side_effect=Exception("conn refused"))
            mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
            mock_http_client.__aexit__ = AsyncMock(return_value=None)
            mock_httpx.return_value = mock_http_client

            models = await backend.list_models()
            assert models == []

    def test_count_tokens(self, backend):
        count = backend.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0

    async def test_close(self, backend):
        mock_client = AsyncMock()
        backend._client = mock_client
        await backend.close()
        mock_client.close.assert_awaited_once()
        assert backend._client is None
