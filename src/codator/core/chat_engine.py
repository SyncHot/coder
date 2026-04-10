"""Chat engine — orchestrator connecting inference, context, indexing, and git."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from codator.config import AppSettings, get_settings
from codator.core.context_manager import AdaptiveContextManager
from codator.core.git_integration import GitContext
from codator.core.project_indexer import TreeSitterProjectIndexer
from codator.domain.interfaces import InferenceBackend
from codator.domain.models import GenerationResult, Message, ProjectMap, Role
from codator.infrastructure.inference import DummyBackend, LlamaCppBackend
from codator.infrastructure.api_clients import ClaudeBackend, OpenAIBackend

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are **codator**, a senior software engineering assistant running locally. \
You have direct access to the user's project structure and git state.

Rules:
1. **Zero-Hallucination Policy**: If you need a file's content and don't have it, \
ASK the user to provide it. Never guess or fabricate code.
2. Reference the Project Map (provided below) for accurate file/function names.
3. When the git diff is provided, prioritize reviewing those changes.
4. Write production-quality code. Explain tradeoffs when relevant.
5. If unsure, say so — then propose a plan to find the answer.

{project_context}
{git_context}
"""


class ChatEngine:
    """Main orchestrator for the coding assistant."""

    def __init__(self, settings: AppSettings | None = None, project_root: str = "."):
        self._settings = settings or get_settings()
        self._project_root = project_root

        # Components (initialized lazily or on startup)
        self._backend: InferenceBackend = DummyBackend()
        self._summary_backend: InferenceBackend | None = None
        self._context: AdaptiveContextManager | None = None
        self._indexer = TreeSitterProjectIndexer()
        self._git = GitContext(project_root)
        self._project_map: ProjectMap | None = None
        self._active_model: str = ""

    # ----- Lifecycle -----

    async def initialize(self, model_path: str | None = None) -> None:
        """Set up backends, index project, and prepare system prompt."""
        # Index project
        try:
            self._project_map = await self._indexer.index(self._project_root)
        except Exception as exc:
            logger.warning("Project indexing failed: %s", exc)

        # Set up inference backend
        if model_path or self._settings.inference.model_path:
            try:
                self._backend = LlamaCppBackend(model_path, self._settings)
                self._active_model = model_path or self._settings.inference.model_path
            except Exception as exc:
                logger.warning("Local model load failed: %s — using API fallback", exc)
                self._try_api_backend()
        else:
            self._try_api_backend()

        # Context manager
        self._context = AdaptiveContextManager(
            context_window=self._settings.inference.context_size,
            config=self._settings.context,
            primary_backend=self._backend,
            summary_backend=self._summary_backend,
        )

        # System prompt with project + git context
        sys_prompt = self._build_system_prompt()
        self._context.add_message(Message(role=Role.SYSTEM, content=sys_prompt))

    def _try_api_backend(self):
        provider = self._settings.api.provider
        if provider == "claude" and self._settings.api.claude_api_key:
            self._backend = ClaudeBackend(self._settings)
            self._active_model = self._settings.api.claude_model
        elif provider == "openai" and self._settings.api.openai_api_key:
            self._backend = OpenAIBackend(self._settings)
            self._active_model = self._settings.api.openai_model

    def _build_system_prompt(self) -> str:
        project_ctx = ""
        if self._project_map:
            project_ctx = f"--- Project Map ---\n{self._project_map.summary(max_files=40)}"

        git_ctx = ""
        if self._git.is_available:
            git_ctx = f"--- Git Context ---\n{self._git.get_work_context()}"

        return SYSTEM_PROMPT.format(project_context=project_ctx, git_context=git_ctx)

    # ----- Chat -----

    async def chat(self, user_input: str) -> GenerationResult:
        """Send a user message and get a complete response."""
        assert self._context is not None, "Call initialize() first"

        user_msg = Message(role=Role.USER, content=user_input)
        self._context.add_message(user_msg)

        # Check compaction before generating
        compacted = await self._context.maybe_compact()

        result = await self._backend.generate(
            self._context.get_messages(),
            max_tokens=self._settings.inference.max_tokens,
            temperature=self._settings.inference.temperature,
        )

        assistant_msg = Message(
            role=Role.ASSISTANT,
            content=result.text,
            token_count=result.tokens_generated,
        )
        self._context.add_message(assistant_msg)

        return result

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
        """Send a user message and stream the response token by token."""
        assert self._context is not None, "Call initialize() first"

        user_msg = Message(role=Role.USER, content=user_input)
        self._context.add_message(user_msg)

        await self._context.maybe_compact()

        full_response: list[str] = []
        async for token in self._backend.generate_stream(
            self._context.get_messages(),
            max_tokens=self._settings.inference.max_tokens,
            temperature=self._settings.inference.temperature,
        ):
            full_response.append(token)
            yield token

        # Save complete response
        response_text = "".join(full_response)
        assistant_msg = Message(role=Role.ASSISTANT, content=response_text)
        self._context.add_message(assistant_msg)

    # ----- Model management -----

    async def switch_model(self, model_path: str) -> str:
        """Hot-swap the inference model. Returns status message."""
        await self._backend.close()
        try:
            self._backend = LlamaCppBackend(model_path, self._settings)
            self._active_model = model_path
            return f"Switched to: {model_path}"
        except Exception as exc:
            self._backend = DummyBackend()
            return f"Failed to load {model_path}: {exc}"

    async def switch_to_api(self, provider: str) -> str:
        """Switch to a cloud API backend."""
        await self._backend.close()
        if provider == "claude":
            self._backend = ClaudeBackend(self._settings)
            self._active_model = self._settings.api.claude_model
        elif provider == "openai":
            self._backend = OpenAIBackend(self._settings)
            self._active_model = self._settings.api.openai_model
        else:
            return f"Unknown provider: {provider}"
        return f"Switched to API: {self._active_model}"

    # ----- State -----

    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def context_status(self) -> dict[str, Any]:
        if self._context:
            return self._context.status_dict()
        return {}

    async def refresh_project_context(self) -> str:
        """Re-index project and update git context."""
        self._project_map = await self._indexer.index(self._project_root)
        msg = f"Re-indexed: {self._project_map.total_files} files, {self._project_map.total_symbols} symbols"
        logger.info(msg)
        return msg

    async def shutdown(self) -> None:
        await self._backend.close()
        if self._summary_backend:
            await self._summary_backend.close()
