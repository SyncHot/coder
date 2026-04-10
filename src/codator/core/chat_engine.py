"""Chat engine — orchestrator connecting inference, context, indexing, and git."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from codator.config import AppSettings, get_settings
from codator.core.context_manager import AdaptiveContextManager
from codator.core.context_retrieval import ContextualIndex
from codator.core.git_integration import GitContext
from codator.core.model_selector import ModelSelector
from codator.core.project_indexer import TreeSitterProjectIndexer
from codator.core.tool_registry import ToolRegistry
from codator.domain.interfaces import InferenceBackend, Tool
from codator.domain.models import (
    GenerationResult,
    Message,
    ProjectMap,
    Role,
    ToolCall,
    ToolResult,
)
from codator.infrastructure.api_clients import ClaudeBackend, OpenAIBackend
from codator.infrastructure.inference import DummyBackend, LlamaCppBackend
from codator.infrastructure.ollama_backend import OllamaBackend
from codator.infrastructure.tools.browser_tool import BrowserTool
from codator.infrastructure.tools.file_tool import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from codator.infrastructure.tools.ssh_tool import SSHTool
from codator.infrastructure.tools.terminal_tool import TerminalTool
from codator.infrastructure.tools.web_tools import WebFetchTool, WebSearchTool

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 5

# Type for the user confirmation callback used by write/edit tools
ConfirmCallback = Callable[[str, str], Awaitable[bool]]


class _ConfirmingTool(Tool):
    """Wrapper that asks for human confirmation before executing a tool."""

    def __init__(self, inner: Tool, engine: ChatEngine) -> None:
        self._inner = inner
        self._engine = engine

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def parameters_schema(self) -> dict:
        return self._inner.parameters_schema

    async def execute(self, **kwargs) -> ToolResult:
        cb = self._engine._confirm_callback
        if cb is not None:
            summary = f"{self.name}: {json.dumps(kwargs, default=str)[:200]}"
            approved = await cb(self.name, summary)
            if not approved:
                return ToolResult(
                    success=False,
                    error="Write operation rejected by user.",
                )
        return await self._inner.execute(**kwargs)

    async def close(self) -> None:
        await self._inner.close()

SYSTEM_PROMPT = """\
You are **codator**, a senior software engineering assistant running locally. \
You have direct access to the user's project structure and git state.

Rules:
1. **Read before writing**: ALWAYS use read_file and list_directory FIRST to \
understand the code before making any changes. NEVER write or edit files without \
reading them first.
2. **Use your tools**: When the user asks about code, USE the read_file and \
list_directory tools to actually look at the files. Do NOT say you can't access files.
3. Reference the Project Map (provided below) for accurate file/function names.
4. When the git diff is provided, prioritize reviewing those changes.
5. Write production-quality code. Explain tradeoffs when relevant.
6. If unsure, say so — then propose a plan to find the answer.
7. Use the terminal tool to run commands when needed (tests, installs, etc.).
8. When calling tools, output ONLY the JSON tool call, no extra text around it.

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
        self._tools = ToolRegistry()
        self._model_selector: ModelSelector | None = None
        self._contextual_index: ContextualIndex | None = None
        self._confirm_callback: ConfirmCallback | None = None
        self._message_count: int = 0
        self._system_refresh_interval: int = 10

    # ----- Lifecycle -----

    async def initialize(self, model_path: str | None = None) -> None:
        """Set up backends, index project, register tools, and prepare system prompt."""
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

        # Initialize model selector for Ollama
        if isinstance(self._backend, OllamaBackend):
            await self._init_model_selector()

        # Initialize contextual index
        try:
            self._contextual_index = ContextualIndex(self._project_root)
            chunk_count = await self._contextual_index.index_project()
            logger.info("Contextual index: %d chunks", chunk_count)
            # Build embeddings if Ollama backend
            if isinstance(self._backend, OllamaBackend):
                try:
                    url = self._settings.ollama.base_url
                    has_cache = self._contextual_index._load_cached_embeddings(
                        "nomic-embed-text",
                    )
                    if has_cache:
                        logger.info("Embeddings loaded from cache")
                    else:
                        logger.info(
                            "Building embeddings for %d chunks (first run)...",
                            chunk_count,
                        )
                        embedded = await self._contextual_index.build_embeddings(
                            model="nomic-embed-text",
                            ollama_url=url,
                        )
                        logger.info("Embeddings: %d chunks", embedded)
                except Exception as emb_exc:
                    logger.debug("Embedding build skipped: %s", emb_exc)
        except Exception as exc:
            logger.warning("Contextual indexing failed: %s", exc)

        # Register agentic tools
        self._register_tools()

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

    async def _init_model_selector(self) -> None:
        """Set up hardware-aware model selector using Ollama model list."""
        try:
            assert isinstance(self._backend, OllamaBackend)
            models = await self._backend.list_models()
            if models:
                self._model_selector = ModelSelector(models)
                logger.info(
                    "Model selector initialized with %d models", len(models)
                )
        except Exception as exc:
            logger.warning("Model selector init failed: %s", exc)

    def _try_api_backend(self):
        provider = self._settings.api.provider
        if provider == "claude" and self._settings.api.claude_api_key:
            self._backend = ClaudeBackend(self._settings)
            self._active_model = self._settings.api.claude_model
        elif provider == "openai" and self._settings.api.openai_api_key:
            self._backend = OpenAIBackend(self._settings)
            self._active_model = self._settings.api.openai_model
        elif provider == "ollama":
            self._backend = OllamaBackend(self._settings)
            self._active_model = self._settings.ollama.model

    def _register_tools(self):
        """Register all tools including file access."""
        cfg = self._settings

        # File tools — read always; write/edit with confirmation gate
        self._tools.register(ReadFileTool(project_root=self._project_root))
        self._tools.register(ListDirectoryTool(project_root=self._project_root))

        # Write/edit tools with human-in-the-loop confirmation
        self._write_tool = WriteFileTool(project_root=self._project_root)
        self._edit_tool = EditFileTool(project_root=self._project_root)
        self._tools.register(_ConfirmingTool(self._write_tool, self))
        self._tools.register(_ConfirmingTool(self._edit_tool, self))

        # Terminal tool
        terminal = TerminalTool(
            working_dir=self._project_root,
            timeout=cfg.terminal.timeout,
            require_confirm=cfg.terminal.require_confirm,
            dangerous_patterns=cfg.terminal.dangerous_patterns,
        )
        self._tools.register(terminal)

        # SSH tool
        ssh = SSHTool(
            host=cfg.ssh.host, port=cfg.ssh.port,
            username=cfg.ssh.username, password=cfg.ssh.password,
            key_path=cfg.ssh.key_path, timeout=cfg.ssh.timeout,
        )
        self._tools.register(ssh)

        # Browser tool
        browser = BrowserTool(
            headless=cfg.browser.headless,
            timeout=cfg.browser.timeout,
            viewport_width=cfg.browser.viewport_width,
            viewport_height=cfg.browser.viewport_height,
        )
        self._tools.register(browser)

        # Web tools — lightweight fetch & search (no API key needed)
        self._tools.register(WebFetchTool())
        self._tools.register(WebSearchTool())

    def _build_system_prompt(self) -> str:
        project_ctx = ""
        if self._project_map:
            project_ctx = f"--- Project Map ---\n{self._project_map.summary(max_files=40)}"

        git_ctx = ""
        if self._git.is_available:
            git_ctx = f"--- Git Context ---\n{self._git.get_work_context()}"

        tools_ctx = self._tools.tool_prompt_section()

        return SYSTEM_PROMPT.format(
            project_context=project_ctx,
            git_context=git_ctx,
        ) + ("\n\n" + tools_ctx if tools_ctx else "")

    def _supports_tool_calling(self) -> bool:
        """Check if the current backend supports native function calling."""
        return isinstance(self._backend, OllamaBackend)

    # ----- Chat -----

    async def chat(self, user_input: str) -> GenerationResult:
        """Send a user message and get a complete response."""
        assert self._context is not None, "Call initialize() first"

        user_msg = Message(role=Role.USER, content=user_input)
        self._context.add_message(user_msg)

        # Check compaction before generating
        await self._context.maybe_compact()

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
        """Send a user message and stream the response, with agentic tool calling."""
        assert self._context is not None, "Call initialize() first"

        # Refresh system prompt periodically (picks up new git diff)
        self._message_count += 1
        if self._message_count % self._system_refresh_interval == 0:
            self._refresh_system_prompt()

        # Auto-select model if model selector is available
        if self._model_selector and isinstance(self._backend, OllamaBackend):
            current_tokens = self._context.total_tokens()
            choice = self._model_selector.select_model(user_input, current_tokens)
            if choice.model_name != self._active_model:
                self._backend.switch_model(choice.model_name, choice.num_ctx)
                self._active_model = choice.model_name
                logger.info(
                    "Auto-switched to %s (num_ctx=%d)",
                    choice.model_name, choice.num_ctx,
                )

        # Inject relevant code context as ephemeral system message
        # (keeps conversation history clean — retrieval context is not persisted)
        if self._contextual_index:
            try:
                if self._contextual_index.has_embeddings:
                    chunks = await self._contextual_index.search_async(
                        user_input, top_k=3,
                    )
                else:
                    chunks = self._contextual_index.search(
                        user_input, top_k=3,
                    )
                if chunks:
                    context_block = (
                        self._contextual_index
                        .format_chunks_for_prompt(chunks)
                    )
                    ctx_msg = Message(
                        role=Role.SYSTEM,
                        content=f"Relevant code context:\n{context_block}",
                        metadata={"ephemeral": True},
                    )
                    self._context.add_message(ctx_msg)
            except Exception as exc:
                logger.debug("Contextual search failed: %s", exc)

        user_msg = Message(role=Role.USER, content=user_input)
        self._context.add_message(user_msg)

        await self._context.maybe_compact()

        # If backend supports tool calling, use the agentic loop
        if self._supports_tool_calling():
            async for token in self._agentic_chat_stream():
                yield token
            return

        # Otherwise, plain streaming (no tool calling)
        full_response: list[str] = []
        async for token in self._backend.generate_stream(
            self._context.get_messages(),
            max_tokens=self._settings.inference.max_tokens,
            temperature=self._settings.inference.temperature,
        ):
            full_response.append(token)
            yield token

        response_text = "".join(full_response)
        assistant_msg = Message(role=Role.ASSISTANT, content=response_text)
        self._context.add_message(assistant_msg)

    async def _agentic_chat_stream(self) -> AsyncIterator[str]:
        """Agentic loop: generate → detect tool calls → execute → re-generate."""
        tools_defs = self._tools.to_openai_tools()
        seen_calls: set[str] = set()  # Track (tool_name, params) to detect loops

        for _iteration in range(MAX_TOOL_ITERATIONS):
            # Non-streaming call with tool support
            result = await self._backend.generate(
                self._context.get_messages(),
                max_tokens=self._settings.inference.max_tokens,
                temperature=self._settings.inference.temperature,
                tools=tools_defs,
            )

            # Check for native tool calls first, then text-based fallback
            tool_calls = result.tool_calls
            if not tool_calls and result.text:
                tool_calls = self._parse_text_tool_calls(result.text)

            if not tool_calls:
                # Final text response — yield it
                if result.text:
                    yield result.text
                assistant_msg = Message(
                    role=Role.ASSISTANT,
                    content=result.text,
                    token_count=result.tokens_generated,
                )
                self._context.add_message(assistant_msg)
                return

            # Detect repeated tool calls (loop prevention)
            # Normalize paths in parameters to catch ./ vs .// vs ./// etc.
            def _normalize_params(params: dict) -> dict:
                normalized = {}
                for k, v in params.items():
                    if isinstance(v, str) and ("/" in v or v == "."):
                        # Normalize path-like values
                        v = os.path.normpath(v)
                    normalized[k] = v
                return normalized

            call_keys = frozenset(
                f"{tc.tool_name}:{json.dumps(_normalize_params(tc.parameters), sort_keys=True)}"
                for tc in tool_calls
            )
            new_calls = call_keys - seen_calls

            # Also detect same-tool repetition: if the model has called the
            # same tool name 3+ times (even with different params), and the
            # latest call returns the same result, it's likely looping.
            same_tool_count = sum(
                1 for sc in seen_calls
                if sc.split(":", 1)[0] in {tc.tool_name for tc in tool_calls}
            )

            if not new_calls or same_tool_count >= 3:
                # All calls are repeats — force a text response
                logger.warning("Loop detected: model repeating same tool calls, forcing answer")
                force_msg = Message(
                    role=Role.USER,
                    content=(
                        "[System]: You already called these tools with the same arguments. "
                        "STOP calling tools NOW. Provide your final answer in plain text "
                        "based on the information you have gathered. Do NOT output JSON. "
                        "Respond in the same language as the user's original question."
                    ),
                )
                self._context.add_message(force_msg)
                # One more generation without tools to force text
                final = await self._backend.generate(
                    self._context.get_messages(),
                    max_tokens=self._settings.inference.max_tokens,
                    temperature=0.7,
                )
                text = final.text or ""
                # Strip any remaining JSON tool calls from the response
                text = re.sub(
                    r'```json\s*\{[^}]*"name"\s*:.*?\}.*?```',
                    "", text, flags=re.DOTALL,
                ).strip()
                if not text:
                    text = (
                        "I analyzed the available information but could not find "
                        "the specific file or module. Could you clarify which "
                        "part of the project you mean?"
                    )
                yield text
                self._context.add_message(
                    Message(role=Role.ASSISTANT, content=text)
                )
                return
            seen_calls.update(call_keys)

            # Model wants to call tools — store assistant message with tool_calls metadata
            raw_tool_calls = []
            for tc in tool_calls:
                raw_tool_calls.append({
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.parameters),
                    },
                })
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=result.text or "",
                metadata={"tool_calls": raw_tool_calls},
            )
            self._context.add_message(assistant_msg)

            # Execute each tool and add results as user messages
            for tc in tool_calls:
                yield f"\n🔧 **{tc.tool_name}**"
                params_str = ", ".join(f"{k}={v!r}" for k, v in tc.parameters.items())
                yield f"({params_str})...\n"

                tool_result = await self._tools.execute(tc)

                if tool_result.success:
                    display = tool_result.output
                    if len(display) > 500:
                        display = display[:500] + "\n... [truncated]"
                    yield f"✅ {display}\n\n"
                else:
                    yield f"❌ {tool_result.error}\n\n"

                # Add tool result as TOOL message (proper role for function calling)
                result_text = tool_result.output or tool_result.error
                if len(result_text) > 8000:
                    result_text = result_text[:8000] + "\n... [truncated]"
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result_text,
                    metadata={"tool_call_id": tc.call_id},
                )
                self._context.add_message(tool_msg)

        # Safety: if we hit max iterations
        yield "\n⚠️ Reached maximum tool iterations.\n"

    @staticmethod
    def _parse_text_tool_calls(text: str) -> list[ToolCall]:
        """Parse tool calls from model text output (fallback for non-native).

        Uses balanced-brace extraction instead of regex to handle nested JSON.
        """
        calls: list[ToolCall] = []
        decoder = json.JSONDecoder()
        i = 0
        while i < len(text):
            # Find next '{'
            idx = text.find("{", i)
            if idx == -1:
                break
            try:
                obj, end = decoder.raw_decode(text, idx)
                if (
                    isinstance(obj, dict)
                    and "name" in obj
                    and "arguments" in obj
                ):
                    calls.append(ToolCall(
                        tool_name=obj["name"],
                        parameters=obj.get("arguments", {}),
                        call_id=f"text_{obj['name']}",
                    ))
                i = idx + end
            except (json.JSONDecodeError, ValueError):
                i = idx + 1

        return calls

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
        elif provider == "ollama":
            self._backend = OllamaBackend(self._settings)
            self._active_model = self._settings.ollama.model
        else:
            return f"Unknown provider: {provider}"
        return f"Switched to API: {self._active_model}"

    async def switch_ollama_model(self, model: str) -> str:
        """Switch to a specific Ollama model."""
        if isinstance(self._backend, OllamaBackend):
            self._backend.switch_model(model)
        else:
            await self._backend.close()
            self._backend = OllamaBackend(self._settings, model=model)
        self._active_model = model
        return f"Switched to Ollama model: {model}"

    # ----- State -----

    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def model_selector(self) -> ModelSelector | None:
        return self._model_selector

    @property
    def contextual_index(self) -> ContextualIndex | None:
        return self._contextual_index

    def set_confirm_callback(self, cb: ConfirmCallback | None) -> None:
        """Set the human-in-the-loop confirmation callback for write/edit tools."""
        self._confirm_callback = cb

    @property
    def context_status(self) -> dict[str, Any]:
        if self._context:
            return self._context.status_dict()
        return {}

    async def refresh_project_context(self) -> str:
        """Re-index project and update git context."""
        self._project_map = await self._indexer.index(self._project_root)
        msg = (
            f"Re-indexed: {self._project_map.total_files} files, "
            f"{self._project_map.total_symbols} symbols"
        )
        # Also refresh contextual index
        if self._contextual_index:
            try:
                chunk_count = await self._contextual_index.index_project()
                msg += f", {chunk_count} chunks"
            except Exception as exc:
                logger.warning("Contextual re-index failed: %s", exc)
        logger.info(msg)
        return msg

    async def shutdown(self) -> None:
        await self._backend.close()
        if self._summary_backend:
            await self._summary_backend.close()
        await self._tools.close_all()

    def clear_context(self) -> None:
        """Fast context reset — keeps backends, tools, and indexes hot."""
        if self._context:
            self._context.clear()
            sys_prompt = self._build_system_prompt()
            self._context.add_message(
                Message(role=Role.SYSTEM, content=sys_prompt)
            )
        self._message_count = 0

    def _refresh_system_prompt(self) -> None:
        """Update the system message in-place with fresh git diff."""
        if not self._context:
            return
        new_prompt = self._build_system_prompt()
        messages = self._context.get_messages()
        for msg in messages:
            if msg.role == Role.SYSTEM and "codator" in msg.content:
                msg.content = new_prompt
                break

    # ----- Tool execution -----

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    async def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        return await self._tools.execute(call)
