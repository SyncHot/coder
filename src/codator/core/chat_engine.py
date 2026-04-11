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
    GlobTool,
    GrepTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from codator.infrastructure.tools.ssh_tool import SSHTool
from codator.infrastructure.tools.terminal_tool import TerminalTool
from codator.infrastructure.tools.web_tools import WebFetchTool, WebSearchTool
from codator.infrastructure.tools.vision_tool import VisionTool
from codator.infrastructure.tools.issue_tool import IssueTool

logger = logging.getLogger(__name__)

MAX_TOOL_ITERATIONS = 15

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
        if cb is None:
            logger.warning("Write blocked — no confirmation handler: %s", self.name)
            return ToolResult(
                success=False,
                error="Write operation blocked: no confirmation handler configured. "
                "Call engine.set_confirm_callback() first.",
            )
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
**codator** — senior fullstack developer and programming partner with direct \
access to project files, terminal, and git state.

## Personality
- Thoughtful, experienced engineer — not a chatbot.
- Communicate naturally and professionally. Before making changes, briefly \
explain *why* this approach is best.
- **STRICT**: NEVER output raw JSON, step plans, action lists, "reasoning" blocks, \
or internal thought processes. Responses must always be natural language with \
clean Markdown formatting.
- Use syntax-highlighted code blocks only for actual code.
- Be concise. If a one-line change is requested, don't rewrite the whole file.
- Be proactive: on noticing a security risk, architectural smell, or flawed \
assumption, warn the user elegantly instead of blindly executing.
- Focus on real issues: logic bugs, memory leaks, performance bottlenecks, \
security vulnerabilities. Skip trivial suggestions like "add comments".

## Approach
1. **PLAN first**: Before acting, briefly state what will be done and why.
2. **READ before WRITE**: ALWAYS use read_file, grep, or list_directory FIRST \
to understand the code. NEVER edit files that haven't been read.
3. **VERIFY after EDIT**: After editing a file, read it back to confirm changes \
applied correctly. Run tests if available.
4. **One step at a time**: Don't try to do everything in one tool call. \
Explore → understand → plan → act → verify.

## Project Navigation
- Use the Project Map below as the source of truth for file paths.
- NEVER guess file paths. If unsure, use list_directory or glob to explore.
- Use grep to find definitions, usages, and patterns across files.

## Tool Constraints
- **read_file**: Max 256KB, text files only.
- **edit_file**: Requires EXACT text match (character-for-character). If rejected, \
re-read the file to get current content.
- **terminal**: 60s timeout, 1MB output limit. Dangerous commands need approval.
- **write_file**: Overwrites the entire file. Use edit_file for partial changes.

## Error Handling
- If a tool call fails, analyze the error and try a different approach.
- If edit_file can't find old_text, re-read the file — content may have changed.
- On permission errors or missing files, communicate the issue clearly — \
never dump raw tracebacks. Say what went wrong and suggest a fix.

## Communication Style
- Respond in the **same language** as the user's message.
- Write production-quality code. Explain tradeoffs when relevant.
- When calling tools, output ONLY the JSON tool call, no extra text around it.
- When the git diff is provided, prioritize reviewing those changes.

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
        self._manual_model_override: bool = False
        self._web_server: Any | None = None
        self._web_task: Any | None = None
        self._mode: str = "chat"  # "chat" or "agent"

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

        # Resolve actual num_ctx for the initial model (query Ollama metadata)
        initial_ctx = self._settings.inference.context_size
        if isinstance(self._backend, OllamaBackend):
            resolved = await self._resolve_num_ctx(self._active_model)
            if resolved:
                initial_ctx = resolved
                self._backend._num_ctx = resolved
                logger.info(
                    "Initial model %s: num_ctx=%d", self._active_model, resolved,
                )

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
            context_window=initial_ctx,
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
            self._backend = OllamaBackend(
                self._settings, num_ctx=self._settings.inference.context_size,
            )
            self._active_model = self._settings.ollama.model

    def _register_tools(self):
        """Register all tools including file access."""
        cfg = self._settings

        # File tools — read always; write/edit with confirmation gate
        self._tools.register(ReadFileTool(project_root=self._project_root))
        self._tools.register(ListDirectoryTool(project_root=self._project_root))
        self._tools.register(GrepTool(project_root=self._project_root))
        self._tools.register(GlobTool(project_root=self._project_root))

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

        # Vision tool — screenshot analysis via multimodal LLM
        if cfg.vision.enabled:
            vision = VisionTool(
                ollama_url=cfg.ollama.base_url,
                vision_model=cfg.vision.model,
                claude_api_key=getattr(cfg.api, 'claude_api_key', ''),
            )
            self._tools.register(vision)

        # Issue tool — create/manage tickets
        issue = IssueTool(
            backend=cfg.issue.backend,
            gitea_url=cfg.issue.gitea_url,
            gitea_token=cfg.issue.gitea_token,
            github_token=cfg.issue.github_token,
            default_repo=cfg.issue.default_repo,
            default_labels=cfg.issue.default_labels,
        )
        self._tools.register(issue)

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

        # Auto-select model if model selector is available (skip if user manually chose)
        if (
            self._model_selector
            and isinstance(self._backend, OllamaBackend)
            and not self._manual_model_override
        ):
            current_tokens = self._context.total_tokens()
            choice = self._model_selector.select_model(user_input, current_tokens)
            if choice.model_name != self._active_model:
                # Use hw-aware context resolution instead of static _MAX_CTX
                num_ctx = await self._resolve_num_ctx(choice.model_name)
                self._backend.switch_model(choice.model_name, num_ctx=num_ctx)
                self._active_model = choice.model_name
                if self._context:
                    self._context._context_window = num_ctx
                    # Compact context if it exceeds new (smaller) window
                    await self._context.maybe_compact()
                logger.info(
                    "Auto-switched to %s (num_ctx=%d)",
                    choice.model_name, num_ctx,
                )

        # Inject relevant code context as ephemeral system message
        # (keeps conversation history clean — retrieval context is not persisted)
        if self._contextual_index:
            try:
                # Dynamic RAG budget: use at most 25% of remaining context
                remaining = self._context.remaining_tokens()
                avg_chunk_tokens = 500  # approximate tokens per code chunk
                rag_budget = int(remaining * 0.25)
                top_k = max(1, min(10, rag_budget // avg_chunk_tokens))

                if self._contextual_index.has_embeddings:
                    chunks = await self._contextual_index.search_async(
                        user_input, top_k=top_k,
                    )
                else:
                    chunks = self._contextual_index.search(
                        user_input, top_k=top_k,
                    )
                if chunks:
                    context_block = (
                        self._contextual_index
                        .format_chunks_for_prompt(chunks)
                    )
                    # Truncate RAG context to token budget
                    block_tokens = self._context._count(context_block)
                    if block_tokens > rag_budget:
                        # Rough character-to-token ratio for truncation
                        char_limit = int(len(context_block) * (rag_budget / block_tokens))
                        context_block = context_block[:char_limit] + "\n... [RAG context truncated]"
                        logger.info("RAG context truncated: %d → %d tokens", block_tokens, rag_budget)
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

        compaction_notice = await self._context.maybe_compact()
        if compaction_notice:
            yield f"\n{compaction_notice}\n\n"
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
        """Agentic loop: generate → detect tool calls → execute → re-generate.

        Uses a nudge-then-break strategy for loop detection:
        1. First duplicate → gentle nudge asking model to explore deeper
        2. Second consecutive duplicate → force a text answer
        This allows models to self-correct instead of being killed mid-analysis.
        """
        tools_defs = self._tools.to_openai_tools()
        seen_calls: set[str] = set()
        nudge_count = 0  # consecutive nudges without progress

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

            # --- Sanitize tool params: some models pass schema dicts instead of values ---
            for tc in tool_calls:
                if not isinstance(tc.parameters, dict):
                    logger.warning("Tool %s has non-dict parameters: %r, replacing with empty dict", tc.tool_name, type(tc.parameters))
                    tc.parameters = {}
                sanitized = {}
                for k, v in tc.parameters.items():
                    if isinstance(v, dict) and "type" in v and "description" in v:
                        # Model passed the JSON schema instead of a value — use a
                        # sensible default based on the param name and type.
                        if k == "path":
                            v = "."
                        elif v.get("type") == "string":
                            v = ""
                        else:
                            v = ""
                        logger.warning(
                            "Sanitized schema-as-value param %s in tool %s",
                            k, tc.tool_name,
                        )
                    sanitized[k] = v
                tc.parameters = sanitized

            # --- Loop detection (nudge-then-break) ---
            def _normalize_params(params: dict) -> dict:
                normalized = {}
                for k, v in params.items():
                    if isinstance(v, str) and ("/" in v or v == "."):
                        v = os.path.normpath(v)
                    normalized[k] = v
                return normalized

            call_keys = frozenset(
                f"{tc.tool_name}:{json.dumps(_normalize_params(tc.parameters), sort_keys=True)}"
                for tc in tool_calls
            )
            new_calls = call_keys - seen_calls

            if not new_calls:
                nudge_count += 1
                if nudge_count >= 2:
                    # Repeated duplicates after nudge — force text answer
                    logger.warning("Loop detected after nudge, forcing text answer")
                    async for chunk in self._force_text_answer():
                        yield chunk
                    return
                # First duplicate — nudge the model to try different tools/paths
                logger.info("Duplicate tool calls detected, nudging model (attempt %d)", nudge_count)
                already_called = ", ".join(
                    f"{k.split(':', 1)[0]}({k.split(':', 1)[1][:60]})"
                    for k in seen_calls
                )
                nudge_msg = Message(
                    role=Role.USER,
                    content=(
                        f"[System]: You already called these tools: {already_called}. "
                        "Do NOT repeat them. Instead, try a DIFFERENT approach:\n"
                        "- Use read_file to look at specific files (e.g. README.md, main config files)\n"
                        "- Use list_directory on subdirectories to explore deeper\n"
                        "- Use terminal to run 'find' or 'grep' for specific patterns\n"
                        "Choose a different tool or different arguments now."
                    ),
                )
                self._context.add_message(nudge_msg)
                continue  # Let the model try again
            else:
                nudge_count = 0  # reset on progress

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
        warning_text = "⚠️ Reached maximum tool iterations. Please try a more specific question."
        yield f"\n{warning_text}\n"
        self._context.add_message(Message(role=Role.ASSISTANT, content=warning_text))

    async def _force_text_answer(self) -> AsyncIterator[str]:
        """Force the model to produce a text-only answer (no tools)."""
        force_msg = Message(
            role=Role.USER,
            content=(
                "[System]: STOP calling tools. Provide your final answer in plain text "
                "based on all the information you have gathered so far. "
                "Do NOT output JSON or tool calls. "
                "Respond in the same language as the user's original question."
            ),
        )
        self._context.add_message(force_msg)
        final = await self._backend.generate(
            self._context.get_messages(),
            max_tokens=self._settings.inference.max_tokens,
            temperature=0.7,
        )
        text = final.text or ""
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

    @staticmethod
    def _parse_text_tool_calls(text: str) -> list[ToolCall]:
        """Parse tool calls from model text output (fallback for non-native).

        Uses balanced-brace extraction instead of regex to handle nested JSON.
        """
        calls: list[ToolCall] = []
        decoder = json.JSONDecoder()
        i = 0
        call_index = 0
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
                        call_id=f"text_{obj['name']}_{call_index}",
                    ))
                    call_index += 1
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
        """Switch to a cloud API backend (disables auto-selection)."""
        await self._backend.close()
        if provider == "claude":
            self._backend = ClaudeBackend(self._settings)
            self._active_model = self._settings.api.claude_model
        elif provider == "openai":
            self._backend = OpenAIBackend(self._settings)
            self._active_model = self._settings.api.openai_model
        elif provider == "ollama":
            self._backend = OllamaBackend(
                self._settings, num_ctx=self._settings.inference.context_size,
            )
            self._active_model = self._settings.ollama.model
        else:
            return f"Unknown provider: {provider}"
        self._manual_model_override = True
        return f"Switched to API: {self._active_model}"

    async def switch_ollama_model(self, model: str) -> str:
        """Switch to a specific Ollama model (disables auto-selection).

        Queries Ollama for the model's native context length and uses
        min(model_max, config_max) so we don't exceed either limit.
        """
        num_ctx = await self._resolve_num_ctx(model)
        if isinstance(self._backend, OllamaBackend):
            self._backend.switch_model(model, num_ctx=num_ctx)
        else:
            await self._backend.close()
            self._backend = OllamaBackend(
                self._settings, model=model, num_ctx=num_ctx,
            )
        self._active_model = model
        self._manual_model_override = True
        # Update context manager window to match
        if self._context:
            self._context._context_window = num_ctx
        return f"Switched to Ollama model: {model} (ctx: {num_ctx:,})"

    async def _resolve_num_ctx(self, model: str | None = None) -> int:
        """Determine optimal num_ctx for a model based on hardware constraints.

        Queries Ollama for the model's native context_length, then caps it
        based on available memory (VRAM + RAM) minus the model weight footprint.
        Each KV-cache token costs approximately *kv_bytes_per_token* bytes,
        which varies by model size (hidden_dim × layers × 2 × 2 bytes).
        """
        config_max = self._settings.inference.context_size  # user upper-bound

        # 1. Get model's native context length from Ollama metadata
        model_max = 0
        if isinstance(self._backend, OllamaBackend):
            model_max = await self._backend.get_model_context_length(model)

        # Fallback to ModelSelector static map
        if not model_max and self._model_selector and model:
            model_max = self._model_selector._MAX_CTX.get(model, 0)

        if not model_max:
            return config_max

        # 2. Estimate safe context based on hardware
        try:
            from codator.infrastructure.hardware import check_hardware
            hw, _ = check_hardware()
            vram_mb = hw.vram_total_mb
            ram_mb = hw.ram_total_mb
        except Exception:
            vram_mb, ram_mb = 16_304, 31_237  # fallback to known values

        # Estimate model VRAM footprint from selector or catalog
        model_vram_mb = 0
        if self._model_selector and model:
            model_vram_mb = self._model_selector.estimate_vram_mb(model)

        # Total usable memory = VRAM + (RAM - 4GB for OS)
        total_mem_mb = vram_mb + max(0, ram_mb - 4_096)
        free_for_kv_mb = max(0, total_mem_mb - model_vram_mb - 2_048)  # 2GB headroom

        # KV cache cost per token depends on model size (rough estimates)
        # hidden_dim × num_layers × 2(K+V) × 2(bytes) per token
        model_lower = (model or "").lower()
        if "70b" in model_lower or "72b" in model_lower:
            kv_bytes_per_token = 8192 * 80 * 2 * 2  # ~2.5 MB/token
        elif "32b" in model_lower or "33b" in model_lower or "34b" in model_lower:
            kv_bytes_per_token = 5120 * 64 * 2 * 2  # ~1.25 MB/token
        elif "14b" in model_lower or "13b" in model_lower:
            kv_bytes_per_token = 5120 * 40 * 2 * 2  # ~0.78 MB/token
        elif "7b" in model_lower or "8b" in model_lower:
            kv_bytes_per_token = 4096 * 32 * 2 * 2  # ~0.5 MB/token
        else:
            kv_bytes_per_token = 2048 * 24 * 2 * 2  # small models ~0.19 MB/token

        kv_mb_per_token = kv_bytes_per_token / (1024 * 1024)
        hw_safe_ctx = int(free_for_kv_mb / kv_mb_per_token) if kv_mb_per_token > 0 else config_max

        # Round down to nearest 1024 for cleanliness
        hw_safe_ctx = max(2048, (hw_safe_ctx // 1024) * 1024)

        # 3. Final: min of (model native, hardware safe, config upper bound)
        num_ctx = min(model_max, hw_safe_ctx, config_max)
        # But never go below 4096
        num_ctx = max(4096, num_ctx)

        logger.info(
            "Resolved num_ctx for %s: %d (model_native=%d, hw_safe=%d, config=%d, "
            "free_mem=%dMB, model_vram=%dMB)",
            model, num_ctx, model_max, hw_safe_ctx, config_max,
            free_for_kv_mb, model_vram_mb,
        )
        return num_ctx

    # ----- State -----

    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def auto_select_enabled(self) -> bool:
        return not self._manual_model_override

    def set_auto_select(self, enabled: bool) -> str:
        """Enable or disable automatic model selection."""
        self._manual_model_override = not enabled
        state = "enabled" if enabled else "disabled"
        return f"Auto model selection {state}"

    @property
    def model_selector(self) -> ModelSelector | None:
        return self._model_selector

    @property
    def contextual_index(self) -> ContextualIndex | None:
        return self._contextual_index

    @property
    def mode(self) -> str:
        """Current interaction mode: 'chat' or 'agent'."""
        return self._mode

    def set_mode(self, mode: str) -> None:
        """Switch interaction mode between 'chat' and 'agent'."""
        self._mode = mode

    async def start_web(self) -> str:
        """Start the web dashboard in the background. Returns status message."""
        import asyncio

        import uvicorn

        from codator.web.app import create_app

        # Stop existing server if running
        if self._web_server is not None:
            self._web_server.should_exit = True
            if self._web_task and not self._web_task.done():
                try:
                    await asyncio.wait_for(self._web_task, timeout=3)
                except asyncio.TimeoutError:
                    self._web_task.cancel()
            self._web_server = None
            self._web_task = None

        app = create_app(self)
        web_cfg = self._settings.web
        config = uvicorn.Config(
            app, host=web_cfg.host, port=web_cfg.port, log_level="warning",
        )
        self._web_server = uvicorn.Server(config)
        self._web_task = asyncio.create_task(self._web_server.serve())
        return f"Web dashboard: http://{web_cfg.host}:{web_cfg.port}"

    async def restart_web(self) -> str:
        """Restart the web dashboard."""
        return await self.start_web()

    def set_confirm_callback(self, cb: ConfirmCallback | None) -> None:
        """Set the human-in-the-loop confirmation callback for write/edit tools."""
        self._confirm_callback = cb

    @property
    def context_status(self) -> dict[str, Any]:
        status = self._context.status_dict() if self._context else {}
        # Add num_ctx actually sent to Ollama for transparency
        if isinstance(self._backend, OllamaBackend):
            status["num_ctx"] = self._backend.num_ctx
        return status

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

    def get_recent_context_summary(self, max_messages: int = 10) -> str:
        """Return a brief summary of recent chat for passing to agent mode."""
        if not self._context:
            return ""
        messages = self._context.get_messages()
        conv = [m for m in messages if m.role not in (Role.SYSTEM, Role.SUMMARY)]
        recent = conv[-max_messages:]
        if not recent:
            return ""
        parts = []
        for m in recent:
            label = m.role.value.upper()
            content = m.content
            if len(content) > 500:
                content = content[:500] + "... [truncated]"
            parts.append(f"[{label}]: {content}")
        return "=== Recent Chat Context ===\n" + "\n\n".join(parts) + "\n=== End Chat Context ==="

    # ----- Tool execution -----

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    async def execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        return await self._tools.execute(call)
