"""CLI commands — slash-command handler for the interactive loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codator.cli.rendering import (
    console,
    print_context_status,
    print_error,
    print_hardware_info,
    print_help,
    print_info,
    print_tool_result,
)
from codator.infrastructure.hardware import check_hardware

if TYPE_CHECKING:
    from codator.core.chat_engine import ChatEngine


async def handle_command(cmd: str, engine: ChatEngine) -> bool:
    """Handle a slash command. Returns True if the app should exit."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    match command:
        case "/quit" | "/exit" | "/q":
            print_info("Goodbye! 👋")
            return True

        case "/help" | "/h":
            print_help()

        case "/model":
            if not arg:
                await _handle_model_picker(engine)
            else:
                print_info(f"Loading model: {arg}...")
                result = await engine.switch_model(arg)
                print_info(result)

        case "/api":
            if arg not in ("claude", "openai", "ollama"):
                print_error("Usage: /api <claude|openai|ollama>")
            else:
                result = await engine.switch_to_api(arg)
                print_info(result)

        case "/context" | "/ctx":
            print_context_status(engine.context_status)

        case "/clear":
            engine.clear_context()
            print_info("Conversation cleared.")

        case "/hardware" | "/hw":
            hw, recs = check_hardware()
            print_hardware_info(hw, recs)

        case "/index":
            print_info("Re-indexing project...")
            result = await engine.refresh_project_context()
            print_info(result)

        case "/git":
            from codator.core.git_integration import GitContext
            git_ctx = GitContext(".")
            console.print(git_ctx.get_work_context())

        case "/web":
            result = await engine.start_web()
            print_info(result)

        case "/restart":
            print_info("Restarting web dashboard...")
            result = await engine.restart_web()
            print_info(result)

        case "/ssh":
            await _handle_ssh(arg, engine)

        case "/browser":
            await _handle_browser(arg, engine)

        case "/terminal" | "/term" | "/shell":
            await _handle_terminal(arg, engine)

        case "/ollama":
            await _handle_ollama(arg, engine)

        case "/agent":
            if not arg.strip():
                engine.set_mode("agent")
                print_info("🤖 Switched to **agent mode**. All messages will run Plan-Act-Verify.\n   Type /chat to switch back to chat mode.")
            else:
                await _handle_agent(arg, engine)

        case "/chat":
            engine.set_mode("chat")
            print_info("💬 Switched to **chat mode**.")

        case "/gpu":
            await _handle_gpu(engine)

        case "/memory":
            await _handle_memory(engine)

        case "/mcp":
            await _handle_mcp(arg, engine)

        case "/save":
            await _handle_save(arg, engine)

        case "/load":
            await _handle_load(arg, engine)

        case "/history":
            await _handle_history(engine)

        case "/undo":
            await _handle_undo(arg, engine)

        case "/fetch":
            await _handle_fetch(arg, engine)

        case "/search":
            await _handle_search(arg, engine)

        case _:
            print_error(f"Unknown command: {command}. Type /help for available commands.")

    return False


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------


async def _handle_model_picker(engine: ChatEngine) -> None:
    """Interactive model picker — lists available Ollama models, user picks by number."""
    from codator.infrastructure.ollama_backend import OllamaBackend
    from rich.table import Table

    print_info("Fetching available models from Ollama...")
    backend = OllamaBackend(engine._settings)
    models = await backend.list_models()
    await backend.close()

    if not models:
        print_error("No models found. Is Ollama running?")
        return

    # Sort by name, show table with numbers
    models.sort(key=lambda m: m.get("name", ""))
    table = Table(title="Available Models", border_style="cyan")
    table.add_column("#", style="bold yellow", justify="right")
    table.add_column("Name", style="bold")
    table.add_column("Size")

    for i, m in enumerate(models, 1):
        name = m.get("name", "?")
        size_bytes = m.get("size", 0)
        size_gb = f"{size_bytes / 1_073_741_824:.1f} GB"
        active = " ← active" if name == engine.active_model else ""
        table.add_row(str(i), f"{name}{active}", size_gb)

    console.print(table)
    console.print("[dim]Enter number to switch, or press Enter to cancel:[/dim]")

    import asyncio
    try:
        choice = await asyncio.get_running_loop().run_in_executor(
            None, lambda: input("model #> ").strip(),
        )
    except (EOFError, KeyboardInterrupt):
        print_info("Cancelled.")
        return

    if not choice:
        print_info("Cancelled.")
        return

    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(models):
            print_error(f"Invalid choice. Pick 1-{len(models)}.")
            return
    except ValueError:
        # Treat as direct model name
        result = await engine.switch_ollama_model(choice)
        print_info(result)
        return

    selected = models[idx]["name"]
    result = await engine.switch_ollama_model(selected)
    print_info(result)


# ---------------------------------------------------------------------------
# Tool command handlers
# ---------------------------------------------------------------------------

async def _handle_ssh(arg: str, engine: ChatEngine) -> None:
    """Handle /ssh <action> [args...]."""
    from codator.domain.models import ToolCall

    parts = arg.strip().split(maxsplit=1)
    action = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if not action:
        print_info(
            "Usage: /ssh connect <host> <user> [password]\n"
            "       /ssh exec <command>\n"
            "       /ssh upload <local> <remote>\n"
            "       /ssh download <remote> <local>\n"
            "       /ssh close"
        )
        return

    params: dict = {"action": action}

    match action:
        case "connect":
            tokens = rest.split()
            if len(tokens) < 2:
                print_error("Usage: /ssh connect <host> <user> [password]")
                return
            params["host"] = tokens[0]
            params["username"] = tokens[1]
            if len(tokens) > 2:
                params["password"] = tokens[2]
        case "exec":
            if not rest:
                print_error("Usage: /ssh exec <command>")
                return
            params["command"] = rest
        case "upload":
            tokens = rest.split()
            if len(tokens) != 2:
                print_error("Usage: /ssh upload <local_path> <remote_path>")
                return
            params["local_path"] = tokens[0]
            params["remote_path"] = tokens[1]
        case "download":
            tokens = rest.split()
            if len(tokens) != 2:
                print_error("Usage: /ssh download <remote_path> <local_path>")
                return
            params["remote_path"] = tokens[0]
            params["local_path"] = tokens[1]
        case "close":
            pass
        case _:
            print_error(f"Unknown SSH action: {action}")
            return

    call = ToolCall(tool_name="ssh", parameters=params)
    result = await engine.execute_tool(call)
    print_tool_result(result)


async def _handle_browser(arg: str, engine: ChatEngine) -> None:
    """Handle /browser <action> [args...]."""
    from codator.domain.models import ToolCall

    parts = arg.strip().split(maxsplit=1)
    action = parts[0] if parts else ""
    rest = parts[1] if len(parts) > 1 else ""

    if not action:
        print_info(
            "Usage: /browser launch\n"
            "       /browser navigate <url>\n"
            "       /browser click <selector>\n"
            "       /browser fill <selector> <value>\n"
            "       /browser screenshot [path]\n"
            "       /browser text [selector]\n"
            "       /browser close"
        )
        return

    params: dict = {"action": action}

    match action:
        case "launch":
            pass
        case "navigate":
            params["url"] = rest
        case "click":
            params["selector"] = rest
        case "fill":
            tokens = rest.split(maxsplit=1)
            if len(tokens) != 2:
                print_error("Usage: /browser fill <selector> <value>")
                return
            params["selector"] = tokens[0]
            params["value"] = tokens[1]
        case "screenshot":
            if rest:
                params["path"] = rest
        case "text":
            if rest:
                params["selector"] = rest
        case "close":
            pass
        case _:
            print_error(f"Unknown browser action: {action}")
            return

    call = ToolCall(tool_name="browser", parameters=params)
    result = await engine.execute_tool(call)
    print_tool_result(result)


async def _handle_terminal(arg: str, engine: ChatEngine) -> None:
    """Handle /terminal <command>."""
    from codator.domain.models import ToolCall

    if not arg.strip():
        print_info("Usage: /terminal <shell command>")
        return

    call = ToolCall(tool_name="terminal", parameters={"command": arg.strip()})
    result = await engine.execute_tool(call)
    print_tool_result(result)


async def _handle_ollama(arg: str, engine: ChatEngine) -> None:
    """Handle /ollama [list|use <model>]."""
    from codator.infrastructure.ollama_backend import OllamaBackend

    parts = arg.strip().split(maxsplit=1)
    action = parts[0] if parts else "list"
    rest = parts[1] if len(parts) > 1 else ""

    match action:
        case "list" | "ls" | "":
            print_info("Fetching models from Ollama...")
            backend = OllamaBackend(engine._settings)
            models = await backend.list_models()
            await backend.close()
            if not models:
                print_error("No models found. Is Ollama running?")
                return
            from rich.table import Table
            table = Table(title="Ollama Models", border_style="cyan")
            table.add_column("Name", style="bold")
            table.add_column("Size")
            table.add_column("Modified")
            for m in models:
                name = m.get("name", "?")
                size_bytes = m.get("size", 0)
                size_gb = f"{size_bytes / 1_073_741_824:.1f} GB"
                modified = m.get("modified_at", "?")[:10]
                active = " ← active" if name == engine.active_model else ""
                table.add_row(f"{name}{active}", size_gb, modified)
            console.print(table)

        case "use" | "switch":
            if not rest:
                print_error("Usage: /ollama use <model-name>")
                return
            result = await engine.switch_ollama_model(rest)
            print_info(result)

        case _:
            # Treat as model name shortcut: /ollama qwen2.5-coder:14b
            result = await engine.switch_ollama_model(action)
            print_info(result)


async def run_agent_task(task: str, engine: ChatEngine) -> None:
    """Run the interactive agent cycle: Analyze → Propose → Pick → Implement.

    Uses streaming to show model thinking in real-time (like Claude).
    """
    import asyncio

    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table

    from codator.core.agent_loop import PlanActVerifyAgent, Proposal

    async def _confirm_command(command: str, reason: str) -> bool:
        """Interactive confirmation for dangerous agent commands."""
        from rich import print as rprint

        rprint(f"[bold yellow]⚠️  Agent wants to run:[/bold yellow] {command}")
        rprint(f"[dim]Reason: {reason}[/dim]")
        answer = await asyncio.get_event_loop().run_in_executor(
            None, lambda: input("Allow? (y/N): ").strip().lower()
        )
        return answer in ("y", "yes")

    ollama_cfg = engine._settings.ollama
    agent = PlanActVerifyAgent(
        ollama_base_url=ollama_cfg.base_url,
        model=engine.active_model or ollama_cfg.model,
        project_root=engine._project_root,
        num_ctx=engine._settings.inference.context_size,
        confirm_callback=_confirm_command,
    )

    # --- Helpers ---

    def on_step(description: str, status: str) -> None:
        icon = {"started": "🔄", "running": "⏳", "done": "✅", "failed": "❌"}.get(
            status, "•"
        )
        console.print(f"  {icon} {description}")

    # Track whether we're showing thinking vs summary
    _thinking_phase = {"active": False, "label": "", "has_output": False}

    def on_token(token: str) -> None:
        """Print model tokens in real-time — dimmed for thinking."""
        if _thinking_phase.get("active"):
            if not _thinking_phase.get("has_output"):
                console.print("[dim italic]  💭 thinking…[/dim italic]")
                _thinking_phase["has_output"] = True
            console.print(f"[dim]{token}[/dim]", end="", highlight=False)

    def _start_thinking(label: str = "thinking") -> None:
        _thinking_phase["active"] = True
        _thinking_phase["label"] = label
        _thinking_phase["has_output"] = False

    def _stop_thinking() -> None:
        if _thinking_phase.get("has_output"):
            console.print()  # newline after streamed tokens
        _thinking_phase["active"] = False

    # --- Build project context ---

    chat_context = engine.get_recent_context_summary()
    project_context = _build_project_context(engine)
    if chat_context:
        project_context = chat_context + "\n\n" + project_context

    print_info(f"🤖 Agent starting: {task}")

    try:
        # ================================================================
        # Phase 1: Analyze & Propose
        # ================================================================
        console.print()
        console.print("[bold cyan]━━━ Phase 1: Analysis ━━━[/bold cyan]")

        _start_thinking("analysis")

        proposals, analysis_actions = await agent.analyze_and_propose(
            task,
            project_context=project_context,
            on_step=on_step,
            on_token=on_token,
        )

        _stop_thinking()

        if not proposals:
            console.print("[yellow]No proposals generated. Try a different question.[/yellow]")
            return

        # ================================================================
        # Phase 2: Present proposals to user
        # ================================================================
        console.print()
        console.print("[bold cyan]━━━ Proposals ━━━[/bold cyan]")
        console.print()

        _priority_colors = {"high": "red", "medium": "yellow", "low": "green"}

        for p in proposals:
            color = _priority_colors.get(p.priority, "white")
            console.print(
                Panel(
                    f"{p.description}\n[dim]File: {p.file}[/dim]",
                    title=f"[bold][{color}]{p.index + 1}. [{p.priority.upper()}] {p.title}[/{color}][/bold]",
                    border_style=color,
                    padding=(0, 1),
                )
            )

        # ================================================================
        # Phase 3: User picks which proposals to implement
        # ================================================================
        console.print()
        console.print(
            "[bold]What next?[/bold] Enter numbers to implement "
            "(e.g. [cyan]1,3,5[/cyan] or [cyan]all[/cyan]), "
            "[cyan]none[/cyan] to skip, or type a question/instruction."
        )

        try:
            choice = await asyncio.get_running_loop().run_in_executor(
                None, lambda: input("implement> ").strip(),
            )
        except (EOFError, KeyboardInterrupt):
            print_info("Cancelled.")
            return

        choice_lower = choice.lower()

        if not choice or choice_lower == "none":
            print_info("Done.")
            return

        if choice_lower == "all":
            selected = proposals
        else:
            # Try to parse comma-separated numbers first
            selected_indices: set[int] = set()
            is_numeric = True
            for part in choice_lower.replace(" ", "").split(","):
                try:
                    idx = int(part) - 1  # 1-based → 0-based
                    if 0 <= idx < len(proposals):
                        selected_indices.add(idx)
                except ValueError:
                    is_numeric = False
                    break

            if not is_numeric or not selected_indices:
                # Freeform text — answer the question using gathered analysis
                console.print()
                console.print("[bold cyan]━━━ Answering ━━━[/bold cyan]")
                _start_thinking("answering")

                answer_prompt = (
                    f"User's original request:\n{task}\n\n"
                    f"Proposals generated:\n"
                    + "\n".join(
                        f"  {p.index + 1}. [{p.priority}] {p.title}: {p.description} (file: {p.file})"
                        for p in proposals
                    )
                    + f"\n\nUser's follow-up question/instruction:\n{choice}\n\n"
                    "Answer in a clear, conversational way. "
                    "Respond in the same language as the user."
                )
                from codator.core.agent_loop import PlanActVerifyAgent

                answer_raw = await agent._ollama_chat(
                    "You are a helpful code assistant. Answer questions about the "
                    "analysis and proposals you generated. Be concise and specific.",
                    answer_prompt,
                    force_json=False,
                    on_token=on_token,
                )
                _stop_thinking()
                # The answer was already streamed via on_token; print a newline
                console.print()
                return

            selected = [p for p in proposals if p.index in selected_indices]

        console.print()
        console.print(
            f"[bold green]Implementing {len(selected)} proposal(s)…[/bold green]"
        )

        # ================================================================
        # Phase 4: Implement selected proposals
        # ================================================================
        console.print()
        console.print("[bold cyan]━━━ Phase 2: Implementation ━━━[/bold cyan]")

        _start_thinking("implementing")

        result = await agent.implement_proposals(
            selected,
            task=task,
            project_context=project_context,
            on_step=on_step,
            on_token=on_token,
        )

        _stop_thinking()

        # ================================================================
        # Results
        # ================================================================
        table = Table(title="Implementation Result", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Proposals implemented", str(len(selected)))
        table.add_row("Steps executed", str(len(result.actions)))
        table.add_row("Heal iterations", str(result.heal_iterations))
        table.add_row(
            "Final status",
            "[green]SUCCESS[/green]" if result.final_success else "[red]FAILED[/red]",
        )
        if result.verification.errors:
            table.add_row("Errors", "\n".join(result.verification.errors[:5]))
        console.print(table)

    except Exception as exc:
        _stop_thinking()
        print_error(f"Agent failed: {exc}")


def _build_project_context(engine: ChatEngine) -> str:
    """Scan project root for code files and return a file-tree string."""
    from pathlib import Path
    try:
        root = Path(engine._project_root).resolve()
        code_exts = {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".rb"}
        files = []
        for f in sorted(root.rglob("*")):
            if f.is_file() and f.suffix in code_exts:
                rel = f.relative_to(root)
                parts = rel.parts
                if any(p.startswith(".") or p in (
                    "__pycache__", "node_modules", ".venv", "venv",
                ) for p in parts):
                    continue
                files.append(str(rel))
        if files:
            ctx = "Project file tree:\n" + "\n".join(files[:200])
            if len(files) > 200:
                ctx += f"\n... and {len(files) - 200} more files"
            return ctx
    except Exception:
        pass
    return ""


async def _handle_agent(arg: str, engine: ChatEngine) -> None:
    """Handle /agent <task description> — run Plan-Act-Verify cycle."""
    if not arg.strip():
        print_info(
            "Usage: /agent <task description>\n"
            "  Runs a Plan-Act-Verify cycle to accomplish the task.\n"
            "  Example: /agent Add docstrings to all functions in config.py\n"
            "  Or type /agent (no args) to enter persistent agent mode."
        )
        return
    await run_agent_task(arg, engine)


async def _handle_gpu(engine: ChatEngine) -> None:
    """Handle /gpu — show real-time GPU and Ollama stats."""
    from rich.table import Table

    from codator.infrastructure.gpu_monitor import GPUMonitor

    print_info("Fetching GPU stats...")
    monitor = GPUMonitor()
    status = await monitor.get_full_status()

    gpu = status["gpu"]
    ollama = status["ollama"]

    table = Table(title="GPU Status (ROCm)", border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("GPU", gpu["name"])
    table.add_row(
        "VRAM",
        f"{gpu['vram_used_mb']:,} / {gpu['vram_total_mb']:,} MB "
        f"({gpu['vram_free_mb']:,} MB free)",
    )
    table.add_row("Utilization", f"{gpu['utilization_pct']:.1f}%")
    table.add_row("Temperature", f"{gpu['temperature_c']}°C")
    console.print(table)

    if ollama["running_models"]:
        m_table = Table(title="Ollama Running Models", border_style="cyan")
        m_table.add_column("Model", style="bold")
        m_table.add_column("Size")
        m_table.add_column("Processor")
        m_table.add_column("Context")
        for m in ollama["running_models"]:
            size_gb = f"{m['size_bytes'] / 1_073_741_824:.1f} GB"
            m_table.add_row(m["name"], size_gb, m["processor"], str(m["num_ctx"]))
        console.print(m_table)
    else:
        print_info("No models currently loaded in Ollama.")


async def _handle_memory(engine: ChatEngine) -> None:
    """Handle /memory — show contextual index stats."""
    from rich.table import Table

    ctx_idx = engine.contextual_index
    table = Table(title="Contextual Index (Memory)", border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    if ctx_idx is None:
        table.add_row("Status", "[yellow]Not initialized[/yellow]")
        table.add_row("Hint", "Run /index to build the index")
    else:
        chunks = ctx_idx._chunks
        files = {c.file_path for c in chunks} if chunks else set()
        table.add_row("Status", "[green]Active[/green]")
        table.add_row("Indexed chunks", str(len(chunks)))
        table.add_row("Indexed files", str(len(files)))
        table.add_row("Vocabulary size", str(len(ctx_idx._vocabulary)))
    console.print(table)

    ms = engine.model_selector
    if ms:
        sel_table = Table(title="Model Selector", border_style="green")
        sel_table.add_column("Property", style="bold")
        sel_table.add_column("Value")
        sel_table.add_row("Available models", str(len(ms._available_models)))
        for name, vram in ms._available_models.items():
            sel_table.add_row(f"  {name}", f"{vram:,} MB VRAM")
        console.print(sel_table)


async def _handle_mcp(arg: str, engine: ChatEngine) -> None:
    """Handle /mcp connect <name> <command...> | list | disconnect <name>."""
    parts = arg.strip().split(maxsplit=2)
    action = parts[0] if parts else ""

    if not action:
        print_info(
            "Usage: /mcp connect <name> <command...>\n"
            "       /mcp list\n"
            "       /mcp disconnect <name>"
        )
        return

    from codator.infrastructure.mcp_client import MCPManager

    if not hasattr(engine, "_mcp_manager"):
        engine._mcp_manager = MCPManager()

    mgr: MCPManager = engine._mcp_manager

    match action:
        case "connect":
            if len(parts) < 3:
                print_error("Usage: /mcp connect <name> <command...>")
                return
            name = parts[1]
            cmd = parts[2].split()
            print_info(f"Connecting to MCP server '{name}'...")
            try:
                count = await mgr.connect_server(name, cmd)
                await mgr.register_all(engine._tools)
                print_info(f"Connected! Registered {count} tools from '{name}'.")
            except Exception as exc:
                print_error(f"Failed to connect: {exc}")

        case "list" | "ls":
            servers = mgr.connected_servers
            if not servers:
                print_info("No MCP servers connected.")
            else:
                from rich.table import Table
                table = Table(title="MCP Servers", border_style="cyan")
                table.add_column("Name", style="bold")
                table.add_column("Status")
                for s in servers:
                    table.add_row(s, "[green]connected[/green]")
                console.print(table)

        case "disconnect":
            if len(parts) < 2:
                print_error("Usage: /mcp disconnect <name>")
                return
            name = parts[1]
            await mgr.disconnect_server(name)
            print_info(f"Disconnected MCP server '{name}'.")

        case _:
            print_error(f"Unknown MCP action: {action}. Use connect/list/disconnect.")


async def _handle_save(arg: str, engine: ChatEngine) -> None:
    """Save current conversation: /save [name]."""
    import uuid

    from codator.core.conversation_store import ConversationStore

    if not engine._context:
        print_error("No active conversation to save.")
        return

    conv_id = arg.strip() or str(uuid.uuid4())[:8]
    store = ConversationStore()
    try:
        messages = engine._context.get_messages()
        store.save(conv_id, messages, title=arg or "", model=engine.active_model)
        print_info(f"Saved conversation as '{conv_id}' ({len(messages)} messages).")
    except Exception as exc:
        print_error(f"Save failed: {exc}")
    finally:
        store.close()


async def _handle_load(arg: str, engine: ChatEngine) -> None:
    """Load a saved conversation: /load <id>."""
    from codator.core.conversation_store import ConversationStore

    if not arg.strip():
        print_error("Usage: /load <conversation-id>. Use /history to list saved.")
        return

    store = ConversationStore()
    try:
        messages = store.load(arg.strip())
        if messages is None:
            print_error(f"Conversation '{arg}' not found.")
            return
        engine.clear_context()
        for msg in messages:
            engine._context.add_message(msg)
        print_info(f"Loaded '{arg}' ({len(messages)} messages).")
    except Exception as exc:
        print_error(f"Load failed: {exc}")
    finally:
        store.close()


async def _handle_history(engine: ChatEngine) -> None:
    """List saved conversations: /history."""
    from rich.table import Table

    from codator.core.conversation_store import ConversationStore

    store = ConversationStore()
    try:
        convs = store.list_conversations(limit=20)
        if not convs:
            print_info("No saved conversations.")
            return
        table = Table(title="Saved Conversations", border_style="cyan")
        table.add_column("ID", style="bold")
        table.add_column("Title")
        table.add_column("Model")
        for c in convs:
            table.add_row(c["id"], c["title"][:60], c["model"])
        console.print(table)
    except Exception as exc:
        print_error(f"History failed: {exc}")
    finally:
        store.close()


async def _handle_undo(arg: str, engine: ChatEngine) -> None:
    """Restore files from .bak backups: /undo [file] or /undo --list."""
    import os

    project_root = engine._settings.project_root

    if arg.strip() == "--list":
        bak_files: list[str] = []
        for root, _dirs, files in os.walk(project_root):
            for f in files:
                if f.endswith(".bak"):
                    rel = os.path.relpath(
                        os.path.join(root, f), project_root,
                    )
                    bak_files.append(rel)
        if not bak_files:
            print_info("No .bak backup files found.")
        else:
            from rich.table import Table
            table = Table(title="Backup Files", border_style="yellow")
            table.add_column("Backup", style="bold")
            table.add_column("Restores to")
            for b in sorted(bak_files):
                table.add_row(b, b.removesuffix(".bak"))
            console.print(table)
        return

    if not arg.strip():
        restored = 0
        for root, _dirs, files in os.walk(project_root):
            for f in files:
                if f.endswith(".bak"):
                    bak = os.path.join(root, f)
                    orig = bak.removesuffix(".bak")
                    os.replace(bak, orig)
                    restored += 1
        if restored:
            print_info(f"Restored {restored} file(s) from .bak backups.")
        else:
            print_info("No .bak backup files found.")
        return

    target = os.path.realpath(os.path.join(project_root, arg.strip()))
    if not target.startswith(os.path.realpath(project_root) + os.sep):
        print_error("Path outside project root.")
        return
    bak = target + ".bak"
    if not os.path.isfile(bak):
        print_error(f"No backup found: {arg}.bak")
        return
    os.replace(bak, target)
    print_info(f"Restored {arg} from backup.")


# ---------------------------------------------------------------------------
# Web tools handlers
# ---------------------------------------------------------------------------

async def _handle_fetch(arg: str, engine: ChatEngine) -> None:
    """Handle /fetch <url>."""
    if not arg.strip():
        print_error("Usage: /fetch <url>")
        return

    from codator.domain.models import ToolCall
    result = await engine._tools.execute(
        ToolCall(tool_name="web_fetch", parameters={"url": arg.strip()})
    )
    print_tool_result(result)


async def _handle_search(arg: str, engine: ChatEngine) -> None:
    """Handle /search <query>."""
    if not arg.strip():
        print_error("Usage: /search <query>")
        return

    from codator.domain.models import ToolCall
    result = await engine._tools.execute(
        ToolCall(tool_name="web_search", parameters={"query": arg.strip()})
    )
    print_tool_result(result)
