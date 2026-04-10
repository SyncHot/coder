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
                print_error("Usage: /model <path-to-gguf-file>")
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
            web_cfg = engine._settings.web
            print_info(f"Starting web dashboard on http://{web_cfg.host}:{web_cfg.port} ...")
            import asyncio

            import uvicorn

            from codator.web.app import create_app
            app = create_app(engine)
            config = uvicorn.Config(app, host=web_cfg.host, port=web_cfg.port, log_level="warning")
            server = uvicorn.Server(config)
            asyncio.create_task(server.serve())
            print_info("Web dashboard running in background.")

        case "/ssh":
            await _handle_ssh(arg, engine)

        case "/browser":
            await _handle_browser(arg, engine)

        case "/terminal" | "/term" | "/shell":
            await _handle_terminal(arg, engine)

        case "/ollama":
            await _handle_ollama(arg, engine)

        case "/agent":
            await _handle_agent(arg, engine)

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

        case _:
            print_error(f"Unknown command: {command}. Type /help for available commands.")

    return False


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


async def _handle_agent(arg: str, engine: ChatEngine) -> None:
    """Handle /agent <task description> — run Plan-Act-Verify cycle."""
    if not arg.strip():
        print_info(
            "Usage: /agent <task description>\n"
            "  Runs a Plan-Act-Verify cycle to accomplish the task.\n"
            "  Example: /agent Add docstrings to all functions in config.py"
        )
        return

    from rich.table import Table

    from codator.core.agent_loop import PlanActVerifyAgent

    ollama_cfg = engine._settings.ollama
    agent = PlanActVerifyAgent(
        ollama_base_url=ollama_cfg.base_url,
        model=engine.active_model or ollama_cfg.model,
        project_root=engine._project_root,
    )

    step_log: list[tuple[str, str]] = []

    def on_step(description: str, status: str) -> None:
        icon = {"started": "🔄", "running": "⏳", "done": "✅", "failed": "❌"}.get(
            status, "•"
        )
        step_log.append((icon, description))
        # Print inline
        console.print(f"  {icon} {description}")

    print_info(f"🤖 Agent starting: {arg}")
    try:
        result = await agent.run(arg, on_step=on_step)

        # Summary
        table = Table(title="Agent Result", border_style="cyan")
        table.add_column("Metric", style="bold")
        table.add_column("Value")
        table.add_row("Task", result.plan.task)
        table.add_row("Steps executed", str(len(result.actions)))
        table.add_row("Heal iterations", str(result.heal_iterations))
        table.add_row(
            "Final status",
            "[green]SUCCESS[/green]" if result.final_success else "[red]FAILED[/red]",
        )
        if result.verification.errors:
            table.add_row("Errors", "\n".join(result.verification.errors[:5]))
        if result.verification.warnings:
            table.add_row("Warnings", "\n".join(result.verification.warnings[:5]))
        console.print(table)
    except Exception as exc:
        print_error(f"Agent failed: {exc}")


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
