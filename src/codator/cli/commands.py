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
            await engine.initialize()
            print_info("Conversation cleared and context reset.")

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
            print_info("Starting web dashboard on http://localhost:8000 ...")
            import asyncio

            import uvicorn

            from codator.web.app import create_app
            app = create_app(engine)
            config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
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

    from rich.live import Live
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
