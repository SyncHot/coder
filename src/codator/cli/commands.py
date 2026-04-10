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
            if arg not in ("claude", "openai"):
                print_error("Usage: /api <claude|openai>")
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
