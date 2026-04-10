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

        case _:
            print_error(f"Unknown command: {command}. Type /help for available commands.")

    return False
