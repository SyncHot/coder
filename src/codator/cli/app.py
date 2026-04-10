"""Main CLI application — async chat loop with rich + prompt_toolkit."""

from __future__ import annotations

import argparse
import asyncio
import logging

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory

from codator.cli.commands import handle_command
from codator.cli.rendering import (
    console,
    print_compaction_notice,
    print_error,
    print_info,
    print_streaming_token,
    print_welcome,
)
from codator.config import load_config
from codator.core.chat_engine import ChatEngine
from codator.infrastructure.hardware import check_hardware

logger = logging.getLogger("codator")

# Slash-command auto-completion
COMMANDS = [
    "/help", "/quit", "/exit", "/model", "/api", "/context",
    "/clear", "/hardware", "/index", "/git", "/web",
    "/ssh", "/browser", "/terminal", "/ollama",
    "/agent", "/gpu", "/memory", "/mcp",
    "/save", "/load", "/history",
]
command_completer = WordCompleter(COMMANDS, sentence=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="codator",
        description="Dev-Assistant-OS — local AI coding assistant",
    )
    parser.add_argument("--model", "-m", help="Path to GGUF model file")
    parser.add_argument(
        "--api", choices=["claude", "openai", "ollama"],
        help="Use cloud API or local Ollama instead of GGUF model",
    )
    parser.add_argument("--project", "-p", default=".", help="Project root directory")
    parser.add_argument("--web", action="store_true", help="Also start web dashboard")
    parser.add_argument("--config", "-c", help="Path to config TOML file")
    parser.add_argument("--context-size", type=int, help="Override context window size")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    return parser.parse_args()


async def async_main():
    args = parse_args()

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s %(levelname)s: %(message)s")

    # Config
    config_path = None
    if args.config:
        from pathlib import Path
        config_path = Path(args.config)
    settings = load_config(config_path)

    if args.context_size:
        settings.inference.context_size = args.context_size
    if args.api:
        settings.api.provider = args.api

    # Hardware summary
    hw, recs = check_hardware()
    hw_summary = f"GPU: {hw.gpu_name} | VRAM: {hw.vram_total_mb:,} MB | RAM: {hw.ram_total_mb:,} MB"

    # Chat engine
    engine = ChatEngine(settings=settings, project_root=args.project)

    # Wire up write/edit confirmation callback for CLI
    async def _cli_confirm(tool_name: str, summary: str) -> bool:
        console.print(f"\n⚠️  [bold yellow]{tool_name}[/]: {summary}")
        try:
            answer = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("Allow? [y/N] ").strip().lower(),
            )
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    engine.set_confirm_callback(_cli_confirm)

    print_info("Initializing...")
    await engine.initialize(model_path=args.model)

    print_welcome(engine.active_model, hw_summary)

    # Web dashboard (background)
    if args.web:
        import uvicorn

        from codator.web.app import create_app
        app = create_app(engine)
        web_cfg = engine._settings.web
        uv_config = uvicorn.Config(app, host=web_cfg.host, port=web_cfg.port, log_level="warning")
        server = uvicorn.Server(uv_config)
        asyncio.create_task(server.serve())
        print_info(f"Web dashboard: http://{web_cfg.host}:{web_cfg.port}")

    # Interactive prompt session
    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        completer=command_completer,
    )

    try:
        while True:
            try:
                # prompt_toolkit is sync; run in executor
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: session.prompt("you> "),
                )
            except (EOFError, KeyboardInterrupt):
                print_info("\nGoodbye! 👋")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # Slash commands
            if user_input.startswith("/"):
                should_exit = await handle_command(user_input, engine)
                if should_exit:
                    break
                continue

            # Chat with streaming
            ctx_before = engine.context_status.get("total_tokens", 0)
            try:
                console.print("[bold green]codator>[/bold green] ", end="")
                full_response: list[str] = []

                async for token in engine.chat_stream(user_input):
                    print_streaming_token(token)
                    full_response.append(token)

                console.print()  # newline after streaming

                # Check if compaction happened
                ctx_after = engine.context_status.get("total_tokens", 0)
                if engine.context_status.get("compaction_count", 0) > 0 and ctx_after < ctx_before:
                    print_compaction_notice(ctx_before, ctx_after)

            except FileNotFoundError as exc:
                print_error(str(exc))
            except Exception as exc:
                print_error(f"Generation failed: {exc}")
                logger.exception("Chat error")

    finally:
        await engine.shutdown()


def main():
    """Synchronous entry point."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
