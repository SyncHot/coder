"""Main CLI application — async chat loop with rich + prompt_toolkit."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory

from codator.cli.commands import handle_command
from codator.cli.rendering import (
    StreamingMarkdownRenderer,
    console,
    print_compaction_inline,
    print_context_line,
    print_error,
    print_info,
    print_welcome,
)
from codator.config import load_config
from codator.core.chat_engine import ChatEngine
from codator.infrastructure.hardware import check_hardware

logger = logging.getLogger("codator")

_SESSION_FILE = Path.home() / ".codator" / "session.json"

# Slash-command auto-completion
COMMANDS = [
    "/help", "/quit", "/exit", "/model", "/api", "/context",
    "/clear", "/hardware", "/index", "/git", "/web", "/restart",
    "/ssh", "/browser", "/terminal", "/ollama",
    "/agent", "/chat", "/gpu", "/memory", "/mcp",
    "/save", "/load", "/history", "/undo",
    "/fetch", "/search",
]
command_completer = WordCompleter(COMMANDS, sentence=True)


def _load_session() -> dict:
    """Load persisted session state (mode, model)."""
    try:
        if _SESSION_FILE.exists():
            return json.loads(_SESSION_FILE.read_text())
    except Exception:
        pass
    return {}


def _save_session(mode: str, model: str) -> None:
    """Persist session state for next startup."""
    try:
        _SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        _SESSION_FILE.write_text(json.dumps({"mode": mode, "model": model}))
    except Exception:
        pass


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

    # Logging — always log to file for debugging, console level from --verbose
    from pathlib import Path
    log_dir = Path.home() / ".codator" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "codator.log"

    # File handler — always DEBUG, rotates at 5 MB
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    # Console handler — WARNING by default, DEBUG with --verbose
    console_level = logging.DEBUG if args.verbose else logging.WARNING
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(logging.Formatter(
        "%(name)s %(levelname)s: %(message)s",
    ))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info("=" * 60)
    logger.info("codator starting — log file: %s", log_file)
    logger.info("=" * 60)

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
            answer = await asyncio.get_running_loop().run_in_executor(
                None, lambda: input("Allow? [y/N] ").strip().lower(),
            )
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    engine.set_confirm_callback(_cli_confirm)

    print_info("Initializing...")

    import time as _time
    _t0 = _time.monotonic()
    await engine.initialize(model_path=args.model)
    _elapsed = _time.monotonic() - _t0
    if _elapsed > 2.0:
        print_info(f"Ready in {_elapsed:.1f}s")

    # Restore session state (mode + model from last session)
    saved = _load_session()
    if saved.get("mode") in ("agent", "chat"):
        engine.set_mode(saved["mode"])
        if saved["mode"] == "agent":
            print_info("🤖 Restored **agent mode** from last session.")
    if saved.get("model") and not args.model:
        # Only restore model if user didn't explicitly specify one
        restored_model = saved["model"]
        if restored_model != engine.active_model:
            try:
                num_ctx = await engine._resolve_num_ctx(restored_model)
                engine._backend.switch_model(restored_model, num_ctx=num_ctx)
                engine._active_model = restored_model
                print_info(f"Restored model: {restored_model} (ctx: {num_ctx:,})")
            except Exception as exc:
                logger.warning("Could not restore model %s: %s", restored_model, exc)

    print_welcome(engine.active_model, hw_summary)

    # Web dashboard (background)
    if args.web:
        result = await engine.start_web()
        print_info(result)

    # Dynamic right prompt — model + mode (shown on same line as input)
    def _rprompt():
        model_name = engine.active_model or "no model"
        short = model_name
        if len(short) > 30:
            short = short[:27] + "…"
        mode_label = "agent" if engine.mode == "agent" else "chat"
        return HTML(f'<style fg="#666666">{short} · {mode_label}</style>')

    # Dynamic bottom toolbar — context bar
    def _toolbar():
        ctx = engine.context_status
        if not ctx:
            return HTML('<style fg="ansiwhite" bg="ansiblack"> codator </style>')

        total = ctx.get("total_tokens", 0)
        window = ctx.get("context_window", 1)
        pct = ctx.get("usage_percent", 0)
        msgs = ctx.get("message_count", 0)
        compactions = ctx.get("compaction_count", 0)
        num_ctx = ctx.get("num_ctx", window)

        # Friendly token format
        def _fmt(n: int) -> str:
            return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

        remaining = max(0, window - total)

        # Color based on usage
        if pct >= 80:
            color = "ansired"
        elif pct >= 50:
            color = "ansiyellow"
        else:
            color = "ansigreen"

        # Context bar (12 chars wide)
        filled = int(pct / (100 / 12))
        bar = "█" * filled + "░" * (12 - filled)

        parts = [
            f' <{color}>[{bar}]</{color}> <b>{pct:.0f}%</b>',
            f'{_fmt(total)} used',
            f'{_fmt(remaining)} free',
            f'ctx {_fmt(num_ctx)}',
            f'{msgs} msgs',
        ]
        if compactions > 0:
            parts.append(f'<ansiyellow>⚡{compactions} compacted</ansiyellow>')

        return HTML(" │ ".join(parts))

    # Prompt style
    from prompt_toolkit.styles import Style as PtStyle
    prompt_style = PtStyle.from_dict({
        "bottom-toolbar": "bg:#1a1a2e #e0e0e0",
        "rprompt": "fg:#888888",
    })

    # Interactive prompt session
    session: PromptSession = PromptSession(
        history=InMemoryHistory(),
        completer=command_completer,
        bottom_toolbar=_toolbar,
        rprompt=_rprompt,
        style=prompt_style,
    )

    try:
        while True:
            try:
                if engine.mode == "agent":
                    prompt_prefix = HTML('<b><style fg="ansimagenta">agent</style></b><style fg="ansiwhite"> › </style>')
                else:
                    prompt_prefix = HTML('<style fg="ansiwhite">> </style>')
                # prompt_toolkit is sync; run in executor
                user_input = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: session.prompt(prompt_prefix),
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
                _save_session(engine.mode, engine.active_model)
                if should_exit:
                    break
                continue

            # Agent mode — run through Plan-Act-Verify
            if engine.mode == "agent":
                # Skip no-ops that aren't real tasks
                if user_input.lower() in ("none", "exit", "quit", "back", "cancel", "no"):
                    continue
                from codator.cli.commands import run_agent_task
                await run_agent_task(user_input, engine)
                continue

            # Chat with streaming
            ctx_before = engine.context_status.get("total_tokens", 0)
            try:
                console.print()  # blank line before response
                full_response: list[str] = []
                renderer = StreamingMarkdownRenderer()

                async for token in engine.chat_stream(user_input):
                    renderer.feed(token)
                    full_response.append(token)

                renderer.flush()
                console.print()  # newline after streaming

                # Check if compaction happened — show inline notice
                ctx_after = engine.context_status
                compaction_count = ctx_after.get("compaction_count", 0)
                if compaction_count > 0 and ctx_after.get("total_tokens", 0) < ctx_before:
                    print_compaction_inline(
                        ctx_before,
                        ctx_after.get("total_tokens", 0),
                    )

                # Show context status line after each response
                # (skip — bottom toolbar shows this permanently)

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
