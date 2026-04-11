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

        case "/architect":
            await _handle_architect(arg, engine)

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

        case "/log":
            await _handle_log(arg)

        case "/qa":
            await _handle_qa(arg, engine)

        case _:
            print_error(f"Unknown command: {command}. Type /help for available commands.")

    return False


# ---------------------------------------------------------------------------
# Log viewer
# ---------------------------------------------------------------------------


async def _handle_log(arg: str) -> None:
    """Show log file location and tail recent entries."""
    from pathlib import Path

    log_path = Path.home() / ".codator" / "logs" / "codator.log"

    if not log_path.exists():
        print_info(f"Log file: {log_path}\n  (no log file yet — start with -v for verbose logging)")
        return

    lines_str = arg.strip() if arg.strip() else "30"
    try:
        n = int(lines_str)
    except ValueError:
        print_error(f"Invalid line count: {lines_str}")
        return

    size = log_path.stat().st_size
    size_str = f"{size / 1024:.1f} KB" if size < 1048576 else f"{size / 1048576:.1f} MB"
    print_info(f"Log file: {log_path}  ({size_str})")
    print_info(f"Last {n} lines:\n")

    all_lines = log_path.read_text(errors="replace").splitlines()
    tail = all_lines[-n:] if len(all_lines) > n else all_lines
    from rich.console import Console
    from rich.syntax import Syntax
    console = Console()
    console.print(Syntax("\n".join(tail), "log", theme="monokai", line_numbers=False))


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------


async def _handle_model_picker(engine: ChatEngine) -> None:
    """Interactive model picker — lists available Ollama models with capabilities."""
    from codator.infrastructure.ollama_backend import OllamaBackend
    from rich.table import Table

    print_info("Fetching available models from Ollama...")
    backend = OllamaBackend(engine._settings)
    models = await backend.list_models()

    if not models:
        await backend.close()
        print_error("No models found. Is Ollama running?")
        return

    # Sort by name, detect capabilities for each
    models.sort(key=lambda m: m.get("name", ""))
    caps_list = []
    for m in models:
        caps = await backend.get_model_capabilities(m.get("name", ""))
        caps_list.append(caps)
    await backend.close()

    # Role icons
    _role_icons = {
        "architect": "🧠",
        "editor": "✏️",
        "both": "🧠✏️",
        "general": "💬",
    }

    table = Table(title="Available Models", border_style="cyan")
    table.add_column("#", style="bold yellow", justify="right")
    table.add_column("Name", style="bold")
    table.add_column("Size")
    table.add_column("Capabilities", style="dim")
    table.add_column("Role", style="cyan")

    for i, (m, caps) in enumerate(zip(models, caps_list), 1):
        name = m.get("name", "?")
        size_bytes = m.get("size", 0)
        size_gb = f"{size_bytes / 1_073_741_824:.1f} GB"
        active = " ← active" if name == engine.active_model else ""
        arch_mark = " ← architect" if name == engine.architect_model else ""

        # Capability badges
        badges = []
        if caps["tools"]:
            badges.append("🔧tools")
        if caps["thinking"]:
            badges.append("💭think")
        if caps["system"]:
            badges.append("📋sys")
        if caps["fim"]:
            badges.append("📝fim")

        role = caps["suggested_role"]
        role_display = f"{_role_icons.get(role, '')} {role}"

        table.add_row(
            str(i),
            f"{name}{active}{arch_mark}",
            size_gb,
            " ".join(badges) if badges else "basic",
            role_display,
        )

    console.print(table)
    console.print(
        "[dim]Enter number to switch editor model, "
        "'a<number>' to set as architect (e.g. a2), "
        "or Enter to cancel:[/dim]"
    )

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

    # Check for architect selection: a1, a2, etc.
    if choice.lower().startswith("a") and choice[1:].isdigit():
        idx = int(choice[1:]) - 1
        if idx < 0 or idx >= len(models):
            print_error(f"Invalid choice. Pick a1-a{len(models)}.")
            return
        selected = models[idx]["name"]
        engine.architect_model = selected
        engine.set_mode("agent")
        caps = caps_list[idx]
        if not caps["thinking"]:
            console.print(
                f"  [yellow]⚠  Note: {selected} has no thinking mode. "
                f"Reasoning models (deepseek-r1, qwen3) work best as architect.[/yellow]"
            )
        print_info(
            f"🏗️  Set **{selected}** as architect model.\n"
            f"   Editor: **{engine.active_model}**\n"
            f"   Auto-switched to agent mode."
        )
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
    caps = caps_list[idx]

    # If user picks a thinking-only model as editor, suggest architect role
    if caps["thinking"] and not caps["tools"]:
        console.print(
            f"  [yellow]💡 {selected} is a reasoning model (no tool support). "
            f"Consider using it as architect instead:[/yellow]\n"
            f"  [dim]   /architect {selected}[/dim]"
        )

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

        case "caps" | "capabilities" | "info":
            target = rest or engine.active_model
            if not target:
                print_error("Usage: /ollama caps [model-name]")
                return
            print_info(f"Detecting capabilities for **{target}**...")
            backend = OllamaBackend(engine._settings)
            caps = await backend.get_model_capabilities(target)
            await backend.close()

            _icons = {"architect": "🧠", "editor": "✏️", "both": "🧠✏️", "general": "💬"}
            lines = [
                f"  Model: **{target}**",
                f"  Family: {caps['family']} ({caps['architecture']})",
                f"  Quant: {caps['quantization']}",
                f"  🔧 Tool calling: {'✅' if caps['tools'] else '❌'}",
                f"  💭 Thinking/reasoning: {'✅' if caps['thinking'] else '❌'}",
                f"  📋 System prompts: {'✅' if caps['system'] else '❌'}",
                f"  📝 Fill-in-Middle: {'✅' if caps['fim'] else '❌'}",
                f"  Suggested role: {_icons.get(caps['suggested_role'], '')} **{caps['suggested_role']}**",
            ]
            print_info("\n".join(lines))

        case _:
            # Treat as model name shortcut: /ollama qwen2.5-coder:14b
            result = await engine.switch_ollama_model(action)
            print_info(result)


async def _handle_architect(arg: str, engine: ChatEngine) -> None:
    """Handle /architect command — set or show the architect (reasoning) model.

    Usage:
      /architect              — show current architect model
      /architect <model>      — set architect model (e.g. deepseek-r1:32b)
      /architect off          — disable architect mode (use single model)
    """
    arg = arg.strip()

    if not arg:
        current = engine.architect_model
        if current:
            print_info(
                f"🏗️  Architect mode: **{current}** (plans/analyzes)\n"
                f"   Editor model: **{engine.active_model}** (edits code)\n"
                f"   Use `/architect off` to disable."
            )
        else:
            print_info(
                "🏗️  Architect mode is **off** (single model for everything).\n"
                "   Use `/architect <model>` to enable, e.g.:\n"
                "   `/architect deepseek-r1:32b-qwen-distill-q4_K_M`"
            )
        return

    if arg.lower() in ("off", "none", "disable", "clear"):
        engine.architect_model = None
        print_info("🏗️  Architect mode **disabled**. Using single model for all tasks.")
        return

    # Set the architect model — verify it exists in Ollama
    try:
        import httpx
        ollama_url = engine._settings.ollama.base_url
        async with httpx.AsyncClient(base_url=ollama_url, timeout=10) as client:
            resp = await client.get("/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

        # Allow partial matching
        matched = None
        for m in models:
            if m == arg or m.startswith(arg):
                matched = m
                break

        if not matched:
            print_error(
                f"Model '{arg}' not found in Ollama. Available models:\n"
                + "\n".join(f"  • {m}" for m in models)
            )
            return

        engine.architect_model = matched
        engine.set_mode("agent")
        print_info(
            f"🏗️  Architect mode **enabled**:\n"
            f"   Architect (reasoning): **{matched}**\n"
            f"   Editor (code changes): **{engine.active_model}**\n"
            f"   Auto-switched to agent mode."
        )

    except Exception as exc:
        # Even if Ollama check fails, set the model anyway
        engine.architect_model = arg
        engine.set_mode("agent")
        print_info(
            f"🏗️  Architect model set to **{arg}** (could not verify: {exc})"
        )


async def run_agent_task(task: str, engine: ChatEngine) -> None:
    """Run the interactive agent cycle: Analyze → Propose → Pick → Implement."""
    import asyncio

    from codator.core.agent_loop import PlanActVerifyAgent, Proposal

    async def _confirm_command(command: str, reason: str) -> bool:
        """Interactive confirmation for dangerous agent commands."""
        console.print(f"\n  [yellow]⚠  Agent wants to run:[/yellow] {command}")
        console.print(f"  [dim]{reason}[/dim]")
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
        architect_model=engine.architect_model,
    )

    # --- Helpers ---

    def on_step(description: str, status: str) -> None:
        """Claude-style compact step indicator."""
        icons = {
            "started": "[dim]●[/dim]",
            "running": "[yellow]●[/yellow]",
            "done": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
        }
        icon = icons.get(status, "[dim]·[/dim]")
        msg = description
        if msg.startswith('{') or msg.startswith('{"'):
            msg = "Applying code change…"
        if len(msg) > 120:
            msg = msg[:117] + "…"
        console.print(f"  {icon} {msg}")

    # Track whether we're showing thinking vs summary
    _thinking_phase = {"active": False, "label": "", "has_output": False}

    def on_token(token: str) -> None:
        """Print model tokens in real-time — dimmed for thinking.

        Only called for natural-language streaming (force_json=False).
        JSON plan/proposal calls no longer stream, so no JSON detection needed.
        """
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

    print_info(f"● Agent: {task}")

    try:
        # ── Analysis ──
        console.print()
        console.print("  [dim]Analyzing…[/dim]")

        _start_thinking("analysis")

        proposals, analysis_actions = await agent.analyze_and_propose(
            task,
            project_context=project_context,
            on_step=on_step,
            on_token=on_token,
        )

        _stop_thinking()

        if not proposals:
            console.print("  [yellow]No proposals generated. Try rephrasing.[/yellow]")
            return

        # ── Proposals ──
        console.print()

        _priority_colors = {"high": "red", "medium": "yellow", "low": "green"}

        for p in proposals:
            color = _priority_colors.get(p.priority, "white")
            idx = p.index + 1
            console.print(f"  [{color}]{idx}.[/{color}] [bold]{p.title}[/bold]")
            console.print(f"     [dim]{p.description}[/dim]")
            console.print(f"     [dim]→ {p.file}  [{p.priority}][/dim]")

        # ── User picks ──
        console.print()
        console.print(
            "  [dim]Enter numbers (e.g.[/dim] [cyan]1,3[/cyan][dim]),[/dim] "
            "[cyan]all[/cyan][dim], [/dim][cyan]none[/cyan][dim] to skip, "
            "or type a question.[/dim]"
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
                # Freeform text — answer using gathered analysis
                console.print()
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
                    "Helpful code assistant. Answer questions about the "
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
            f"  [dim]Implementing {len(selected)} proposal(s)…[/dim]"
        )

        # ── Implement ──
        console.print()

        _start_thinking("implementing")

        result = await agent.implement_proposals(
            selected,
            task=task,
            project_context=project_context,
            on_step=on_step,
            on_token=on_token,
        )

        _stop_thinking()

        # ── Result ──
        console.print()
        if result.final_success:
            console.print("  [green]✓ Done[/green]", end="")
        else:
            console.print("  [red]✗ Failed[/red]", end="")
        console.print(
            f" [dim]— {len(result.actions)} steps, "
            f"{result.heal_iterations} heal iterations[/dim]"
        )
        if result.verification.errors:
            for err in result.verification.errors[:5]:
                console.print(f"    [red]{err}[/red]")
        console.print()

    except Exception as exc:
        _stop_thinking()
        print_error(f"Agent failed: {exc}")
    finally:
        await agent.close()


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


# ---------------------------------------------------------------------------
# QA Runner
# ---------------------------------------------------------------------------


async def _handle_qa(arg: str, engine: ChatEngine) -> None:
    """Handle /qa — comprehensive QA testing for Ethos OS NAS.

    Usage:
        /qa              — run full QA suite on configured NAS
        /qa status       — show QA config status
        /qa <url>        — run QA against specific URL
        /qa report       — show last QA report
    """
    from codator.config import get_settings
    from codator.core.qa_runner import QARunner
    from codator.infrastructure.tools.ethos_ticket_tool import EthosClient

    cfg = get_settings()
    subcommand = arg.strip().lower()

    if subcommand == "status":
        url = cfg.qa.ethos_url
        user = cfg.qa.ethos_username
        has_pw = bool(cfg.qa.ethos_password)
        project = cfg.qa.project_name
        print_info(
            f"🔧 QA Config:\n"
            f"  URL: {url or '(not set — use ETHOS_URL env var)'}\n"
            f"  User: {user or '(not set — use ETHOS_USERNAME env var)'}\n"
            f"  Password: {'✓ set' if has_pw else '✗ not set (use ETHOS_PASSWORD env var)'}\n"
            f"  Project: {project}\n"
            f"  Auto-create tickets: {cfg.qa.auto_create_tickets}\n"
            f"  Skip suites: {cfg.qa.skip_suites or '(none)'}"
        )
        return

    if subcommand == "report":
        last = getattr(engine, '_last_qa_report', None)
        if last:
            console.print(last.summary)
        else:
            print_info("No QA report available. Run /qa first.")
        return

    # Determine target URL
    ethos_url = cfg.qa.ethos_url
    ethos_user = cfg.qa.ethos_username
    ethos_pw = cfg.qa.ethos_password

    if subcommand and subcommand not in ("status", "report"):
        ethos_url = arg.strip()

    if not ethos_url:
        print_error(
            "No Ethos NAS URL configured.\n"
            "  Set ETHOS_URL env var or add ethos_url to [qa] in config.\n"
            "  Or: /qa https://nas.myserver.pl"
        )
        return

    if not ethos_user or not ethos_pw:
        print_error(
            "Ethos credentials not set.\n"
            "  Set ETHOS_USERNAME and ETHOS_PASSWORD env vars."
        )
        return

    # Run QA
    print_info(f"🧪 Starting comprehensive QA for {ethos_url}...")

    client = EthosClient(
        base_url=ethos_url,
        username=ethos_user,
        password=ethos_pw,
        verify_ssl=cfg.qa.verify_ssl,
    )

    # Login first
    try:
        await client.login()
        print_info("✓ Authenticated successfully")
    except Exception as e:
        print_error(f"Login failed: {e}")
        return

    findings_count = [0]

    async def on_finding(finding):
        findings_count[0] += 1
        icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
            finding.severity, "⚪"
        )
        console.print(
            f"  {icon} [{finding.severity.upper()}] [{finding.category}] "
            f"{finding.app}: {finding.title}"
        )

    # Get browser & vision tools from engine for UI/workflow testing
    browser_tool = engine._tools.get("browser")
    vision_tool = engine._tools.get("vision")

    runner = QARunner(
        client=client,
        project_name=cfg.qa.project_name,
        on_finding=on_finding,
        browser_tool=browser_tool,
        vision_tool=vision_tool,
    )

    skip = set(cfg.qa.skip_suites)
    report = await runner.run_all(skip=skip)
    engine._last_qa_report = report  # type: ignore[attr-defined]

    # Print summary
    console.print(f"\n{report.summary}")

    # Create tickets
    if cfg.qa.auto_create_tickets and report.findings:
        print_info(f"\n📝 Creating {len(report.findings)} ticket(s) in '{cfg.qa.project_name}'...")
        created = await runner.create_tickets(cfg.qa.project_name)
        print_info(f"✓ Created {len(created)} ticket(s)")
    elif report.findings:
        print_info(
            f"\n{len(report.findings)} finding(s). "
            f"Set auto_create_tickets=true to auto-file tickets."
        )
    else:
        print_info("\n✅ No issues found!")

