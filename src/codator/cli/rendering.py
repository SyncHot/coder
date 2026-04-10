"""Rich-based rendering helpers for the CLI — Claude-style minimal UI."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_welcome(model: str, hw_summary: str):
    """Minimal Claude-style startup — no heavy box."""
    console.print()
    console.print(f"  [bold cyan]codator[/bold cyan] [dim]— Dev-Assistant-OS[/dim]")
    console.print(f"  [dim]{hw_summary}[/dim]")
    console.print(f"  [dim]Model:[/dim] [green]{model or 'none'}[/green]")
    console.print(f"  [dim]Type[/dim] /help [dim]for commands,[/dim] /quit [dim]to exit.[/dim]")
    console.print()


def print_assistant(text: str):
    """Render assistant response as Markdown with syntax highlighting."""
    console.print()
    console.print(Markdown(text))
    console.print()


class StreamingMarkdownRenderer:
    """Buffers streaming tokens and renders code blocks with syntax highlighting."""

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._in_code_block = False
        self._code_lang = ""
        self._code_lines: list[str] = []

    def feed(self, token: str) -> None:
        """Feed a token. Renders immediately or buffers code blocks."""
        self._buffer.append(token)
        text = "".join(self._buffer)

        if self._in_code_block:
            # Check for closing fence
            if "\n```" in text or text.strip() == "```":
                # Find where the closing ``` is
                lines = text.split("\n")
                code_lines: list[str] = []
                closed = False
                for line in lines:
                    if line.strip() == "```":
                        closed = True
                        break
                    code_lines.append(line)
                if closed:
                    code = "\n".join(code_lines)
                    from rich.syntax import Syntax
                    lang = self._code_lang or "text"
                    console.print()
                    console.print(Syntax(
                        code, lang, theme="monokai",
                        line_numbers=True, word_wrap=True,
                    ))
                    self._in_code_block = False
                    self._code_lang = ""
                    self._buffer.clear()
                    # Print anything after the closing fence
                    idx = text.find("\n```")
                    rest = text[idx + 4:]  # skip \n```
                    rest_nl = rest.find("\n")
                    if rest_nl >= 0:
                        rest = rest[rest_nl + 1:]
                        if rest:
                            console.print(rest, end="", highlight=False)
            return

        # Check if we're entering a code block
        if "```" in text:
            # Split on the fence
            before, _, after = text.partition("```")
            if before:
                console.print(before, end="", highlight=False)
            # Extract language hint
            nl = after.find("\n")
            if nl >= 0:
                self._code_lang = after[:nl].strip()
                self._buffer.clear()
                self._buffer.append(after[nl + 1:])
            else:
                self._code_lang = after.strip()
                self._buffer.clear()
            self._in_code_block = True
            return

        # Normal text — flush complete lines immediately
        if "\n" in text:
            last_nl = text.rfind("\n")
            console.print(text[:last_nl + 1], end="", highlight=False)
            self._buffer.clear()
            remainder = text[last_nl + 1:]
            if remainder:
                self._buffer.append(remainder)
        elif len(text) > 200:
            # Flush long buffered text
            console.print(text, end="", highlight=False)
            self._buffer.clear()

    def flush(self) -> None:
        """Flush any remaining buffered content."""
        if self._buffer:
            text = "".join(self._buffer)
            if self._in_code_block:
                from rich.syntax import Syntax
                lang = self._code_lang or "text"
                console.print()
                console.print(Syntax(
                    text, lang, theme="monokai",
                    line_numbers=True, word_wrap=True,
                ))
            else:
                console.print(text, end="", highlight=False)
            self._buffer.clear()
            self._in_code_block = False


def print_streaming_token(token: str):
    """Print a single token without newline (for streaming)."""
    console.print(token, end="", highlight=False)


def print_context_status(status: dict):
    table = Table(title="Context Status", show_header=False, border_style="dim")
    table.add_column("Key", style="bold")
    table.add_column("Value")
    table.add_row("Tokens", f"{status['total_tokens']:,} / {status['context_window']:,}")
    table.add_row("Usage", f"{status['usage_percent']}%")
    table.add_row("Messages", str(status['message_count']))
    table.add_row("Compactions", str(status['compaction_count']))
    table.add_row("Threshold", f"{status['threshold_percent']}%")
    console.print(table)


def print_compaction_notice(before: int, after: int):
    console.print(Panel(
        f"[yellow]⚡ Context compacted[/yellow]\n"
        f"  Before: {before:,} tokens → After: {after:,} tokens\n"
        f"  A Summary Snapshot was generated and old messages removed.\n"
        f"  Use [bold]/context[/bold] to inspect.",
        border_style="yellow",
    ))


def print_compaction_inline(before: int, after: int):
    """Slim inline compaction notice (Claude-style)."""
    console.print(
        f"[yellow]⚡ Context compacted: {before:,} → {after:,} tokens "
        f"| Summary preserved | /context for details[/yellow]"
    )


def print_context_line(status: dict, model: str | None = None):
    """Dim after-turn status line showing context usage."""
    if not status:
        return

    total = status.get("total_tokens", 0)
    window = status.get("context_window", 1)
    pct = status.get("usage_percent", 0)
    msgs = status.get("message_count", 0)
    model_name = model or "unknown"

    # Format tokens in K for readability
    def _fmt(n: int) -> str:
        return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

    line = f"── {_fmt(total)} / {_fmt(window)} tokens ({pct:.0f}%) ─ {msgs} msgs ─ {model_name} ──"

    if pct >= 80:
        console.print(f"[bold yellow]⚠ Context {pct:.0f}% full — will compact soon[/bold yellow]")
    elif pct >= 70:
        console.print(f"[yellow]{line}[/yellow]")
    else:
        console.print(f"[dim]{line}[/dim]")


def print_hardware_info(hw, recommendations):
    table = Table(title="Hardware Detection", border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Value")
    table.add_row("GPU", hw.gpu_name)
    table.add_row("VRAM", f"{hw.vram_total_mb:,} MB ({hw.vram_free_mb:,} MB free)")
    table.add_row("RAM", f"{hw.ram_total_mb:,} MB ({hw.ram_free_mb:,} MB free)")
    table.add_row("CPU Cores", str(hw.cpu_cores))
    table.add_row("ROCm", hw.rocm_version or "not detected")
    table.add_row("Recommended Mode", hw.recommended_mode.value)
    console.print(table)

    if recommendations:
        rec_table = Table(title="Model Recommendations", border_style="cyan")
        rec_table.add_column("Model")
        rec_table.add_column("Params")
        rec_table.add_column("Quant")
        rec_table.add_column("Mode")
        rec_table.add_column("Est. VRAM")
        rec_table.add_column("Notes")
        for r in recommendations:
            rec_table.add_row(
                r.name, r.params, r.quant, r.mode.value,
                f"{r.estimated_vram_mb:,} MB", r.description,
            )
        console.print(rec_table)


def print_error(msg: str):
    console.print(f"[bold red]Error:[/bold red] {msg}")


def print_info(msg: str):
    console.print(f"[dim]{msg}[/dim]")


def print_help():
    help_text = """
**Commands:**
| Command | Description |
|---------|-------------|
| `/help` | Show this help |
| `/quit` or `/exit` | Exit codator |
| `/model` | Pick from available Ollama models (interactive) |
| `/model <path>` | Load a local GGUF model |
| `/api <claude\\|openai\\|ollama>` | Switch to cloud/local API |
| `/context` | Show context usage |
| `/clear` | Clear conversation |
| `/hardware` | Show GPU/hardware info |
| `/index` | Re-index project |
| `/git` | Show git work context |
| `/web` | Start web dashboard |
| `/restart` | Restart web dashboard |
| `/ssh <action>` | SSH tool (connect/exec/upload/download/close) |
| `/browser <action>` | Browser tool (launch/navigate/click/screenshot/close) |
| `/terminal <cmd>` | Run a local shell command (sandboxed) |
| `/ollama [list\\|use <model>]` | List/switch Ollama models |
| `/agent <task>` | Run Plan-Act-Verify agent cycle |
| `/agent` | Switch to persistent agent mode |
| `/chat` | Switch back to chat mode |
| `/gpu` | Show GPU/VRAM & Ollama stats |
| `/memory` | Show contextual index stats |
| `/mcp <action>` | MCP server management (connect/list/disconnect) |
| `/save [name]` | Save current conversation |
| `/load <id>` | Load a saved conversation |
| `/history` | List saved conversations |
| `/undo [file]` | Restore file(s) from .bak backups |
| `/fetch <url>` | Fetch a web page as text |
| `/search <query>` | Search the web via DuckDuckGo |
"""
    console.print(Markdown(help_text))


def print_tool_result(result):
    """Compact Claude-style tool result — one or two lines, no box."""
    from codator.domain.models import ToolResult

    if not isinstance(result, ToolResult):
        console.print(f"  [dim]{result}[/dim]")
        return

    if result.success:
        icon = "[green]✓[/green]"
    else:
        icon = "[red]✗[/red]"

    # Main line
    output = result.output or ""
    if len(output) > 200:
        output = output[:197] + "…"
    console.print(f"  {icon} {output}" if output else f"  {icon} [dim](no output)[/dim]")

    if result.error:
        for line in result.error.strip().split("\n")[:5]:
            console.print(f"    [red]{line}[/red]")
