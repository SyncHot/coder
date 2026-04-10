"""Rich-based rendering helpers for the CLI."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

console = Console()


def print_welcome(model: str, hw_summary: str):
    console.print(Panel(
        f"[bold cyan]codator[/bold cyan] — Dev-Assistant-OS\n"
        f"Model: [green]{model or 'none'}[/green]\n"
        f"{hw_summary}\n\n"
        f"Type [bold]/help[/bold] for commands, [bold]/quit[/bold] to exit.",
        title="🤖 codator",
        border_style="cyan",
    ))


def print_assistant(text: str):
    """Render assistant response as Markdown with syntax highlighting."""
    console.print()
    console.print(Markdown(text))
    console.print()


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
| `/model <path>` | Load a local GGUF model |
| `/api <claude\\|openai\\|ollama>` | Switch to cloud/local API |
| `/context` | Show context usage |
| `/clear` | Clear conversation |
| `/hardware` | Show GPU/hardware info |
| `/index` | Re-index project |
| `/git` | Show git work context |
| `/web` | Start web dashboard |
| `/ssh <action>` | SSH tool (connect/exec/upload/download/close) |
| `/browser <action>` | Browser tool (launch/navigate/click/screenshot/close) |
| `/terminal <cmd>` | Run a local shell command (sandboxed) |
| `/ollama [list\\|use <model>]` | List/switch Ollama models |
| `/agent <task>` | Run Plan-Act-Verify agent cycle |
| `/gpu` | Show GPU/VRAM & Ollama stats |
| `/memory` | Show contextual index stats |
| `/mcp <action>` | MCP server management (connect/list/disconnect) |
"""
    console.print(Markdown(help_text))


def print_tool_result(result):
    """Render a ToolResult in the terminal."""
    from codator.domain.models import ToolResult

    if not isinstance(result, ToolResult):
        console.print(f"[dim]{result}[/dim]")
        return

    if result.success:
        style = "green"
        icon = "✓"
    else:
        style = "red"
        icon = "✗"

    console.print(Panel(
        f"[{style}]{icon}[/{style}] "
        + (f"[bold]{result.output}[/bold]" if result.output else "")
        + (f"\n[red]{result.error}[/red]" if result.error else "")
        + (
            f"\n[dim]exit code: {result.exit_code}[/dim]"
            if result.exit_code is not None else ""
        ),
        title="Tool Result",
        border_style=style,
    ))
