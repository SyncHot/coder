"""Terminal Tool — sandboxed local command execution with safety checks."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)

# Default patterns that trigger human-in-the-loop confirmation
DEFAULT_DANGEROUS_PATTERNS: list[str] = [
    r"rm\s+-rf\s+/",
    r"sudo\s+rm",
    r"mkfs\.",
    r"dd\s+if=",
    r"chmod\s+777",
    r":\(\)\s*\{",
    r">\s*/dev/sd",
    r"shutdown",
    r"reboot",
    r"init\s+0",
]

# Type for the confirmation callback:
#   async def confirm(command: str, reason: str) -> bool
ConfirmCallback = Callable[[str, str], Awaitable[bool]]


class TerminalTool(Tool):
    """Execute local shell commands with safety guardrails."""

    def __init__(
        self,
        working_dir: str = ".",
        timeout: int = 60,
        require_confirm: bool = True,
        dangerous_patterns: list[str] | None = None,
        confirm_callback: ConfirmCallback | None = None,
    ):
        self._working_dir = working_dir
        self._timeout = timeout
        self._require_confirm = require_confirm
        self._patterns = [
            re.compile(p) for p in (dangerous_patterns or DEFAULT_DANGEROUS_PATTERNS)
        ]
        self._confirm = confirm_callback

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return (
            "Execute local shell commands in a sandboxed environment. "
            "Dangerous commands require human approval."
        )

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------

    def is_dangerous(self, command: str) -> str | None:
        """Return matched pattern string if command is dangerous, else None."""
        for pattern in self._patterns:
            if pattern.search(command):
                return pattern.pattern
        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def run_command(self, command: str, timeout: int = 0,
                          working_dir: str = "") -> ToolResult:
        """Run a shell command with safety checks."""
        t = timeout or self._timeout
        cwd = working_dir or self._working_dir

        # Safety check
        danger = self.is_dangerous(command)
        if danger and self._require_confirm:
            if self._confirm is None:
                return ToolResult(
                    success=False,
                    error=f"Dangerous command blocked (matched: {danger}). "
                    f"No confirmation callback configured.",
                )
            approved = await self._confirm(
                command,
                f"Command matches dangerous pattern: {danger}",
            )
            if not approved:
                return ToolResult(
                    success=False,
                    error="Command rejected by user.",
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=t,
                )
            except TimeoutError:
                proc.kill()
                await proc.communicate()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {t}s.",
                    exit_code=-1,
                )

            out = stdout.decode("utf-8", errors="replace")
            err = stderr.decode("utf-8", errors="replace")
            code = proc.returncode or 0

            return ToolResult(
                success=(code == 0),
                output=out,
                error=err,
                exit_code=code,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Execution failed: {exc}")

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        """Run a terminal command.

        Parameters
        ----------
        command : str
            The shell command to execute.
        timeout : int, optional
            Timeout in seconds.
        working_dir : str, optional
            Working directory override.
        """
        command = kwargs.get("command", "")
        if not command:
            return ToolResult(success=False, error="No command provided.")
        return await self.run_command(
            command=command,
            timeout=kwargs.get("timeout", 0),
            working_dir=kwargs.get("working_dir", ""),
        )
