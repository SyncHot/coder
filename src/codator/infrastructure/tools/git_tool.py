"""Git Tool — structured git operations with parsed output.

Provides git status, diff, log, commit, branch operations with structured
results instead of raw terminal output parsing.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class GitTool(Tool):
    """Structured git operations with parsed output."""

    def __init__(self, cwd: str | Path = ""):
        self._cwd = str(cwd) if cwd else ""

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return (
            "Execute structured git operations. Actions: status, diff, log, "
            "commit, branch, stash, show, blame, add, checkout."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "status", "diff", "log", "commit", "branch",
                        "stash", "show", "blame", "add", "checkout",
                    ],
                },
                "path": {
                    "type": "string",
                    "description": "File path or pattern for targeted operations.",
                },
                "message": {
                    "type": "string",
                    "description": "Commit message (for 'commit' action).",
                },
                "ref": {
                    "type": "string",
                    "description": "Branch/tag/SHA reference.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of entries (for 'log' — default 10).",
                },
                "staged": {
                    "type": "boolean",
                    "description": "Show only staged changes (for 'diff').",
                },
                "all": {
                    "type": "boolean",
                    "description": "Stage all changes before commit.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "status")
        dispatch = {
            "status": self._status,
            "diff": self._diff,
            "log": self._log,
            "commit": self._commit,
            "branch": self._branch,
            "stash": self._stash,
            "show": self._show,
            "blame": self._blame,
            "add": self._add,
            "checkout": self._checkout,
        }
        handler = dispatch.get(action)
        if not handler:
            return ToolResult(success=False, error=f"Unknown git action: {action}")
        return await handler(**kwargs)

    async def _run(self, *args: str) -> tuple[int, str, str]:
        """Run git command and return (returncode, stdout, stderr)."""
        cmd = ["git", "--no-pager"] + list(args)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd or None,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def _status(self, **kwargs) -> ToolResult:
        rc, out, err = await self._run("status", "--porcelain=v2", "--branch")
        if rc != 0:
            return ToolResult(success=False, error=err or out)

        lines = out.strip().split("\n") if out.strip() else []
        branch = ""
        staged, modified, untracked = [], [], []

        for line in lines:
            if line.startswith("# branch.head"):
                branch = line.split()[-1]
            elif line.startswith("1 ") or line.startswith("2 "):
                parts = line.split()
                xy = parts[1]
                path = parts[-1]
                if xy[0] != ".":
                    staged.append(path)
                if xy[1] != ".":
                    modified.append(path)
            elif line.startswith("? "):
                untracked.append(line[2:])

        summary = f"Branch: {branch}\n"
        if staged:
            summary += f"Staged ({len(staged)}): {', '.join(staged[:10])}\n"
        if modified:
            summary += f"Modified ({len(modified)}): {', '.join(modified[:10])}\n"
        if untracked:
            summary += f"Untracked ({len(untracked)}): {', '.join(untracked[:10])}\n"
        if not staged and not modified and not untracked:
            summary += "Working tree clean.\n"

        return ToolResult(
            success=True,
            output=summary,
            artifacts={"branch": branch, "staged": staged, "modified": modified, "untracked": untracked},
        )

    async def _diff(self, **kwargs) -> ToolResult:
        args = ["diff", "--stat"]
        if kwargs.get("staged"):
            args.append("--cached")
        path = kwargs.get("path", "")
        if path:
            args.extend(["--", path])
        rc, stat_out, err = await self._run(*args)
        if rc != 0:
            return ToolResult(success=False, error=err)

        # Also get the actual diff (limited)
        diff_args = ["diff"]
        if kwargs.get("staged"):
            diff_args.append("--cached")
        if path:
            diff_args.extend(["--", path])
        _, diff_out, _ = await self._run(*diff_args)

        # Truncate large diffs
        if len(diff_out) > 8000:
            diff_out = diff_out[:8000] + "\n... [truncated, use 'path' to narrow scope]"

        return ToolResult(
            success=True,
            output=f"Stats:\n{stat_out}\nDiff:\n{diff_out}" if diff_out else "No differences.",
        )

    async def _log(self, **kwargs) -> ToolResult:
        count = kwargs.get("count", 10)
        args = ["log", f"--oneline", f"-{count}", "--decorate"]
        ref = kwargs.get("ref", "")
        if ref:
            args.append(ref)
        path = kwargs.get("path", "")
        if path:
            args.extend(["--", path])
        rc, out, err = await self._run(*args)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=out or "No commits.")

    async def _commit(self, **kwargs) -> ToolResult:
        message = kwargs.get("message", "")
        if not message:
            return ToolResult(success=False, error="'message' required for commit.")

        if kwargs.get("all"):
            rc, _, err = await self._run("add", "-A")
            if rc != 0:
                return ToolResult(success=False, error=f"git add failed: {err}")

        rc, out, err = await self._run("commit", "-m", message)
        if rc != 0:
            return ToolResult(success=False, error=err or out)
        return ToolResult(success=True, output=out)

    async def _branch(self, **kwargs) -> ToolResult:
        ref = kwargs.get("ref", "")
        if ref:
            # Create or switch branch
            rc, out, err = await self._run("checkout", "-b", ref)
            if rc != 0:
                # Branch exists, just switch
                rc, out, err = await self._run("checkout", ref)
                if rc != 0:
                    return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=f"Switched to branch: {ref}")
        else:
            rc, out, err = await self._run("branch", "-a")
            if rc != 0:
                return ToolResult(success=False, error=err)
            return ToolResult(success=True, output=out)

    async def _stash(self, **kwargs) -> ToolResult:
        message = kwargs.get("message", "")
        if message == "pop":
            rc, out, err = await self._run("stash", "pop")
        elif message == "list":
            rc, out, err = await self._run("stash", "list")
        elif message:
            rc, out, err = await self._run("stash", "push", "-m", message)
        else:
            rc, out, err = await self._run("stash")
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=out or "Stash operation complete.")

    async def _show(self, **kwargs) -> ToolResult:
        ref = kwargs.get("ref", "HEAD")
        path = kwargs.get("path", "")
        args = ["show", "--stat", ref]
        if path:
            args = ["show", ref, "--", path]
        rc, out, err = await self._run(*args)
        if rc != 0:
            return ToolResult(success=False, error=err)
        if len(out) > 8000:
            out = out[:8000] + "\n... [truncated]"
        return ToolResult(success=True, output=out)

    async def _blame(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        if not path:
            return ToolResult(success=False, error="'path' required for blame.")
        rc, out, err = await self._run("blame", "--porcelain", path)
        if rc != 0:
            return ToolResult(success=False, error=err)
        if len(out) > 8000:
            out = out[:8000] + "\n... [truncated]"
        return ToolResult(success=True, output=out)

    async def _add(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", ".")
        if kwargs.get("all"):
            rc, out, err = await self._run("add", "-A")
        else:
            rc, out, err = await self._run("add", path)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=f"Added: {path}")

    async def _checkout(self, **kwargs) -> ToolResult:
        ref = kwargs.get("ref", "")
        path = kwargs.get("path", "")
        if path:
            rc, out, err = await self._run("checkout", "--", path)
        elif ref:
            rc, out, err = await self._run("checkout", ref)
        else:
            return ToolResult(success=False, error="'ref' or 'path' required.")
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=out or "Checkout complete.")
