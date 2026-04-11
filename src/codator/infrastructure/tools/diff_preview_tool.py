"""Diff Preview Tool — show changes before applying.

Creates and displays unified diffs, helps verify edits will apply correctly,
and can apply patches. Reduces failed edit_file operations.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class DiffPreviewTool(Tool):
    """Preview and apply diffs — see changes before committing them."""

    def __init__(self, cwd: str | Path = ""):
        self._cwd = Path(cwd) if cwd else Path.cwd()

    @property
    def name(self) -> str:
        return "diff_preview"

    @property
    def description(self) -> str:
        return (
            "Preview file changes as unified diff before applying. "
            "Actions: preview (show what edit would change), compare (diff two files), "
            "patch (apply a unified diff), three_way (merge conflict resolution)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["preview", "compare", "patch", "three_way"],
                },
                "path": {
                    "type": "string",
                    "description": "File path to operate on.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find/replace (for 'preview').",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text (for 'preview').",
                },
                "path_b": {
                    "type": "string",
                    "description": "Second file path (for 'compare').",
                },
                "patch_content": {
                    "type": "string",
                    "description": "Unified diff content to apply (for 'patch').",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around changes (default: 3).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "preview")
        if action == "preview":
            return await self._preview(**kwargs)
        elif action == "compare":
            return await self._compare(**kwargs)
        elif action == "patch":
            return await self._patch(**kwargs)
        elif action == "three_way":
            return await self._three_way(**kwargs)
        return ToolResult(success=False, error=f"Unknown action: {action}")

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self._cwd / p
        return p

    async def _preview(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        old_text = kwargs.get("old_text", "")
        new_text = kwargs.get("new_text", "")
        ctx = kwargs.get("context_lines", 3)

        if not path:
            return ToolResult(success=False, error="'path' required.")
        if not old_text:
            return ToolResult(success=False, error="'old_text' required for preview.")

        fpath = self._resolve_path(path)
        if not fpath.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        content = fpath.read_text(errors="replace")
        occurrences = content.count(old_text)

        if occurrences == 0:
            # Try fuzzy match
            lines = content.split("\n")
            old_lines = old_text.split("\n")
            best_ratio = 0.0
            best_pos = -1
            for i in range(len(lines) - len(old_lines) + 1):
                chunk = "\n".join(lines[i:i + len(old_lines)])
                ratio = difflib.SequenceMatcher(None, old_text, chunk).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pos = i

            if best_ratio > 0.6:
                actual_text = "\n".join(lines[best_pos:best_pos + len(old_lines)])
                return ToolResult(
                    success=False,
                    output=(
                        f"Exact match not found. Best fuzzy match ({best_ratio:.0%} similar) "
                        f"at line {best_pos + 1}:\n"
                        f"---\n{actual_text[:500]}\n---\n"
                        f"Did you mean this text?"
                    ),
                    artifacts={"fuzzy_match": actual_text, "similarity": best_ratio, "line": best_pos + 1},
                )
            return ToolResult(success=False, error="old_text not found in file (no fuzzy match).")

        if occurrences > 1:
            return ToolResult(
                success=False,
                error=f"old_text found {occurrences} times — ambiguous. Add more context.",
            )

        # Generate diff
        new_content = content.replace(old_text, new_text, 1)
        old_lines = content.split("\n")
        new_lines = new_content.split("\n")

        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{path}", tofile=f"b/{path}",
            lineterm="", n=ctx,
        )
        diff_text = "\n".join(diff)

        changes = sum(1 for l in diff_text.split("\n") if l.startswith("+") and not l.startswith("+++"))
        removals = sum(1 for l in diff_text.split("\n") if l.startswith("-") and not l.startswith("---"))

        return ToolResult(
            success=True,
            output=(
                f"Preview of changes to {path}:\n"
                f"  +{changes} lines added, -{removals} lines removed\n\n"
                f"{diff_text}"
            ),
            artifacts={"additions": changes, "removals": removals, "will_apply": True},
        )

    async def _compare(self, **kwargs) -> ToolResult:
        path_a = kwargs.get("path", "")
        path_b = kwargs.get("path_b", "")
        ctx = kwargs.get("context_lines", 3)

        if not path_a or not path_b:
            return ToolResult(success=False, error="Both 'path' and 'path_b' required.")

        fa = self._resolve_path(path_a)
        fb = self._resolve_path(path_b)

        if not fa.exists():
            return ToolResult(success=False, error=f"File not found: {path_a}")
        if not fb.exists():
            return ToolResult(success=False, error=f"File not found: {path_b}")

        a_lines = fa.read_text(errors="replace").split("\n")
        b_lines = fb.read_text(errors="replace").split("\n")

        diff = difflib.unified_diff(
            a_lines, b_lines,
            fromfile=path_a, tofile=path_b,
            lineterm="", n=ctx,
        )
        diff_text = "\n".join(diff)

        if not diff_text:
            return ToolResult(success=True, output="Files are identical.")

        if len(diff_text) > 8000:
            diff_text = diff_text[:8000] + "\n... [truncated]"

        return ToolResult(success=True, output=diff_text)

    async def _patch(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        patch_content = kwargs.get("patch_content", "")

        if not path or not patch_content:
            return ToolResult(success=False, error="'path' and 'patch_content' required.")

        fpath = self._resolve_path(path)
        if not fpath.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        # Apply via system patch command
        proc = await asyncio.create_subprocess_exec(
            "patch", "--no-backup-if-mismatch", "-p1", str(fpath),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(patch_content.encode()), timeout=10
        )

        if proc.returncode != 0:
            return ToolResult(
                success=False,
                error=f"Patch failed: {stderr.decode(errors='replace')}",
            )

        return ToolResult(
            success=True,
            output=f"Patch applied to {path}: {stdout.decode(errors='replace')}",
        )

    async def _three_way(self, **kwargs) -> ToolResult:
        """Show merge conflict sections from a file."""
        path = kwargs.get("path", "")
        if not path:
            return ToolResult(success=False, error="'path' required.")

        fpath = self._resolve_path(path)
        if not fpath.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        content = fpath.read_text(errors="replace")
        conflicts = []
        in_conflict = False
        current: dict[str, list[str]] = {"ours": [], "theirs": []}
        side = ""

        for line in content.split("\n"):
            if line.startswith("<<<<<<<"):
                in_conflict = True
                side = "ours"
                current = {"ours": [], "theirs": []}
            elif line.startswith("=======") and in_conflict:
                side = "theirs"
            elif line.startswith(">>>>>>>") and in_conflict:
                conflicts.append(current.copy())
                in_conflict = False
            elif in_conflict:
                current[side].append(line)

        if not conflicts:
            return ToolResult(success=True, output="No merge conflicts found in file.")

        parts = [f"Found {len(conflicts)} merge conflict(s) in {path}:\n"]
        for i, c in enumerate(conflicts, 1):
            parts.append(f"--- Conflict {i} ---")
            parts.append("OURS:")
            parts.extend(f"  {l}" for l in c["ours"][:10])
            parts.append("THEIRS:")
            parts.extend(f"  {l}" for l in c["theirs"][:10])
            parts.append("")

        return ToolResult(success=True, output="\n".join(parts))
