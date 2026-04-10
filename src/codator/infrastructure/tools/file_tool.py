"""File Tool — read files and list directories from the project."""

from __future__ import annotations

import logging
import os

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)

# Max file size to read (256 KB)
MAX_READ_SIZE = 256 * 1024


class ReadFileTool(Tool):
    """Read file contents from the project directory."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.abspath(project_root)

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file in the project directory."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        if not path:
            return ToolResult(success=False, error="No path provided.")

        full = os.path.normpath(os.path.join(self._root, path))
        if not full.startswith(self._root):
            return ToolResult(success=False, error="Access denied: path outside project.")

        if not os.path.isfile(full):
            return ToolResult(success=False, error=f"File not found: {path}")

        size = os.path.getsize(full)
        if size > MAX_READ_SIZE:
            return ToolResult(
                success=False,
                error=f"File too large ({size:,} bytes, max {MAX_READ_SIZE:,}).",
            )

        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                content = f.read()
            return ToolResult(success=True, output=content)
        except Exception as exc:
            return ToolResult(success=False, error=f"Read error: {exc}")


class ListDirectoryTool(Tool):
    """List files and directories in the project."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.abspath(project_root)

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and subdirectories in a project directory."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to the project root. Use '.' for project root.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", ".")
        full = os.path.normpath(os.path.join(self._root, path))
        if not full.startswith(self._root):
            return ToolResult(success=False, error="Access denied: path outside project.")

        if not os.path.isdir(full):
            return ToolResult(success=False, error=f"Directory not found: {path}")

        try:
            entries = sorted(os.listdir(full))
            lines = []
            for entry in entries:
                fp = os.path.join(full, entry)
                if os.path.isdir(fp):
                    lines.append(f"  {entry}/")
                else:
                    size = os.path.getsize(fp)
                    lines.append(f"  {entry}  ({size:,} bytes)")
            header = f"Directory: {path}/ ({len(entries)} entries)"
            return ToolResult(success=True, output=header + "\n" + "\n".join(lines))
        except Exception as exc:
            return ToolResult(success=False, error=f"List error: {exc}")
