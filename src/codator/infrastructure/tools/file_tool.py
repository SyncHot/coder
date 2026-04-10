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
        self._root = os.path.realpath(project_root)

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

        full = os.path.realpath(os.path.join(self._root, path))
        if not full.startswith(self._root + os.sep) and full != self._root:
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


class WriteFileTool(Tool):
    """Write or update file contents in the project directory."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.realpath(project_root)

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Creates parent directories as needed."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        if not path:
            return ToolResult(success=False, error="No path provided.")

        full = os.path.realpath(os.path.join(self._root, path))
        if not full.startswith(self._root + os.sep) and full != self._root:
            return ToolResult(success=False, error="Access denied: path outside project.")

        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            # Backup existing file
            if os.path.isfile(full):
                backup = full + ".bak"
                with open(full, encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
                with open(backup, "w", encoding="utf-8") as f:
                    f.write(old_content)
            with open(full, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                success=True,
                output=f"Written {len(content)} bytes to {path}",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Write error: {exc}")


class EditFileTool(Tool):
    """Apply a search-and-replace edit to a file in the project."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.realpath(project_root)

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing an exact string match with new content. "
            "Use read_file first to see the current content."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
                "old_text": {
                    "type": "string",
                    "description": "The exact text to find and replace.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        old_text = kwargs.get("old_text", "")
        new_text = kwargs.get("new_text", "")
        if not path:
            return ToolResult(success=False, error="No path provided.")
        if not old_text:
            return ToolResult(success=False, error="No old_text provided.")

        full = os.path.realpath(os.path.join(self._root, path))
        if not full.startswith(self._root + os.sep) and full != self._root:
            return ToolResult(success=False, error="Access denied: path outside project.")

        if not os.path.isfile(full):
            return ToolResult(success=False, error=f"File not found: {path}")

        try:
            with open(full, encoding="utf-8", errors="replace") as f:
                content = f.read()

            if old_text not in content:
                return ToolResult(
                    success=False,
                    error="old_text not found in file. Use read_file to check current content.",
                )

            count = content.count(old_text)
            if count > 1:
                return ToolResult(
                    success=False,
                    error=f"old_text matches {count} locations. Provide more context to be unique.",
                )

            # Backup
            with open(full + ".bak", "w", encoding="utf-8") as f:
                f.write(content)

            new_content = content.replace(old_text, new_text, 1)
            with open(full, "w", encoding="utf-8") as f:
                f.write(new_content)

            return ToolResult(
                success=True,
                output=f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars",
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Edit error: {exc}")


class ListDirectoryTool(Tool):
    """List files and directories in the project."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.realpath(project_root)

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
                    "description": (
                        "Directory path relative to project root. '.' for root."
                    ),
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", ".")
        full = os.path.realpath(os.path.join(self._root, path))
        if not full.startswith(self._root + os.sep) and full != self._root:
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
