"""File Tool — read files, list directories, search, and glob from the project."""

from __future__ import annotations

import errno
import logging
import os
import pathlib
import re
import shutil
import subprocess

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)

# Max file size to read (256 KB)
MAX_READ_SIZE = 256 * 1024


class ReadFileTool(Tool):
    """Smart file reader — handles directories and missing files gracefully."""

    def __init__(self, project_root: str = "."):
        self._root = pathlib.Path(project_root).resolve()

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a text file in the project directory. "
            "If the path is a directory, returns a listing of its contents. "
            "If the file doesn't exist, returns a helpful hint. "
            "Max 256KB. Binary files are rejected."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root (e.g. 'src/main.py'). Max 256KB, text files only.",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        if not path:
            return ToolResult(success=False, error="No path provided.")

        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(success=False, error="Access denied: path outside project.")

        try:
            # Smart handling: directory → list contents
            if target.is_dir():
                return self._list_directory(target, path)

            # Smart handling: file doesn't exist → helpful hint
            if not target.exists():
                return ToolResult(
                    success=False,
                    error=(
                        f"FILE_NOT_FOUND: Plik '{path}' jeszcze nie istnieje. "
                        "Jeśli chcesz go stworzyć, użyj akcji zapisu (write_file / create_file)."
                    ),
                )

            if not target.is_file():
                return ToolResult(success=False, error=f"Not a regular file: {path}")

            size = target.stat().st_size
            if size > MAX_READ_SIZE:
                return ToolResult(
                    success=False,
                    error=f"File too large ({size:,} bytes, max {MAX_READ_SIZE:,}).",
                )

            # Check for binary content before reading as text
            with target.open("rb") as bf:
                chunk = bf.read(8192)
                if b"\x00" in chunk:
                    return ToolResult(success=False, error=f"File appears to be binary: {path}")
            content = target.read_text(encoding="utf-8", errors="replace")
            return ToolResult(success=True, output=content)

        except OSError as exc:
            return self._translate_os_error(exc, path)
        except Exception as exc:
            return ToolResult(success=False, error=f"Read error: {exc}")

    def _list_directory(self, target: pathlib.Path, rel_path: str) -> ToolResult:
        """Return a directory listing when agent tries to read_file on a folder."""
        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for entry in entries:
                if entry.is_dir():
                    lines.append(f"  📁 {entry.name}/")
                else:
                    size = entry.stat().st_size
                    lines.append(f"  📄 {entry.name}  ({size:,} bytes)")
            header = (
                f"DIRECTORY_LISTING: '{rel_path}' jest folderem, nie plikiem. "
                f"Oto jego zawartość ({len(entries)} pozycji):"
            )
            return ToolResult(success=True, output=header + "\n" + "\n".join(lines))
        except OSError as exc:
            return self._translate_os_error(exc, rel_path)

    @staticmethod
    def _translate_os_error(exc: OSError, path: str) -> ToolResult:
        """Translate OS-level I/O errors into agent-friendly messages."""
        if exc.errno == errno.ENOENT:
            return ToolResult(
                success=False,
                error=(
                    f"FILE_NOT_FOUND: Plik '{path}' jeszcze nie istnieje. "
                    "Jeśli chcesz go stworzyć, użyj akcji zapisu (write_file / create_file)."
                ),
            )
        if exc.errno == errno.EISDIR:
            return ToolResult(
                success=False,
                error=(
                    f"IS_DIRECTORY: '{path}' jest folderem, nie plikiem. "
                    "Użyj list_directory lub sprawdź ścieżkę."
                ),
            )
        return ToolResult(success=False, error=f"I/O error: {exc}")


class WriteFileTool(Tool):
    """Atomic file writer — auto-creates missing parent directories."""

    def __init__(self, project_root: str = "."):
        self._root = pathlib.Path(project_root).resolve()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories as needed."
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

        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(success=False, error="Access denied: path outside project.")

        try:
            # Atomic: auto-create all missing parent directories
            target.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing file
            if target.is_file():
                backup = target.with_suffix(target.suffix + ".bak")
                backup.write_text(
                    target.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                )

            target.write_text(content, encoding="utf-8")
            return ToolResult(
                success=True,
                output=f"Written {len(content)} bytes to {path}",
            )
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return ToolResult(
                    success=False,
                    error=f"Cannot create file '{path}': parent path issue — {exc}",
                )
            return ToolResult(success=False, error=f"Write I/O error: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Write error: {exc}")


class EditFileTool(Tool):
    """Apply a search-and-replace edit to a file in the project."""

    def __init__(self, project_root: str = "."):
        self._root = pathlib.Path(project_root).resolve()

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing an EXACT string match with new content. "
            "The old_text must match exactly (including whitespace and indentation). "
            "If it matches multiple locations, the edit is rejected — include more context to be unique. "
            "Always read_file first to see current content."
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
                    "description": "The EXACT text to find and replace (must match character-for-character, including whitespace). If multiple matches exist, include more surrounding context.",
                },
                "new_text": {
                    "type": "string",
                    "description": "The replacement text. Can be empty string to delete old_text.",
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

        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(success=False, error="Access denied: path outside project.")

        if target.is_dir():
            return ToolResult(
                success=False,
                error=(
                    f"IS_DIRECTORY: '{path}' jest folderem, nie plikiem. "
                    "Sprawdź ścieżkę — nie można edytować folderu."
                ),
            )
        if not target.exists():
            return ToolResult(
                success=False,
                error=(
                    f"FILE_NOT_FOUND: Plik '{path}' nie istnieje. "
                    "Użyj read_file, aby sprawdzić ścieżkę, lub create_file, aby go stworzyć."
                ),
            )

        try:
            content = target.read_text(encoding="utf-8", errors="replace")

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
            backup = target.with_suffix(target.suffix + ".bak")
            backup.write_text(content, encoding="utf-8")

            new_content = content.replace(old_text, new_text, 1)
            target.write_text(new_content, encoding="utf-8")

            return ToolResult(
                success=True,
                output=f"Edited {path}: replaced {len(old_text)} chars with {len(new_text)} chars",
            )
        except OSError as exc:
            if exc.errno == errno.EISDIR:
                return ToolResult(
                    success=False,
                    error=f"IS_DIRECTORY: '{path}' jest folderem. Sprawdź ścieżkę.",
                )
            return ToolResult(success=False, error=f"Edit I/O error: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Edit error: {exc}")


class ListDirectoryTool(Tool):
    """List files and directories in the project."""

    def __init__(self, project_root: str = "."):
        self._root = pathlib.Path(project_root).resolve()

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return (
            "List files and subdirectories in a project directory. "
            "Shows file sizes. Does not recurse into subdirectories — call again for deeper paths."
        )

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
        target = (self._root / path).resolve()
        if not str(target).startswith(str(self._root)):
            return ToolResult(success=False, error="Access denied: path outside project.")

        if not target.is_dir():
            return ToolResult(success=False, error=f"Directory not found: {path}")

        try:
            entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            lines = []
            for entry in entries:
                if entry.is_dir():
                    lines.append(f"  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    lines.append(f"  {entry.name}  ({size:,} bytes)")
            header = f"Directory: {path}/ ({len(list(target.iterdir()))} entries)"
            return ToolResult(success=True, output=header + "\n" + "\n".join(lines))
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                return ToolResult(
                    success=False,
                    error=f"Directory '{path}' does not exist.",
                )
            return ToolResult(success=False, error=f"List I/O error: {exc}")
        except Exception as exc:
            return ToolResult(success=False, error=f"List error: {exc}")


# Directories to skip when searching/globbing
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv"}

# Max characters in search output before truncation
_MAX_OUTPUT_CHARS = 8000


class GrepTool(Tool):
    """Search for a pattern in file contents across the project."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.realpath(project_root)

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "Search for a pattern in file contents across the project. "
            "Returns matching lines with file paths and line numbers. "
            "Use to find code, function definitions, imports, or any text pattern."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory or file to search in, relative to project root. "
                        "Defaults to '.' (entire project)."
                    ),
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter, e.g. '*.py' or '*.ts'. Defaults to all files.",
                },
                "context_lines": {
                    "type": "integer",
                    "description": "Lines of context around each match (default 0).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matching lines to return (default 100).",
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, error="No pattern provided.")

        rel_path = kwargs.get("path", ".") or "."
        glob_filter = kwargs.get("glob", "") or ""
        context_lines = int(kwargs.get("context_lines", 0) or 0)
        max_results = int(kwargs.get("max_results", 100) or 100)

        full = os.path.realpath(os.path.join(self._root, rel_path))
        if not full.startswith(self._root + os.sep) and full != self._root:
            return ToolResult(success=False, error="Access denied: path outside project.")
        if not os.path.exists(full):
            return ToolResult(success=False, error=f"Path not found: {rel_path}")

        # Try ripgrep first, fall back to Python implementation
        rg = shutil.which("rg")
        if rg:
            return self._search_ripgrep(
                rg, pattern, full, glob_filter, context_lines, max_results,
            )
        return self._search_python(
            pattern, full, glob_filter, context_lines, max_results,
        )

    def _search_ripgrep(
        self,
        rg_path: str,
        pattern: str,
        search_path: str,
        glob_filter: str,
        context_lines: int,
        max_results: int,
    ) -> ToolResult:
        cmd = [
            rg_path,
            "--no-heading",
            "--line-number",
            "--color=never",
            f"--max-count={max_results}",
        ]
        if context_lines > 0:
            cmd.append(f"-C{context_lines}")
        if glob_filter:
            cmd.extend(["-g", glob_filter])
        for d in _SKIP_DIRS:
            cmd.extend(["-g", f"!{d}"])
        cmd.extend([pattern, search_path])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30, cwd=self._root,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(success=False, error="Search timed out after 30 seconds.")
        except Exception as exc:
            return ToolResult(success=False, error=f"Ripgrep error: {exc}")

        if result.returncode not in (0, 1):
            return ToolResult(success=False, error=result.stderr.strip() or "Ripgrep error.")

        output = self._make_relative(result.stdout)
        return self._format_output(output)

    def _search_python(
        self,
        pattern: str,
        search_path: str,
        glob_filter: str,
        context_lines: int,
        max_results: int,
    ) -> ToolResult:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return ToolResult(success=False, error=f"Invalid regex: {exc}")

        import fnmatch

        lines_found: list[str] = []

        def _walk(base: str) -> None:
            if len(lines_found) >= max_results:
                return
            if os.path.isfile(base):
                _search_file(base)
                return
            try:
                entries = sorted(os.listdir(base))
            except PermissionError:
                return
            for entry in entries:
                if entry in _SKIP_DIRS:
                    continue
                fp = os.path.join(base, entry)
                if os.path.isdir(fp):
                    _walk(fp)
                elif os.path.isfile(fp):
                    if glob_filter and not fnmatch.fnmatch(entry, glob_filter):
                        continue
                    _search_file(fp)
                if len(lines_found) >= max_results:
                    return

        def _search_file(filepath: str) -> None:
            try:
                with open(filepath, "rb") as bf:
                    chunk = bf.read(8192)
                    if b"\x00" in chunk:
                        return
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    file_lines = f.readlines()
            except (PermissionError, OSError):
                return

            relpath = os.path.relpath(filepath, self._root)
            for i, line in enumerate(file_lines, 1):
                if len(lines_found) >= max_results:
                    return
                if regex.search(line):
                    if context_lines > 0:
                        start = max(0, i - 1 - context_lines)
                        end = min(len(file_lines), i + context_lines)
                        for ci in range(start, end):
                            marker = ":" if ci == i - 1 else "-"
                            lines_found.append(
                                f"{relpath}{marker}{ci + 1}{marker} {file_lines[ci].rstrip()}"
                            )
                    else:
                        lines_found.append(f"{relpath}:{i}: {line.rstrip()}")

        _walk(search_path)
        output = "\n".join(lines_found)
        return self._format_output(output)

    def _make_relative(self, text: str) -> str:
        """Convert absolute paths in ripgrep output to relative paths."""
        root_prefix = self._root + os.sep
        return text.replace(root_prefix, "")

    def _format_output(self, output: str) -> ToolResult:
        if not output.strip():
            return ToolResult(success=True, output="No matches found.")
        if len(output) > _MAX_OUTPUT_CHARS:
            output = output[:_MAX_OUTPUT_CHARS] + "\n[... truncated]"
        return ToolResult(success=True, output=output)


class GlobTool(Tool):
    """Find files by name pattern."""

    def __init__(self, project_root: str = "."):
        self._root = os.path.realpath(project_root)

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "Find files by name pattern. Returns matching file paths relative to "
            "the project root. Use to locate files when you know part of the name."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": (
                        "Glob pattern to match files, e.g. '**/*.py', 'src/**/*.ts', "
                        "'*.json'. Supports ** for recursive matching."
                    ),
                },
                "path": {
                    "type": "string",
                    "description": (
                        "Directory to search in, relative to project root. "
                        "Defaults to '.' (entire project)."
                    ),
                },
            },
            "required": ["pattern"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        pattern = kwargs.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, error="No pattern provided.")

        rel_path = kwargs.get("path", ".") or "."
        full = os.path.realpath(os.path.join(self._root, rel_path))
        if not full.startswith(self._root + os.sep) and full != self._root:
            return ToolResult(success=False, error="Access denied: path outside project.")
        if not os.path.isdir(full):
            return ToolResult(success=False, error=f"Directory not found: {rel_path}")

        base = pathlib.Path(full)
        matches: list[str] = []
        max_files = 200

        try:
            for p in sorted(base.glob(pattern)):
                if any(part in _SKIP_DIRS for part in p.parts):
                    continue
                if p.is_file():
                    matches.append(os.path.relpath(str(p), self._root))
                if len(matches) >= max_files:
                    break
        except Exception as exc:
            return ToolResult(success=False, error=f"Glob error: {exc}")

        if not matches:
            return ToolResult(success=True, output="No files matched.")

        output = "\n".join(matches)
        suffix = f"\n[... limited to {max_files} files]" if len(matches) >= max_files else ""
        return ToolResult(success=True, output=output + suffix)
