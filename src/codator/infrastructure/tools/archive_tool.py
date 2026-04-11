"""Archive tool — zip/tar/gz/7z operations."""

import asyncio
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class ArchiveTool:
    name = "archive"
    description = (
        "Handle archive files: create, extract, and list contents of "
        "zip, tar, tar.gz, tar.bz2, and 7z archives. "
        "Useful for packaging deployments and working with compressed files."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "extract", "list", "info"],
                "description": "Archive action to perform",
            },
            "path": {
                "type": "string",
                "description": "Archive file path (for extract/list) or output path (for create)",
            },
            "source": {
                "type": "string",
                "description": "Source directory or file to archive (for create action)",
            },
            "destination": {
                "type": "string",
                "description": "Extraction destination directory",
            },
            "format": {
                "type": "string",
                "enum": ["zip", "tar.gz", "tar.bz2", "tar", "7z"],
                "description": "Archive format (auto-detected from extension if not specified)",
            },
            "include": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Only include files matching these patterns",
            },
            "exclude": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Exclude files matching these patterns",
            },
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs["path"]
        try:
            if action == "create":
                return await self._create(path, kwargs)
            elif action == "extract":
                return await self._extract(path, kwargs)
            elif action == "list":
                return self._list_contents(path)
            elif action == "info":
                return self._info(path)
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _detect_format(self, path: str) -> str:
        if path.endswith(".zip"):
            return "zip"
        elif path.endswith(".tar.gz") or path.endswith(".tgz"):
            return "tar.gz"
        elif path.endswith(".tar.bz2") or path.endswith(".tbz2"):
            return "tar.bz2"
        elif path.endswith(".tar"):
            return "tar"
        elif path.endswith(".7z"):
            return "7z"
        return "zip"

    async def _create(self, path: str, kwargs: dict) -> ToolResult:
        """Create an archive."""
        source = kwargs.get("source", "")
        if not source:
            return ToolResult(success=False, error="source is required for create action")
        if not os.path.exists(source):
            return ToolResult(success=False, error="Source not found: " + source)

        fmt = kwargs.get("format") or self._detect_format(path)
        exclude = set(kwargs.get("exclude", []))
        exclude.update({".git", "node_modules", "__pycache__", ".venv", "venv"})

        def should_include(filepath):
            for pattern in exclude:
                if pattern in filepath:
                    return False
            return True

        file_count = 0
        total_size = 0

        if fmt == "zip":
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                if os.path.isfile(source):
                    zf.write(source, os.path.basename(source))
                    file_count = 1
                else:
                    for root, dirs, files in os.walk(source):
                        dirs[:] = [d for d in dirs if d not in exclude]
                        for fname in files:
                            filepath = os.path.join(root, fname)
                            arcname = os.path.relpath(filepath, os.path.dirname(source))
                            if should_include(arcname):
                                zf.write(filepath, arcname)
                                file_count += 1
                                total_size += os.path.getsize(filepath)

        elif fmt in ("tar.gz", "tar.bz2", "tar"):
            mode_map = {"tar.gz": "w:gz", "tar.bz2": "w:bz2", "tar": "w"}
            mode = mode_map[fmt]
            with tarfile.open(path, mode) as tf:
                if os.path.isfile(source):
                    tf.add(source, arcname=os.path.basename(source))
                    file_count = 1
                else:
                    for root, dirs, files in os.walk(source):
                        dirs[:] = [d for d in dirs if d not in exclude]
                        for fname in files:
                            filepath = os.path.join(root, fname)
                            arcname = os.path.relpath(filepath, os.path.dirname(source))
                            if should_include(arcname):
                                tf.add(filepath, arcname=arcname)
                                file_count += 1
                                total_size += os.path.getsize(filepath)

        elif fmt == "7z":
            cmd = ["7z", "a", path, source]
            for pattern in exclude:
                cmd.extend(["-xr!" + pattern])
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return ToolResult(success=False, error="7z failed: " + stderr.decode())
            return ToolResult(success=True, output="Created " + path + "\n" + stdout.decode()[:500])

        archive_size = os.path.getsize(path)
        ratio = (1 - archive_size / max(total_size, 1)) * 100 if total_size else 0

        output = "Created: " + path + "\n"
        output += "  Format: " + fmt + "\n"
        output += "  Files: " + str(file_count) + "\n"
        output += "  Original size: " + self._human_size(total_size) + "\n"
        output += "  Archive size: " + self._human_size(archive_size) + "\n"
        output += "  Compression: " + f"{ratio:.1f}" + "%"

        return ToolResult(success=True, output=output, artifacts={
            "files": file_count, "size": archive_size, "compression_ratio": round(ratio, 1)
        })

    async def _extract(self, path: str, kwargs: dict) -> ToolResult:
        """Extract an archive."""
        if not os.path.isfile(path):
            return ToolResult(success=False, error="Archive not found: " + path)

        destination = kwargs.get("destination", os.path.splitext(path)[0])
        os.makedirs(destination, exist_ok=True)
        fmt = kwargs.get("format") or self._detect_format(path)

        file_count = 0

        if fmt == "zip":
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(destination)
                file_count = len(zf.namelist())

        elif fmt in ("tar.gz", "tar.bz2", "tar"):
            with tarfile.open(path, "r:*") as tf:
                tf.extractall(destination, filter="data")
                file_count = len(tf.getnames())

        elif fmt == "7z":
            proc = await asyncio.create_subprocess_exec(
                "7z", "x", path, "-o" + destination, "-y",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return ToolResult(success=False, error="7z extraction failed: " + stderr.decode())
            return ToolResult(success=True, output="Extracted to: " + destination + "\n" + stdout.decode()[:500])

        return ToolResult(
            success=True,
            output="Extracted " + str(file_count) + " files to: " + destination,
            artifacts={"files": file_count, "destination": destination},
        )

    def _list_contents(self, path: str) -> ToolResult:
        """List archive contents."""
        if not os.path.isfile(path):
            return ToolResult(success=False, error="Archive not found: " + path)

        fmt = self._detect_format(path)
        entries = []

        if fmt == "zip":
            with zipfile.ZipFile(path, "r") as zf:
                for info in zf.infolist():
                    entries.append((info.filename, info.file_size, info.compress_size))

        elif fmt in ("tar.gz", "tar.bz2", "tar"):
            with tarfile.open(path, "r:*") as tf:
                for member in tf.getmembers():
                    entries.append((member.name, member.size, member.size))

        lines = ["Contents of " + path + " (" + str(len(entries)) + " entries):\n"]
        total_size = 0
        for name, size, compressed in entries[:100]:
            total_size += size
            lines.append("  " + self._human_size(size).rjust(8) + "  " + name)

        if len(entries) > 100:
            lines.append("  ... and " + str(len(entries) - 100) + " more")
        lines.append("\nTotal: " + self._human_size(total_size))

        return ToolResult(success=True, output="\n".join(lines)[:5000], artifacts={"entries": len(entries), "total_size": total_size})

    def _info(self, path: str) -> ToolResult:
        """Get archive metadata."""
        if not os.path.isfile(path):
            return ToolResult(success=False, error="Archive not found: " + path)

        fmt = self._detect_format(path)
        file_size = os.path.getsize(path)

        info = {
            "path": path,
            "format": fmt,
            "size": self._human_size(file_size),
            "size_bytes": file_size,
        }

        if fmt == "zip":
            with zipfile.ZipFile(path, "r") as zf:
                info["entries"] = len(zf.namelist())
                info["comment"] = zf.comment.decode() if zf.comment else ""
        elif fmt in ("tar.gz", "tar.bz2", "tar"):
            with tarfile.open(path, "r:*") as tf:
                info["entries"] = len(tf.getnames())

        lines = ["Archive info:"]
        for k, v in info.items():
            lines.append("  " + k + ": " + str(v))

        return ToolResult(success=True, output="\n".join(lines), artifacts=info)

    def _human_size(self, size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"
