"""MCP (Model Context Protocol) server — exposes codator tools via JSON-RPC over stdio."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from typing import Any

from codator.domain.interfaces import Tool
from codator.infrastructure.tools.file_tool import (
    EditFileTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from codator.infrastructure.tools.terminal_tool import TerminalTool

logger = logging.getLogger(__name__)

# JSON-RPC error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

SERVER_NAME = "codator"
SERVER_VERSION = "0.2.0"


class MCPServer:
    """MCP server that exposes codator's tools over JSON-RPC stdio transport."""

    def __init__(self, project_root: str = ".") -> None:
        self._project_root = os.path.abspath(project_root)
        self._tools: dict[str, Tool] = {}
        self._register_tools()

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_tools(self) -> None:
        tools: list[Tool] = [
            ReadFileTool(self._project_root),
            ListDirectoryTool(self._project_root),
            WriteFileTool(self._project_root),
            EditFileTool(self._project_root),
            TerminalTool(working_dir=self._project_root, require_confirm=False),
        ]
        for tool in tools:
            self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # JSON-RPC helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _result_response(req_id: int | str | None, result: Any) -> dict:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error_response(
        req_id: int | str | None, code: int, message: str, data: Any = None,
    ) -> dict:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    # ------------------------------------------------------------------
    # Method dispatch
    # ------------------------------------------------------------------

    async def _handle_message(self, msg: dict) -> dict | None:
        if msg.get("jsonrpc") != "2.0":
            return self._error_response(msg.get("id"), INVALID_REQUEST, "Missing jsonrpc 2.0")

        method: str | None = msg.get("method")
        req_id = msg.get("id")
        params: dict = msg.get("params") or {}

        # Notifications (no id) are silently ignored per JSON-RPC spec.
        if req_id is None and method not in ("initialize",):
            return None

        if not method:
            return self._error_response(req_id, INVALID_REQUEST, "Missing method")

        handler = {
            "initialize": self._handle_initialize,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
            "resources/list": self._handle_resources_list,
            "resources/read": self._handle_resources_read,
            "notifications/initialized": self._handle_noop,
            "ping": self._handle_ping,
        }.get(method)

        if handler is None:
            return self._error_response(req_id, METHOD_NOT_FOUND, f"Unknown method: {method}")

        try:
            result = await handler(params)
            return self._result_response(req_id, result)
        except TypeError as exc:
            return self._error_response(req_id, INVALID_PARAMS, str(exc))
        except Exception as exc:
            logger.exception("Internal error handling %s", method)
            return self._error_response(req_id, INTERNAL_ERROR, str(exc))

    # ------------------------------------------------------------------
    # Method handlers
    # ------------------------------------------------------------------

    async def _handle_initialize(self, params: dict) -> dict:
        return {
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False},
            },
        }

    async def _handle_tools_list(self, params: dict) -> dict:
        tools_list = []
        for tool in self._tools.values():
            tools_list.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.parameters_schema,
            })
        return {"tools": tools_list}

    async def _handle_tools_call(self, params: dict) -> dict:
        tool_name = params.get("name")
        if not tool_name:
            raise TypeError("Missing required parameter: name")

        tool = self._tools.get(tool_name)
        if tool is None:
            raise TypeError(f"Unknown tool: {tool_name}")

        arguments: dict = params.get("arguments") or {}
        result = await tool.execute(**arguments)

        if result.success:
            text = result.output or ""
        else:
            text = f"Error: {result.error}"

        content = [{"type": "text", "text": text}]
        return {"content": content, "isError": not result.success}

    async def _handle_resources_list(self, params: dict) -> dict:
        resources: list[dict] = []
        for dirpath, _dirnames, filenames in os.walk(self._project_root):
            # Skip hidden directories and common non-essential dirs
            rel_dir = os.path.relpath(dirpath, self._project_root)
            parts = rel_dir.split(os.sep)
            skip = ("node_modules", "__pycache__", ".git")
            if any(p.startswith(".") or p in skip for p in parts):
                if rel_dir != ".":
                    continue

            for fname in sorted(filenames):
                if fname.startswith("."):
                    continue
                full_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(full_path, self._project_root)
                uri = f"file:///{os.path.abspath(full_path)}"
                resources.append({
                    "uri": uri,
                    "name": rel_path,
                    "mimeType": _guess_mime(fname),
                })
        return {"resources": resources}

    async def _handle_resources_read(self, params: dict) -> dict:
        uri: str | None = params.get("uri")
        if not uri:
            raise TypeError("Missing required parameter: uri")

        # Strip file:/// prefix to get the absolute path
        if uri.startswith("file:///"):
            file_path = uri[len("file:///"):]
            if not os.path.isabs(file_path):
                file_path = "/" + file_path
        else:
            raise TypeError(f"Unsupported URI scheme: {uri}")

        file_path = os.path.normpath(file_path)
        if not file_path.startswith(self._project_root):
            raise TypeError("Access denied: path outside project")

        if not os.path.isfile(file_path):
            raise TypeError(f"File not found: {file_path}")

        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": _guess_mime(os.path.basename(file_path)),
                    "text": content,
                }
            ],
        }

    async def _handle_noop(self, params: dict) -> dict:
        return {}

    async def _handle_ping(self, params: dict) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Stdio transport
    # ------------------------------------------------------------------

    async def run_stdio(self) -> None:
        """Read JSON-RPC messages from stdin (line-delimited), write responses to stdout."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin.buffer)

        # Use raw stdout to avoid encoding issues
        transport, _ = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout.buffer,
        )

        logger.info("MCP server started (project: %s)", self._project_root)

        while True:
            line = await reader.readline()
            if not line:
                break  # EOF

            line_str = line.decode("utf-8", errors="replace").strip()
            if not line_str:
                continue

            try:
                msg = json.loads(line_str)
            except json.JSONDecodeError as exc:
                response = self._error_response(None, PARSE_ERROR, f"Parse error: {exc}")
                transport.write((json.dumps(response) + "\n").encode("utf-8"))
                continue

            response = await self._handle_message(msg)
            if response is not None:
                transport.write((json.dumps(response) + "\n").encode("utf-8"))

        logger.info("MCP server shutting down")

    async def run(self) -> None:
        """Main entry point — start the stdio transport."""
        await self.run_stdio()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_MIME_MAP: dict[str, str] = {
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".json": "application/json",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".html": "text/html",
    ".css": "text/css",
    ".toml": "text/x-toml",
    ".cfg": "text/plain",
    ".sh": "text/x-shellscript",
    ".rs": "text/x-rust",
    ".go": "text/x-go",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".h": "text/x-c",
    ".rb": "text/x-ruby",
}


def _guess_mime(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return _MIME_MAP.get(ext.lower(), "text/plain")


# ------------------------------------------------------------------
# Standalone entry point
# ------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python -m codator.infrastructure.mcp_server``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,  # Keep stdout clean for JSON-RPC
    )
    server = MCPServer(project_root=os.getcwd())
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
