"""MCP client — connects to external MCP servers over stdio and registers their tools."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

if TYPE_CHECKING:
    from codator.core.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

_MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPClientTool(Tool):
    """Wraps a single tool exposed by an external MCP server."""

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        client: MCPClient,
    ) -> None:
        self._name = name
        self._description = description
        self._input_schema = input_schema
        self._client = client

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict:
        return self._input_schema

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            output = await self._client.call_tool(self._name, kwargs)
            return ToolResult(success=True, output=output)
        except MCPError as exc:
            return ToolResult(success=False, error=str(exc))
        except Exception as exc:
            return ToolResult(success=False, error=f"MCP tool execution error: {exc}")

    async def close(self) -> None:
        # Lifecycle is managed by the MCPClient, not individual tools.
        pass


class MCPError(Exception):
    """Raised when an MCP JSON-RPC exchange fails."""


class MCPClient:
    """Manages a connection to a single MCP server over stdio."""

    def __init__(
        self,
        server_command: list[str],
        server_env: dict[str, str] | None = None,
    ) -> None:
        self._server_command = server_command
        self._server_env = server_env
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Launch the server process and perform the MCP ``initialize`` handshake."""
        logger.info("Starting MCP server: %s", self._server_command)
        self._process = await asyncio.create_subprocess_exec(
            *self._server_command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._server_env,
        )

        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "codator", "version": "1.0.0"},
            },
        )

        server_version = response.get("protocolVersion", "")
        if not server_version:
            raise MCPError("Server did not return a protocolVersion in initialize response")

        logger.info(
            "MCP server initialized (protocol %s, server %s)",
            server_version,
            response.get("serverInfo", {}),
        )

        # Send initialized notification (no id, no response expected).
        await self._send_notification("notifications/initialized", {})

    async def close(self) -> None:
        """Shut down the server process gracefully."""
        if self._process is None:
            return

        try:
            await self._send_notification("notifications/cancelled", {})
        except Exception:
            logger.debug("Failed to send shutdown notification", exc_info=True)

        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except TimeoutError:
            logger.warning("MCP server did not exit in time, killing")
            self._process.kill()
            await self._process.wait()
        except ProcessLookupError:
            pass
        finally:
            self._process = None
            logger.info("MCP server stopped: %s", self._server_command)

    # ------------------------------------------------------------------
    # Public RPC helpers
    # ------------------------------------------------------------------

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of tool definitions advertised by the server."""
        response = await self._send_request("tools/list", {})
        return response.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool on the server and return its text content."""
        response = await self._send_request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )

        is_error = response.get("isError", False)
        content_parts = response.get("content", [])
        text_pieces: list[str] = []
        for part in content_parts:
            if part.get("type") == "text":
                text_pieces.append(part.get("text", ""))

        text = "\n".join(text_pieces) if text_pieces else json.dumps(content_parts)

        if is_error:
            raise MCPError(f"MCP tool '{name}' returned an error: {text}")

        return text

    async def discover_and_register(self, registry: ToolRegistry) -> int:
        """Discover tools from the server and register them in *registry*.

        Returns the number of tools registered.
        """
        tools = await self.list_tools()
        count = 0
        for tool_def in tools:
            tool = MCPClientTool(
                name=tool_def["name"],
                description=tool_def.get("description", ""),
                input_schema=tool_def.get("inputSchema", {"type": "object", "properties": {}}),
                client=self,
            )
            registry.register(tool)
            count += 1
            logger.info("Registered MCP tool: %s", tool.name)
        return count

    # ------------------------------------------------------------------
    # Low-level JSON-RPC transport
    # ------------------------------------------------------------------

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and wait for the corresponding response."""
        async with self._lock:
            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise MCPError("MCP server process is not running")

            request_id = self._next_id
            self._next_id += 1

            message = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            }

            line = json.dumps(message) + "\n"
            self._process.stdin.write(line.encode())
            await self._process.stdin.drain()
            logger.debug("MCP → %s (id=%d)", method, request_id)

            # Read lines until we get a response matching our request id.
            while True:
                raw = await self._process.stdout.readline()
                if not raw:
                    raise MCPError(
                        f"MCP server closed stdout while waiting for response to '{method}'"
                    )

                raw_str = raw.decode().strip()
                if not raw_str:
                    continue

                try:
                    response = json.loads(raw_str)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON line from MCP server: %s", raw_str[:200])
                    continue

                # Skip notifications (no "id" field).
                if "id" not in response:
                    logger.debug("MCP notification (ignored): %s", response.get("method", "?"))
                    continue

                if response["id"] != request_id:
                    logger.warning(
                        "MCP response id mismatch: expected %d, got %s",
                        request_id,
                        response.get("id"),
                    )
                    continue

                if "error" in response:
                    err = response["error"]
                    code = err.get("code", -1)
                    msg = err.get("message", "Unknown error")
                    raise MCPError(f"JSON-RPC error {code}: {msg}")

                return response.get("result", {})

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no ``id``, no response expected)."""
        if self._process is None or self._process.stdin is None:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        line = json.dumps(message) + "\n"
        self._process.stdin.write(line.encode())
        await self._process.stdin.drain()
        logger.debug("MCP notification → %s", method)


class MCPManager:
    """Manages multiple MCP server connections."""

    def __init__(self) -> None:
        self._clients: dict[str, MCPClient] = {}

    async def connect_server(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
    ) -> int:
        """Connect to an MCP server and return the number of tools it exposes."""
        if name in self._clients:
            logger.warning("MCP server '%s' already connected, disconnecting first", name)
            await self.disconnect_server(name)

        client = MCPClient(server_command=command, server_env=env)
        await client.connect()
        tools = await client.list_tools()
        self._clients[name] = client
        logger.info("Connected MCP server '%s' with %d tools", name, len(tools))
        return len(tools)

    async def disconnect_server(self, name: str) -> None:
        """Disconnect and shut down a single MCP server."""
        client = self._clients.pop(name, None)
        if client is not None:
            await client.close()
            logger.info("Disconnected MCP server '%s'", name)

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        names = list(self._clients.keys())
        for name in names:
            await self.disconnect_server(name)

    async def register_all(self, registry: ToolRegistry) -> int:
        """Register tools from every connected server into *registry*.

        Returns the total number of tools registered.
        """
        total = 0
        for name, client in self._clients.items():
            count = await client.discover_and_register(registry)
            logger.info("Registered %d tools from MCP server '%s'", count, name)
            total += count
        return total

    @property
    def connected_servers(self) -> list[str]:
        return list(self._clients.keys())
