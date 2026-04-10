"""Tool registry — discovers, holds, and dispatches agentic tools."""

from __future__ import annotations

import logging

from codator.domain.interfaces import Tool
from codator.domain.models import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry for all agentic tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool by its name."""
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def tool_descriptions(self) -> dict[str, str]:
        """Map of tool name → description for the model system prompt."""
        return {t.name: t.description for t in self._tools.values()}

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute a tool call, dispatching to the appropriate tool."""
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {call.tool_name}. "
                f"Available: {', '.join(self.tool_names)}",
            )
        try:
            return await tool.execute(**call.parameters)
        except Exception as exc:
            logger.exception("Tool %s failed", call.tool_name)
            return ToolResult(success=False, error=f"Tool execution error: {exc}")

    async def close_all(self) -> None:
        """Release all tool resources."""
        for tool in self._tools.values():
            try:
                await tool.close()
            except Exception:
                logger.warning("Error closing tool: %s", tool.name, exc_info=True)
        self._tools.clear()

    def tool_prompt_section(self) -> str:
        """Generate a tools description block for the system prompt."""
        if not self._tools:
            return ""
        lines = ["--- Available Tools ---"]
        for tool in self._tools.values():
            lines.append(f"• **{tool.name}**: {tool.description}")
        lines.append(
            "\nYou can call these tools directly to read files, list directories, "
            "and run commands. Use them proactively when the user asks about code."
        )
        return "\n".join(lines)

    def to_openai_tools(self) -> list[dict]:
        """Return tool definitions in OpenAI function-calling format."""
        tools = []
        for tool in self._tools.values():
            schema = tool.parameters_schema
            if schema:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                })
        return tools
