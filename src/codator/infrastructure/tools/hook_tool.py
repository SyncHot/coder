"""Hook/event tool — pre/post execution hooks on tool calls."""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


@dataclass
class Hook:
    name: str
    event: str  # "pre_tool", "post_tool", "on_error", "on_file_change"
    tool_filter: str = "*"  # Which tool(s) this hook applies to
    command: str = ""  # Shell command to execute
    script_path: str = ""  # Path to script to run
    enabled: bool = True
    created_at: float = 0


class HookTool:
    name = "hook"
    description = (
        "Manage execution hooks: register pre/post hooks on tool calls, "
        "file change watchers, and error handlers. Hooks can run shell commands "
        "or scripts when specific events occur. Similar to Claude Code hooks and git hooks."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["register", "unregister", "list", "enable", "disable", "trigger", "history"],
                "description": "Hook management action",
            },
            "name": {
                "type": "string",
                "description": "Hook name (unique identifier)",
            },
            "event": {
                "type": "string",
                "enum": ["pre_tool", "post_tool", "on_error", "on_file_change", "on_commit", "on_test_fail"],
                "description": "Event type to hook into",
            },
            "tool_filter": {
                "type": "string",
                "description": "Tool name filter (* for all tools, or specific tool name)",
                "default": "*",
            },
            "command": {
                "type": "string",
                "description": "Shell command to execute when hook fires",
            },
            "script_path": {
                "type": "string",
                "description": "Path to script to execute when hook fires",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        self._hooks: dict = {}  # name -> Hook
        self._history: list = []  # execution history
        self._hooks_file = os.path.expanduser("~/.codator/hooks.json")
        self._load_hooks()

    def _load_hooks(self):
        """Load hooks from persistent storage."""
        if os.path.isfile(self._hooks_file):
            try:
                with open(self._hooks_file) as f:
                    data = json.load(f)
                for name, hook_data in data.items():
                    self._hooks[name] = Hook(**hook_data)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_hooks(self):
        """Save hooks to persistent storage."""
        os.makedirs(os.path.dirname(self._hooks_file), exist_ok=True)
        data = {}
        for name, hook in self._hooks.items():
            data[name] = {
                "name": hook.name,
                "event": hook.event,
                "tool_filter": hook.tool_filter,
                "command": hook.command,
                "script_path": hook.script_path,
                "enabled": hook.enabled,
                "created_at": hook.created_at,
            }
        with open(self._hooks_file, "w") as f:
            json.dump(data, f, indent=2)

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "register":
                return self._register(kwargs)
            elif action == "unregister":
                return self._unregister(kwargs)
            elif action == "list":
                return self._list_hooks()
            elif action == "enable":
                return self._toggle(kwargs, True)
            elif action == "disable":
                return self._toggle(kwargs, False)
            elif action == "trigger":
                return await self._trigger(kwargs)
            elif action == "history":
                return self._get_history()
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _register(self, kwargs: dict) -> ToolResult:
        """Register a new hook."""
        name = kwargs.get("name", "")
        event = kwargs.get("event", "")
        command = kwargs.get("command", "")
        script_path = kwargs.get("script_path", "")

        if not name:
            return ToolResult(success=False, error="name is required")
        if not event:
            return ToolResult(success=False, error="event is required")
        if not command and not script_path:
            return ToolResult(success=False, error="Either command or script_path is required")

        if script_path and not os.path.isfile(script_path):
            return ToolResult(success=False, error="Script not found: " + script_path)

        hook = Hook(
            name=name,
            event=event,
            tool_filter=kwargs.get("tool_filter", "*"),
            command=command,
            script_path=script_path,
            enabled=True,
            created_at=time.time(),
        )
        self._hooks[name] = hook
        self._save_hooks()

        return ToolResult(
            success=True,
            output="Registered hook: " + name + " (on " + event + ")",
            artifacts={"name": name, "event": event},
        )

    def _unregister(self, kwargs: dict) -> ToolResult:
        """Remove a hook."""
        name = kwargs.get("name", "")
        if not name:
            return ToolResult(success=False, error="name is required")
        if name not in self._hooks:
            return ToolResult(success=False, error="Hook not found: " + name)

        del self._hooks[name]
        self._save_hooks()
        return ToolResult(success=True, output="Unregistered hook: " + name)

    def _list_hooks(self) -> ToolResult:
        """List all registered hooks."""
        if not self._hooks:
            return ToolResult(success=True, output="No hooks registered")

        lines = ["Registered hooks (" + str(len(self._hooks)) + "):\n"]
        for name, hook in sorted(self._hooks.items()):
            status = "enabled" if hook.enabled else "disabled"
            cmd = hook.command or hook.script_path
            lines.append("  " + name + " [" + status + "]")
            lines.append("    Event: " + hook.event + " | Filter: " + hook.tool_filter)
            lines.append("    Command: " + cmd[:80])
            lines.append("")

        return ToolResult(success=True, output="\n".join(lines))

    def _toggle(self, kwargs: dict, enabled: bool) -> ToolResult:
        """Enable or disable a hook."""
        name = kwargs.get("name", "")
        if not name:
            return ToolResult(success=False, error="name is required")
        if name not in self._hooks:
            return ToolResult(success=False, error="Hook not found: " + name)

        self._hooks[name].enabled = enabled
        self._save_hooks()
        state = "enabled" if enabled else "disabled"
        return ToolResult(success=True, output="Hook '" + name + "' " + state)

    async def _trigger(self, kwargs: dict) -> ToolResult:
        """Manually trigger a hook or event."""
        name = kwargs.get("name", "")
        event = kwargs.get("event", "")

        hooks_to_run = []
        if name:
            if name in self._hooks:
                hooks_to_run.append(self._hooks[name])
            else:
                return ToolResult(success=False, error="Hook not found: " + name)
        elif event:
            hooks_to_run = [h for h in self._hooks.values() if h.event == event and h.enabled]
        else:
            return ToolResult(success=False, error="Either name or event is required for trigger")

        if not hooks_to_run:
            return ToolResult(success=True, output="No hooks to trigger")

        results = []
        for hook in hooks_to_run:
            result = await self._execute_hook(hook)
            results.append(result)

        output = "Triggered " + str(len(hooks_to_run)) + " hook(s):\n"
        for hook, (success, out) in zip(hooks_to_run, results):
            icon = "ok" if success else "FAIL"
            output += "  " + hook.name + ": " + icon + "\n"
            if out:
                output += "    " + out[:100] + "\n"

        all_success = all(r[0] for r in results)
        return ToolResult(success=all_success, output=output)

    async def _execute_hook(self, hook: Hook) -> tuple:
        """Execute a single hook and return (success, output)."""
        cmd = hook.command or ("bash " + hook.script_path if hook.script_path else "")
        if not cmd:
            return (False, "No command configured")

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = stdout.decode() + stderr.decode()
            success = proc.returncode == 0

            self._history.append({
                "hook": hook.name,
                "event": hook.event,
                "success": success,
                "output": output[:200],
                "timestamp": time.time(),
            })
            # Keep only last 100 history entries
            self._history = self._history[-100:]

            return (success, output[:200])
        except asyncio.TimeoutError:
            return (False, "Hook timed out (30s)")
        except Exception as e:
            return (False, str(e))

    def _get_history(self) -> ToolResult:
        """Get hook execution history."""
        if not self._history:
            return ToolResult(success=True, output="No hook execution history")

        lines = ["Hook execution history (last " + str(len(self._history)) + "):\n"]
        for entry in reversed(self._history[-20:]):
            icon = "ok" if entry["success"] else "FAIL"
            lines.append("  [" + icon + "] " + entry["hook"] + " (" + entry["event"] + ")")
            if entry.get("output"):
                lines.append("    " + entry["output"][:80])

        return ToolResult(success=True, output="\n".join(lines))

    async def fire_event(self, event: str, tool_name: str = "", context: dict = None) -> list:
        """Fire an event and run matching hooks. Called by chat_engine."""
        matching = [
            h for h in self._hooks.values()
            if h.enabled and h.event == event and (h.tool_filter == "*" or h.tool_filter == tool_name)
        ]

        results = []
        for hook in matching:
            result = await self._execute_hook(hook)
            results.append((hook.name, result))

        return results
