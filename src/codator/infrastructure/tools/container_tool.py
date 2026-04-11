"""Container Tool — Docker operations for managing containers and images.

Provides structured docker operations: build, run, stop, logs, exec, ps,
images, and compose support.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class ContainerTool(Tool):
    """Manage Docker containers and images with structured output."""

    @property
    def name(self) -> str:
        return "container"

    @property
    def description(self) -> str:
        return (
            "Docker container management. Actions: ps (list running), "
            "run (start container), stop, logs, exec (run command in container), "
            "build, images, compose_up, compose_down, inspect."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "ps", "run", "stop", "logs", "exec", "build",
                        "images", "compose_up", "compose_down", "inspect",
                        "pull", "rm",
                    ],
                },
                "image": {"type": "string", "description": "Docker image name."},
                "container": {"type": "string", "description": "Container name or ID."},
                "command": {"type": "string", "description": "Command to run."},
                "ports": {"type": "string", "description": "Port mapping (e.g., '8080:80')."},
                "env": {"type": "object", "description": "Environment variables."},
                "volumes": {"type": "string", "description": "Volume mount."},
                "dockerfile": {"type": "string", "description": "Dockerfile path."},
                "tag": {"type": "string", "description": "Image tag."},
                "compose_file": {"type": "string", "description": "docker-compose file."},
                "tail": {"type": "integer", "description": "Log lines (default: 100)."},
                "detach": {"type": "boolean", "description": "Run in background."},
            },
            "required": ["action"],
        }

    async def _run_docker(self, *args: str, timeout: int = 60) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "docker", *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return (
            proc.returncode or 0,
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
        )

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "ps")
        dispatch = {
            "ps": self._ps, "run": self._run, "stop": self._stop,
            "logs": self._logs, "exec": self._exec, "build": self._build,
            "images": self._images, "compose_up": self._compose_up,
            "compose_down": self._compose_down, "inspect": self._inspect,
            "pull": self._pull, "rm": self._rm,
        }
        handler = dispatch.get(action)
        if not handler:
            return ToolResult(success=False, error=f"Unknown action: {action}")
        return await handler(**kwargs)

    async def _ps(self, **kwargs) -> ToolResult:
        fmt = "table {{.ID}}\\t{{.Names}}\\t{{.Image}}\\t{{.Status}}\\t{{.Ports}}"
        rc, out, err = await self._run_docker("ps", "--format", fmt)
        if rc != 0:
            return ToolResult(success=False, error=err or "docker ps failed")
        return ToolResult(success=True, output=out or "No containers running.")

    async def _run(self, **kwargs) -> ToolResult:
        image = kwargs.get("image", "")
        if not image:
            return ToolResult(success=False, error="'image' required for run.")

        args = ["run"]
        if kwargs.get("detach", True):
            args.append("-d")

        container_name = kwargs.get("container", "")
        if container_name:
            args.extend(["--name", container_name])

        ports = kwargs.get("ports", "")
        if ports:
            args.extend(["-p", ports])

        volumes = kwargs.get("volumes", "")
        if volumes:
            args.extend(["-v", volumes])

        env_vars = kwargs.get("env", {})
        if env_vars:
            for k, v in env_vars.items():
                args.extend(["-e", f"{k}={v}"])

        args.append(image)

        command = kwargs.get("command", "")
        if command:
            args.extend(command.split())

        rc, out, err = await self._run_docker(*args)
        if rc != 0:
            return ToolResult(success=False, error=err or out)
        return ToolResult(success=True, output=f"Container started: {out.strip()[:64]}")

    async def _stop(self, **kwargs) -> ToolResult:
        container = kwargs.get("container", "")
        if not container:
            return ToolResult(success=False, error="'container' required.")
        rc, out, err = await self._run_docker("stop", container)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=f"Stopped: {container}")

    async def _logs(self, **kwargs) -> ToolResult:
        container = kwargs.get("container", "")
        if not container:
            return ToolResult(success=False, error="'container' required.")
        tail = str(kwargs.get("tail", 100))
        rc, out, err = await self._run_docker("logs", "--tail", tail, container)
        if rc != 0:
            return ToolResult(success=False, error=err)
        output = out or err
        if len(output) > 8000:
            output = output[-8000:]
        return ToolResult(success=True, output=output)

    async def _exec(self, **kwargs) -> ToolResult:
        container = kwargs.get("container", "")
        command = kwargs.get("command", "")
        if not container or not command:
            return ToolResult(success=False, error="'container' and 'command' required.")
        args = ["exec", container] + command.split()
        rc, out, err = await self._run_docker(*args)
        if rc != 0:
            return ToolResult(success=False, error=err or out)
        return ToolResult(success=True, output=out)

    async def _build(self, **kwargs) -> ToolResult:
        tag = kwargs.get("tag", "")
        dockerfile = kwargs.get("dockerfile", "Dockerfile")
        if not tag:
            return ToolResult(success=False, error="'tag' required for build.")
        args = ["build", "-t", tag, "-f", dockerfile, "."]
        rc, out, err = await self._run_docker(*args, timeout=300)
        if rc != 0:
            return ToolResult(success=False, error=f"Build failed:\n{err[-2000:]}")
        return ToolResult(success=True, output=f"Built image: {tag}\n{out[-500:]}")

    async def _images(self, **kwargs) -> ToolResult:
        fmt = "table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}\\t{{.CreatedSince}}"
        rc, out, err = await self._run_docker("images", "--format", fmt)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=out or "No images.")

    async def _compose_up(self, **kwargs) -> ToolResult:
        compose_file = kwargs.get("compose_file", "docker-compose.yml")
        args = ["compose", "-f", compose_file, "up", "-d"]
        rc, out, err = await self._run_docker(*args, timeout=120)
        if rc != 0:
            return ToolResult(success=False, error=err or out)
        return ToolResult(success=True, output=f"Compose up:\n{out}")

    async def _compose_down(self, **kwargs) -> ToolResult:
        compose_file = kwargs.get("compose_file", "docker-compose.yml")
        args = ["compose", "-f", compose_file, "down"]
        rc, out, err = await self._run_docker(*args)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=f"Compose down:\n{out}")

    async def _inspect(self, **kwargs) -> ToolResult:
        container = kwargs.get("container", "")
        if not container:
            return ToolResult(success=False, error="'container' required.")
        rc, out, err = await self._run_docker("inspect", container)
        if rc != 0:
            return ToolResult(success=False, error=err)
        try:
            data = json.loads(out)
            if isinstance(data, list) and data:
                info = data[0]
                summary = {
                    "id": info.get("Id", "")[:12],
                    "name": info.get("Name", ""),
                    "image": info.get("Config", {}).get("Image", ""),
                    "status": info.get("State", {}).get("Status", ""),
                    "started_at": info.get("State", {}).get("StartedAt", ""),
                    "ports": info.get("NetworkSettings", {}).get("Ports", {}),
                }
                return ToolResult(
                    success=True,
                    output=json.dumps(summary, indent=2),
                    artifacts=summary,
                )
        except json.JSONDecodeError:
            pass
        return ToolResult(success=True, output=out[:5000])

    async def _pull(self, **kwargs) -> ToolResult:
        image = kwargs.get("image", "")
        if not image:
            return ToolResult(success=False, error="'image' required.")
        rc, out, err = await self._run_docker("pull", image, timeout=120)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=f"Pulled: {image}\n{out[-300:]}")

    async def _rm(self, **kwargs) -> ToolResult:
        container = kwargs.get("container", "")
        if not container:
            return ToolResult(success=False, error="'container' required.")
        rc, out, err = await self._run_docker("rm", "-f", container)
        if rc != 0:
            return ToolResult(success=False, error=err)
        return ToolResult(success=True, output=f"Removed: {container}")
