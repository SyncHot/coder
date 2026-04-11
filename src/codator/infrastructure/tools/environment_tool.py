"""Environment management tool — venv, packages, system info, dependency resolution."""

import asyncio
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class EnvironmentTool:
    name = "environment"
    description = (
        "Manage development environments: create/activate venv, list/install/upgrade packages, "
        "resolve dependency conflicts, check system info (OS, Python, RAM, disk, available tools). "
        "Supports Python (venv/pip), Node.js (nvm/npm), and system-level inspection."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "system_info", "python_info", "create_venv", "list_packages",
                    "install", "upgrade", "check_conflicts", "which", "disk_usage",
                    "node_info", "env_vars",
                ],
                "description": "Action to perform",
            },
            "path": {
                "type": "string",
                "description": "Project directory or venv path",
            },
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Package names for install/upgrade actions",
            },
            "python_version": {
                "type": "string",
                "description": "Python version for venv creation (e.g., 'python3.11')",
            },
            "tool_name": {
                "type": "string",
                "description": "Tool name for 'which' action (check if available)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "system_info":
                return await self._system_info()
            elif action == "python_info":
                return await self._python_info(kwargs.get("path"))
            elif action == "create_venv":
                return await self._create_venv(kwargs.get("path", ".venv"), kwargs.get("python_version"))
            elif action == "list_packages":
                return await self._list_packages(kwargs.get("path"))
            elif action == "install":
                return await self._install_packages(kwargs.get("packages", []), kwargs.get("path"))
            elif action == "upgrade":
                return await self._upgrade_packages(kwargs.get("packages", []), kwargs.get("path"))
            elif action == "check_conflicts":
                return await self._check_conflicts(kwargs.get("path"))
            elif action == "which":
                return await self._which(kwargs.get("tool_name", ""))
            elif action == "disk_usage":
                return await self._disk_usage(kwargs.get("path", "."))
            elif action == "node_info":
                return await self._node_info()
            elif action == "env_vars":
                return await self._env_vars()
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _run_command(self, cmd: list, cwd: str = None) -> tuple:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _system_info(self) -> ToolResult:
        """Comprehensive system information."""
        import shutil

        info = {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "architecture": platform.machine(),
            "hostname": platform.node(),
            "python_version": platform.python_version(),
            "python_path": sys.executable,
        }

        # CPU info
        try:
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            cores = cpuinfo.count("processor")
            model = ""
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    model = line.split(":")[1].strip()
                    break
            info["cpu_model"] = model
            info["cpu_cores"] = cores
        except (IOError, OSError):
            info["cpu_cores"] = os.cpu_count()

        # Memory info
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        mem_kb = int(line.split()[1])
                        info["ram_gb"] = round(mem_kb / 1024 / 1024, 1)
                    elif "MemAvailable" in line:
                        avail_kb = int(line.split()[1])
                        info["ram_available_gb"] = round(avail_kb / 1024 / 1024, 1)
        except (IOError, OSError):
            pass

        # Disk info
        total, used, free = shutil.disk_usage("/")
        info["disk_total_gb"] = round(total / (1024**3), 1)
        info["disk_used_gb"] = round(used / (1024**3), 1)
        info["disk_free_gb"] = round(free / (1024**3), 1)

        # Available dev tools
        tools_to_check = ["git", "docker", "node", "npm", "go", "rustc", "cargo", "make", "gcc", "java"]
        available_tools = []
        for tool in tools_to_check:
            if shutil.which(tool):
                available_tools.append(tool)
        info["available_tools"] = available_tools

        lines = []
        for k, v in info.items():
            lines.append(f"  {k}: {v}")

        return ToolResult(success=True, output="System Info:\n" + "\n".join(lines), artifacts=info)

    async def _python_info(self, path: str = None) -> ToolResult:
        """Python environment details."""
        python_cmd = self._find_python(path)

        stdout, _, _ = await self._run_command([python_cmd, "--version"])
        version = stdout.strip()

        stdout, _, _ = await self._run_command([python_cmd, "-c", "import sys; print(sys.prefix)"])
        prefix = stdout.strip()

        stdout, _, _ = await self._run_command([python_cmd, "-c", "import sys; print(sys.executable)"])
        executable = stdout.strip()

        stdout, _, _ = await self._run_command([python_cmd, "-c", "import site; print('\\n'.join(site.getsitepackages()))"])
        site_packages = stdout.strip()

        # Check if in virtualenv
        stdout_venv, _, _ = await self._run_command([python_cmd, "-c", "import sys; print(sys.prefix != sys.base_prefix)"])
        is_venv = "True" in stdout_venv

        info = f"Python: {version}\nExecutable: {executable}\nPrefix: {prefix}\nVirtualenv: {is_venv}\nSite-packages:\n  {site_packages}"

        return ToolResult(success=True, output=info, artifacts={"version": version, "is_venv": is_venv, "prefix": prefix})

    def _find_python(self, path: str = None) -> str:
        """Find the project's python executable."""
        if path:
            venv_python = os.path.join(path, ".venv", "bin", "python")
            if os.path.isfile(venv_python):
                return venv_python
            venv_python2 = os.path.join(path, "venv", "bin", "python")
            if os.path.isfile(venv_python2):
                return venv_python2
        return sys.executable

    async def _create_venv(self, path: str, python_version: str = None) -> ToolResult:
        """Create a virtual environment."""
        python_cmd = python_version or "python3"
        cmd = [python_cmd, "-m", "venv", path]

        stdout, stderr, rc = await self._run_command(cmd)
        if rc != 0:
            return ToolResult(success=False, error=f"Failed to create venv: {stderr}")

        # Upgrade pip
        pip_path = os.path.join(path, "bin", "pip")
        await self._run_command([pip_path, "install", "--upgrade", "pip", "--quiet"])

        return ToolResult(
            success=True,
            output=f"Created virtual environment at {path}\nActivate with: source {path}/bin/activate",
            artifacts={"venv_path": path, "python": os.path.join(path, "bin", "python")},
        )

    async def _list_packages(self, path: str = None) -> ToolResult:
        """List installed packages."""
        python_cmd = self._find_python(path)
        stdout, _, rc = await self._run_command([python_cmd, "-m", "pip", "list", "--format=json"])

        if rc != 0:
            return ToolResult(success=False, error="Failed to list packages")

        try:
            packages = json.loads(stdout)
            lines = [f"Installed packages ({len(packages)}):\n"]
            for pkg in sorted(packages, key=lambda x: x["name"].lower()):
                lines.append(f"  {pkg['name']}=={pkg['version']}")
            return ToolResult(success=True, output="\n".join(lines)[:5000], artifacts={"count": len(packages)})
        except json.JSONDecodeError:
            return ToolResult(success=True, output=stdout[:3000])

    async def _install_packages(self, packages: list, path: str = None) -> ToolResult:
        """Install packages."""
        if not packages:
            return ToolResult(success=False, error="No packages specified")

        python_cmd = self._find_python(path)
        cmd = [python_cmd, "-m", "pip", "install"] + packages
        stdout, stderr, rc = await self._run_command(cmd)

        if rc != 0:
            return ToolResult(success=False, error=f"Install failed:\n{stderr}", output=stdout)

        return ToolResult(success=True, output=f"Installed: {', '.join(packages)}\n{stdout[-500:]}")

    async def _upgrade_packages(self, packages: list, path: str = None) -> ToolResult:
        """Upgrade packages."""
        if not packages:
            # Upgrade all outdated
            python_cmd = self._find_python(path)
            stdout, _, _ = await self._run_command([python_cmd, "-m", "pip", "list", "--outdated", "--format=json"])
            try:
                outdated = json.loads(stdout)
                packages = [p["name"] for p in outdated]
            except json.JSONDecodeError:
                return ToolResult(success=False, error="Could not determine outdated packages")

        if not packages:
            return ToolResult(success=True, output="All packages are up to date")

        python_cmd = self._find_python(path)
        cmd = [python_cmd, "-m", "pip", "install", "--upgrade"] + packages
        stdout, stderr, rc = await self._run_command(cmd)

        if rc != 0:
            return ToolResult(success=False, error=stderr, output=stdout)

        return ToolResult(success=True, output=f"Upgraded: {', '.join(packages)}")

    async def _check_conflicts(self, path: str = None) -> ToolResult:
        """Check for dependency conflicts."""
        python_cmd = self._find_python(path)
        stdout, stderr, rc = await self._run_command([python_cmd, "-m", "pip", "check"])

        if rc == 0:
            return ToolResult(success=True, output="✅ No dependency conflicts detected")

        return ToolResult(
            success=False,
            output=f"⚠️  Dependency conflicts found:\n{stdout}\n{stderr}",
        )

    async def _which(self, tool_name: str) -> ToolResult:
        """Check if a tool is available and get its version."""
        if not tool_name:
            return ToolResult(success=False, error="tool_name is required")

        path = shutil.which(tool_name)
        if not path:
            return ToolResult(success=False, output=f"'{tool_name}' not found in PATH")

        # Try to get version
        version = ""
        for flag in ["--version", "-v", "version"]:
            stdout, stderr, rc = await self._run_command([tool_name, flag])
            output = (stdout + stderr).strip()
            if rc == 0 and output:
                version = output.split("\n")[0]
                break

        return ToolResult(
            success=True,
            output=f"{tool_name}: {path}\nVersion: {version}" if version else f"{tool_name}: {path}",
            artifacts={"path": path, "version": version},
        )

    async def _disk_usage(self, path: str) -> ToolResult:
        """Check disk usage of a directory."""
        stdout, _, rc = await self._run_command(["du", "-sh", path])
        if rc != 0:
            total, used, free = shutil.disk_usage(path)
            return ToolResult(
                success=True,
                output=f"Disk: {round(used/1024**3, 1)}GB used / {round(total/1024**3, 1)}GB total ({round(free/1024**3, 1)}GB free)",
            )

        # Also get top subdirectories
        stdout2, _, _ = await self._run_command(["du", "-sh", "--max-depth=1", path])
        return ToolResult(success=True, output=f"Total: {stdout.strip()}\n\nBreakdown:\n{stdout2[:3000]}")

    async def _node_info(self) -> ToolResult:
        """Node.js environment info."""
        results = []

        stdout, _, rc = await self._run_command(["node", "--version"])
        if rc == 0:
            results.append(f"Node.js: {stdout.strip()}")

        stdout, _, rc = await self._run_command(["npm", "--version"])
        if rc == 0:
            results.append(f"npm: {stdout.strip()}")

        stdout, _, rc = await self._run_command(["npx", "--version"])
        if rc == 0:
            results.append(f"npx: {stdout.strip()}")

        # Check for nvm
        nvm_dir = os.environ.get("NVM_DIR", os.path.expanduser("~/.nvm"))
        if os.path.isdir(nvm_dir):
            results.append(f"nvm: installed at {nvm_dir}")

        # Check for yarn/pnpm
        for tool in ["yarn", "pnpm", "bun"]:
            path = shutil.which(tool)
            if path:
                stdout, _, _ = await self._run_command([tool, "--version"])
                results.append(f"{tool}: {stdout.strip()}")

        if not results:
            return ToolResult(success=False, output="Node.js not found")

        return ToolResult(success=True, output="Node.js Environment:\n  " + "\n  ".join(results))

    async def _env_vars(self) -> ToolResult:
        """List relevant environment variables."""
        relevant_prefixes = ["PATH", "HOME", "USER", "SHELL", "LANG", "VIRTUAL_ENV",
                           "PYTHONPATH", "NODE_PATH", "GOPATH", "CARGO_HOME",
                           "EDITOR", "TERM", "SSH_", "DISPLAY", "XDG_"]
        env = {}
        for key, value in sorted(os.environ.items()):
            if any(key.startswith(p) or key == p for p in relevant_prefixes):
                env[key] = value

        lines = [f"  {k}={v}" for k, v in env.items()]
        return ToolResult(success=True, output="Environment Variables:\n" + "\n".join(lines)[:3000], artifacts=env)
