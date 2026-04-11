"""Profiler tool — run py-spy, cProfile, perf; identify hot paths and bottlenecks."""

import asyncio
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class ProfilerTool:
    name = "profiler"
    description = (
        "Profile code to identify performance bottlenecks. Supports: "
        "cProfile (Python), py-spy (live sampling), time command, and custom benchmarks. "
        "Returns hot paths, call counts, and timing data."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["profile", "py_spy", "time", "memory", "analyze_output"],
                "description": "Profiling method to use",
            },
            "command": {
                "type": "string",
                "description": "Command to profile (e.g., 'python script.py')",
            },
            "path": {
                "type": "string",
                "description": "Working directory or script path",
            },
            "duration": {
                "type": "integer",
                "description": "Profiling duration in seconds (for py-spy)",
                "default": 10,
            },
            "pid": {
                "type": "integer",
                "description": "Process ID to attach to (for py-spy)",
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top functions to show",
                "default": 20,
            },
            "sort_by": {
                "type": "string",
                "enum": ["cumulative", "tottime", "calls", "filename"],
                "description": "Sort profiling results by",
                "default": "cumulative",
            },
            "output_file": {
                "type": "string",
                "description": "Path to save profiling output (e.g., .prof, .svg)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "profile":
                return await self._cprofile(kwargs)
            elif action == "py_spy":
                return await self._py_spy(kwargs)
            elif action == "time":
                return await self._time_command(kwargs)
            elif action == "memory":
                return await self._memory_profile(kwargs)
            elif action == "analyze_output":
                return await self._analyze_output(kwargs)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _run_command(self, cmd: list, cwd: str = None, timeout: int = 120) -> tuple:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return "", "Profiling timed out", -1
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _cprofile(self, kwargs: dict) -> ToolResult:
        """Run Python cProfile on a command or script."""
        command = kwargs.get("command", "")
        path = kwargs.get("path", ".")
        top_n = kwargs.get("top_n", 20)
        sort_by = kwargs.get("sort_by", "cumulative")
        output_file = kwargs.get("output_file", "")

        if not command:
            return ToolResult(success=False, error="command is required for profile action")

        with tempfile.NamedTemporaryFile(suffix=".prof", delete=False) as f:
            prof_path = f.name

        # Build profiling command
        script = f"""
import cProfile
import pstats
import sys
import os

os.chdir({repr(path)})
prof = cProfile.Profile()
prof.enable()

# Execute the command
exec(open({repr(command.replace('python ', '').replace('python3 ', ''))}).read())

prof.disable()
prof.dump_stats({repr(prof_path)})

# Print stats
stats = pstats.Stats({repr(prof_path)})
stats.sort_stats({repr(sort_by)})
stats.print_stats({top_n})
"""
        # Simpler approach: use -m cProfile directly
        parts = command.split()
        if parts[0] in ("python", "python3"):
            script_path = parts[1] if len(parts) > 1 else ""
            args = parts[2:] if len(parts) > 2 else []
        else:
            script_path = parts[0]
            args = parts[1:]

        cmd = ["python", "-m", "cProfile", "-s", sort_by]
        if output_file:
            cmd.extend(["-o", output_file])
        cmd.append(script_path)
        cmd.extend(args)

        stdout, stderr, rc = await self._run_command(cmd, cwd=path)

        # Clean up temp file
        try:
            os.unlink(prof_path)
        except OSError:
            pass

        if rc != 0 and not stdout:
            return ToolResult(success=False, error=f"Profiling failed: {stderr}")

        # Parse top functions
        hot_functions = self._parse_cprofile_output(stdout, top_n)

        return ToolResult(
            success=True,
            output=stdout[:5000] if stdout else stderr[:2000],
            artifacts={"hot_functions": hot_functions, "sort_by": sort_by},
        )

    def _parse_cprofile_output(self, output: str, top_n: int) -> list:
        """Parse cProfile text output into structured data."""
        functions = []
        in_stats = False
        for line in output.split("\n"):
            if "ncalls" in line and "tottime" in line:
                in_stats = True
                continue
            if in_stats and line.strip():
                parts = line.split(None, 5)
                if len(parts) >= 6:
                    try:
                        functions.append({
                            "ncalls": parts[0],
                            "tottime": float(parts[1]),
                            "cumtime": float(parts[3]),
                            "function": parts[5],
                        })
                    except (ValueError, IndexError):
                        continue
            if len(functions) >= top_n:
                break
        return functions

    async def _py_spy(self, kwargs: dict) -> ToolResult:
        """Use py-spy for sampling profiler."""
        command = kwargs.get("command", "")
        pid = kwargs.get("pid")
        duration = kwargs.get("duration", 10)
        output_file = kwargs.get("output_file", "")
        top_n = kwargs.get("top_n", 20)

        if pid:
            # Attach to running process
            cmd = ["py-spy", "top", "--pid", str(pid), "--duration", str(duration)]
        elif command:
            # Profile a command
            if output_file and output_file.endswith(".svg"):
                cmd = ["py-spy", "record", "-o", output_file, "--duration", str(duration), "--"] + command.split()
            else:
                cmd = ["py-spy", "top", "--duration", str(duration), "--"] + command.split()
        else:
            return ToolResult(success=False, error="Either 'command' or 'pid' is required for py_spy")

        stdout, stderr, rc = await self._run_command(cmd, timeout=duration + 30)

        if rc != 0 and "Permission" in stderr:
            return ToolResult(
                success=False,
                error=f"py-spy needs elevated permissions. Try: sudo {' '.join(cmd)}\n{stderr}",
            )

        output = stdout or stderr
        if output_file:
            output += f"\n\nFlamegraph saved to: {output_file}"

        return ToolResult(success=True, output=output[:5000], artifacts={"duration": duration})

    async def _time_command(self, kwargs: dict) -> ToolResult:
        """Time a command execution with detailed stats."""
        command = kwargs.get("command", "")
        path = kwargs.get("path", ".")

        if not command:
            return ToolResult(success=False, error="command is required for time action")

        # Use /usr/bin/time for detailed output
        cmd = ["/usr/bin/time", "-v"] + command.split()
        stdout, stderr, rc = await self._run_command(cmd, cwd=path, timeout=300)

        # Parse time output
        timing = {}
        for line in stderr.split("\n"):
            if ":" in line:
                key, _, value = line.strip().partition(":")
                timing[key.strip()] = value.strip()

        output = f"Command: {command}\nExit code: {rc}\n\n"
        if timing:
            output += "Timing:\n"
            important_keys = [
                "Elapsed (wall clock) time",
                "Maximum resident set size",
                "User time (seconds)",
                "System time (seconds)",
                "Percent of CPU this job got",
                "Voluntary context switches",
            ]
            for key in important_keys:
                if key in timing:
                    output += f"  {key}: {timing[key]}\n"
        else:
            output += stderr

        if stdout:
            output += f"\nOutput:\n{stdout[:2000]}"

        return ToolResult(success=True, output=output, artifacts=timing)

    async def _memory_profile(self, kwargs: dict) -> ToolResult:
        """Profile memory usage of a Python script."""
        command = kwargs.get("command", "")
        path = kwargs.get("path", ".")

        if not command:
            return ToolResult(success=False, error="command is required for memory action")

        parts = command.split()
        script_path = parts[1] if len(parts) > 1 and parts[0] in ("python", "python3") else parts[0]

        # Try memory_profiler
        cmd = ["python", "-m", "memory_profiler", script_path]
        stdout, stderr, rc = await self._run_command(cmd, cwd=path, timeout=120)

        if rc != 0 and "No module named" in stderr:
            # Fallback: use tracemalloc
            tracemalloc_script = f"""
import tracemalloc
import sys
tracemalloc.start()
exec(open('{script_path}').read())
snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics('lineno')
print("Top memory allocations:")
for stat in stats[:20]:
    print(f"  {{stat}}")
current, peak = tracemalloc.get_traced_memory()
print(f"\\nCurrent: {{current/1024/1024:.1f}} MB")
print(f"Peak: {{peak/1024/1024:.1f}} MB")
"""
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(tracemalloc_script)
                tmp_script = f.name

            stdout, stderr, rc = await self._run_command(["python", tmp_script], cwd=path)
            os.unlink(tmp_script)

        return ToolResult(
            success=rc == 0,
            output=stdout[:5000] if stdout else stderr[:2000],
            error=stderr if rc != 0 else "",
        )

    async def _analyze_output(self, kwargs: dict) -> ToolResult:
        """Analyze an existing profiling output file (.prof, .pstats)."""
        path = kwargs.get("path", "")
        top_n = kwargs.get("top_n", 20)
        sort_by = kwargs.get("sort_by", "cumulative")

        if not path or not os.path.isfile(path):
            return ToolResult(success=False, error=f"Profile file not found: {path}")

        cmd = ["python", "-c", f"""
import pstats
stats = pstats.Stats('{path}')
stats.sort_stats('{sort_by}')
stats.print_stats({top_n})
stats.print_callers({top_n // 2})
"""]
        stdout, stderr, rc = await self._run_command(cmd)

        if rc != 0:
            return ToolResult(success=False, error=stderr)

        return ToolResult(success=True, output=stdout[:5000])
