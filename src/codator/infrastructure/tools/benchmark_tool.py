"""Benchmark tool — run and compare performance benchmarks."""

import asyncio
import json
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class BenchmarkTool:
    name = "benchmark"
    description = (
        "Run performance benchmarks: time commands (like hyperfine), run pytest-benchmark, "
        "go bench, or custom timing loops. Compare results across runs and detect regressions."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "compare", "hyperfine", "pytest_bench", "go_bench"],
                "description": "Benchmark action to perform",
            },
            "commands": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Commands to benchmark (for run/hyperfine)",
            },
            "path": {
                "type": "string",
                "description": "Working directory or benchmark file path",
            },
            "iterations": {
                "type": "integer",
                "description": "Number of iterations per command",
                "default": 10,
            },
            "warmup": {
                "type": "integer",
                "description": "Number of warmup runs before measuring",
                "default": 2,
            },
            "baseline_file": {
                "type": "string",
                "description": "Path to baseline results JSON for comparison",
            },
            "output_file": {
                "type": "string",
                "description": "Save results to this JSON file",
            },
            "filter": {
                "type": "string",
                "description": "Filter benchmarks by name pattern (for pytest/go)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "run":
                return await self._run_benchmark(kwargs)
            elif action == "compare":
                return await self._compare(kwargs)
            elif action == "hyperfine":
                return await self._hyperfine(kwargs)
            elif action == "pytest_bench":
                return await self._pytest_bench(kwargs)
            elif action == "go_bench":
                return await self._go_bench(kwargs)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _run_command(self, cmd, cwd=None, timeout=300):
        if isinstance(cmd, str):
            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return "", "Timed out", -1
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _run_benchmark(self, kwargs: dict) -> ToolResult:
        """Run custom benchmarks with timing."""
        commands = kwargs.get("commands", [])
        iterations = kwargs.get("iterations", 10)
        warmup = kwargs.get("warmup", 2)
        path = kwargs.get("path", ".")
        output_file = kwargs.get("output_file")

        if not commands:
            return ToolResult(success=False, error="commands list is required")

        results = {}
        for cmd in commands:
            # Warmup
            for _ in range(warmup):
                await self._run_command(cmd, cwd=path)

            # Measure
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _, _, rc = await self._run_command(cmd, cwd=path)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            results[cmd] = {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "min": min(times),
                "max": max(times),
                "stddev": statistics.stdev(times) if len(times) > 1 else 0,
                "iterations": iterations,
            }

        # Format output
        lines = [f"Benchmark Results ({iterations} iterations, {warmup} warmup):\n"]
        for cmd, data in sorted(results.items(), key=lambda x: x[1]["mean"]):
            lines.append(f"  {cmd}")
            lines.append(f"    Mean:   {data['mean']*1000:.2f} ms")
            lines.append(f"    Median: {data['median']*1000:.2f} ms")
            lines.append(f"    Min:    {data['min']*1000:.2f} ms")
            lines.append(f"    Max:    {data['max']*1000:.2f} ms")
            lines.append(f"    Stddev: {data['stddev']*1000:.2f} ms")
            lines.append("")

        # Compare if multiple commands
        if len(results) > 1:
            fastest = min(results.items(), key=lambda x: x[1]["mean"])
            lines.append(f"  Fastest: {fastest[0]} ({fastest[1]['mean']*1000:.2f} ms)")
            for cmd, data in results.items():
                if cmd != fastest[0]:
                    ratio = data["mean"] / fastest[1]["mean"]
                    lines.append(f"  {cmd} is {ratio:.2f}x slower")

        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            lines.append(f"\nResults saved to: {output_file}")

        return ToolResult(success=True, output="\n".join(lines), artifacts=results)

    async def _compare(self, kwargs: dict) -> ToolResult:
        """Compare current benchmark with baseline."""
        baseline_file = kwargs.get("baseline_file", "")
        if not baseline_file or not os.path.isfile(baseline_file):
            return ToolResult(success=False, error="baseline_file is required and must exist")

        with open(baseline_file) as f:
            baseline = json.load(f)

        # Run current benchmarks
        commands = list(baseline.keys())
        kwargs["commands"] = commands
        current_result = await self._run_benchmark(kwargs)

        if not current_result.success:
            return current_result

        current = current_result.artifacts

        # Compare
        lines = ["Benchmark Comparison (current vs baseline):\n"]
        regressions = 0
        improvements = 0

        for cmd in commands:
            if cmd in current and cmd in baseline:
                curr_mean = current[cmd]["mean"]
                base_mean = baseline[cmd]["mean"]
                change_pct = ((curr_mean - base_mean) / base_mean) * 100

                if change_pct > 10:
                    icon = "🔴"
                    regressions += 1
                elif change_pct < -10:
                    icon = "🟢"
                    improvements += 1
                else:
                    icon = "⚪"

                lines.append(f"  {icon} {cmd}: {change_pct:+.1f}% ({base_mean*1000:.2f}ms → {curr_mean*1000:.2f}ms)")

        lines.append(f"\nSummary: {improvements} improvements, {regressions} regressions")

        return ToolResult(
            success=regressions == 0,
            output="\n".join(lines),
            artifacts={"regressions": regressions, "improvements": improvements},
        )

    async def _hyperfine(self, kwargs: dict) -> ToolResult:
        """Use hyperfine for accurate command benchmarking."""
        commands = kwargs.get("commands", [])
        warmup = kwargs.get("warmup", 2)
        iterations = kwargs.get("iterations", 10)
        path = kwargs.get("path", ".")

        if not commands:
            return ToolResult(success=False, error="commands list is required")

        cmd = ["hyperfine", "--warmup", str(warmup), "--min-runs", str(iterations), "--export-json", "/dev/stdout"]
        cmd.extend(commands)

        stdout, stderr, rc = await self._run_command(cmd, cwd=path)

        if rc != 0:
            if "not found" in stderr.lower() or rc == 127:
                # Fallback to manual timing
                return await self._run_benchmark(kwargs)
            return ToolResult(success=False, error=f"hyperfine failed: {stderr}")

        try:
            data = json.loads(stdout)
            lines = ["Hyperfine Results:\n"]
            for result in data.get("results", []):
                lines.append(f"  {result['command']}")
                lines.append(f"    Mean:   {result['mean']*1000:.2f} ms ± {result['stddev']*1000:.2f} ms")
                lines.append(f"    Min:    {result['min']*1000:.2f} ms")
                lines.append(f"    Max:    {result['max']*1000:.2f} ms")
                lines.append("")
            return ToolResult(success=True, output="\n".join(lines), artifacts=data)
        except json.JSONDecodeError:
            return ToolResult(success=True, output=stdout[:3000])

    async def _pytest_bench(self, kwargs: dict) -> ToolResult:
        """Run pytest-benchmark."""
        path = kwargs.get("path", ".")
        bench_filter = kwargs.get("filter", "")
        output_file = kwargs.get("output_file", "")

        cmd = ["python", "-m", "pytest", "--benchmark-only", "--benchmark-json=/dev/stdout", "-q"]
        if bench_filter:
            cmd.extend(["-k", bench_filter])

        stdout, stderr, rc = await self._run_command(cmd, cwd=path)

        if rc != 0 and not stdout:
            return ToolResult(success=False, error=f"pytest-benchmark failed: {stderr}")

        try:
            data = json.loads(stdout)
            benchmarks = data.get("benchmarks", [])
            lines = [f"pytest-benchmark ({len(benchmarks)} benchmarks):\n"]
            for b in sorted(benchmarks, key=lambda x: x.get("stats", {}).get("mean", 0)):
                stats = b.get("stats", {})
                lines.append(f"  {b['name']}")
                lines.append(f"    Mean: {stats.get('mean', 0)*1000:.3f} ms")
                lines.append(f"    Min:  {stats.get('min', 0)*1000:.3f} ms")
                lines.append(f"    Ops/s: {stats.get('ops', 0):.0f}")
                lines.append("")
            return ToolResult(success=True, output="\n".join(lines)[:5000], artifacts={"benchmarks": len(benchmarks)})
        except json.JSONDecodeError:
            return ToolResult(success=True, output=stdout[:3000])

    async def _go_bench(self, kwargs: dict) -> ToolResult:
        """Run Go benchmarks."""
        path = kwargs.get("path", ".")
        bench_filter = kwargs.get("filter", ".")
        iterations = kwargs.get("iterations", 1)

        cmd = ["go", "test", "-bench", bench_filter, "-benchmem", f"-count={iterations}", "./..."]
        stdout, stderr, rc = await self._run_command(cmd, cwd=path)

        if rc != 0:
            return ToolResult(success=False, error=f"go bench failed: {stderr}")

        # Parse results
        benchmarks = []
        for line in stdout.split("\n"):
            match = re.match(r'(Benchmark\w+)\s+(\d+)\s+([\d.]+)\s+ns/op', line)
            if match:
                benchmarks.append({
                    "name": match.group(1),
                    "iterations": int(match.group(2)),
                    "ns_per_op": float(match.group(3)),
                })

        output = f"Go Benchmarks ({len(benchmarks)} found):\n\n{stdout[:4000]}"
        return ToolResult(success=True, output=output, artifacts={"benchmarks": benchmarks})
