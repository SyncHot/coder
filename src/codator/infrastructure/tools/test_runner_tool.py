"""Test Runner Tool — structured test execution with parsed results.

Runs pytest/jest/go test/cargo test and parses output into structured
pass/fail results per test with timing and error details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class TestRunnerTool(Tool):
    """Run tests and parse results into structured pass/fail per test."""

    def __init__(self, cwd: str | Path = ""):
        self._cwd = str(cwd) if cwd else ""

    @property
    def name(self) -> str:
        return "test_runner"

    @property
    def description(self) -> str:
        return (
            "Run tests and get structured results. Auto-detects framework "
            "(pytest, jest, go test, cargo test). Returns pass/fail per test "
            "with timing, error messages, and summary."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "framework": {
                    "type": "string",
                    "enum": ["auto", "pytest", "jest", "go", "cargo", "unittest"],
                    "description": "Test framework (default: auto-detect).",
                },
                "path": {
                    "type": "string",
                    "description": "Test file/directory to run.",
                },
                "filter": {
                    "type": "string",
                    "description": "Test name filter/pattern (e.g., 'test_auth' for pytest -k).",
                },
                "verbose": {
                    "type": "boolean",
                    "description": "Show full output (default: false, only failures shown).",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120).",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs) -> ToolResult:
        framework = kwargs.get("framework", "auto")
        path = kwargs.get("path", "")
        test_filter = kwargs.get("filter", "")
        verbose = kwargs.get("verbose", False)
        timeout = kwargs.get("timeout", 120)

        if framework == "auto":
            framework = await self._detect_framework()

        cmd = self._build_command(framework, path, test_filter)
        if not cmd:
            return ToolResult(success=False, error=f"Cannot build command for framework: {framework}")

        start = time.time()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd or None,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = stdout.decode(errors="replace")
            elapsed = time.time() - start
        except asyncio.TimeoutError:
            return ToolResult(success=False, error=f"Tests timed out after {timeout}s")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to run tests: {e}")

        # Parse results based on framework
        results = self._parse_results(framework, output)
        results["elapsed_seconds"] = round(elapsed, 2)
        results["command"] = cmd
        results["exit_code"] = proc.returncode

        # Build readable output
        summary = self._format_summary(results, verbose, output)

        return ToolResult(
            success=proc.returncode == 0,
            output=summary,
            artifacts=results,
        )

    async def _detect_framework(self) -> str:
        cwd = Path(self._cwd) if self._cwd else Path.cwd()
        if (cwd / "pytest.ini").exists() or (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
            return "pytest"
        if (cwd / "package.json").exists():
            return "jest"
        if (cwd / "go.mod").exists():
            return "go"
        if (cwd / "Cargo.toml").exists():
            return "cargo"
        # Default
        return "pytest"

    def _build_command(self, framework: str, path: str, test_filter: str) -> str:
        if framework == "pytest":
            cmd = "python -m pytest --tb=short -q"
            if test_filter:
                cmd += f" -k '{test_filter}'"
            if path:
                cmd += f" {path}"
            return cmd
        elif framework == "jest":
            cmd = "npx jest --no-color"
            if test_filter:
                cmd += f" -t '{test_filter}'"
            if path:
                cmd += f" {path}"
            return cmd
        elif framework == "go":
            target = f"./{path}/..." if path else "./..."
            cmd = f"go test -v {target}"
            if test_filter:
                cmd += f" -run '{test_filter}'"
            return cmd
        elif framework == "cargo":
            cmd = "cargo test"
            if test_filter:
                cmd += f" {test_filter}"
            return cmd + " 2>&1"
        elif framework == "unittest":
            cmd = "python -m unittest"
            if path:
                cmd += f" {path}"
            return cmd + " 2>&1"
        return ""

    def _parse_results(self, framework: str, output: str) -> dict:
        results: dict[str, Any] = {"passed": [], "failed": [], "skipped": [], "errors": []}

        if framework == "pytest":
            self._parse_pytest(output, results)
        elif framework == "jest":
            self._parse_jest(output, results)
        elif framework == "go":
            self._parse_go(output, results)
        elif framework == "cargo":
            self._parse_cargo(output, results)
        else:
            self._parse_generic(output, results)

        results["total"] = len(results["passed"]) + len(results["failed"]) + len(results["skipped"])
        results["pass_count"] = len(results["passed"])
        results["fail_count"] = len(results["failed"])
        results["skip_count"] = len(results["skipped"])
        return results

    def _parse_pytest(self, output: str, results: dict):
        # Parse PASSED/FAILED lines
        for m in re.finditer(r"^(\S+::\S+)\s+(PASSED|FAILED|SKIPPED)", output, re.M):
            name, status = m.group(1), m.group(2).lower()
            if status == "passed":
                results["passed"].append(name)
            elif status == "failed":
                results["failed"].append(name)
            elif status == "skipped":
                results["skipped"].append(name)

        # Fallback: parse summary line "X passed, Y failed"
        if not results["passed"] and not results["failed"]:
            m = re.search(r"(\d+) passed", output)
            if m:
                results["passed"] = [f"test_{i}" for i in range(int(m.group(1)))]
            m = re.search(r"(\d+) failed", output)
            if m:
                results["failed"] = [f"failed_{i}" for i in range(int(m.group(1)))]
            m = re.search(r"(\d+) skipped", output)
            if m:
                results["skipped"] = [f"skipped_{i}" for i in range(int(m.group(1)))]

        # Extract failure details
        failures = re.findall(r"FAILED (.+?) - (.+?)$", output, re.M)
        for name, reason in failures:
            results["errors"].append({"test": name, "error": reason})

        # Also capture short traceback sections
        for block in re.findall(r"_{5,} (.+?) _{5,}\n(.+?)(?=\n_{5,}|\nFAILED|\Z)", output, re.S):
            results["errors"].append({"test": block[0], "traceback": block[1][:500]})

    def _parse_jest(self, output: str, results: dict):
        for m in re.finditer(r"^\s*(✓|✗|✕|PASS|FAIL)\s+(.+?)(?:\s+\((\d+)\s*ms\))?$", output, re.M):
            status, name = m.group(1), m.group(2)
            if status in ("✓", "PASS"):
                results["passed"].append(name)
            else:
                results["failed"].append(name)

    def _parse_go(self, output: str, results: dict):
        for m in re.finditer(r"^--- (PASS|FAIL|SKIP): (\S+)\s+\(([^)]+)\)", output, re.M):
            status, name, duration = m.group(1), m.group(2), m.group(3)
            if status == "PASS":
                results["passed"].append(name)
            elif status == "FAIL":
                results["failed"].append(name)
            else:
                results["skipped"].append(name)

    def _parse_cargo(self, output: str, results: dict):
        for m in re.finditer(r"^test (\S+)\s+\.\.\.\s+(ok|FAILED|ignored)", output, re.M):
            name, status = m.group(1), m.group(2)
            if status == "ok":
                results["passed"].append(name)
            elif status == "FAILED":
                results["failed"].append(name)
            else:
                results["skipped"].append(name)

    def _parse_generic(self, output: str, results: dict):
        # Try to find any pass/fail indicators
        for line in output.split("\n"):
            if re.search(r"\bpass(ed)?\b", line, re.I):
                results["passed"].append(line.strip()[:80])
            elif re.search(r"\bfail(ed)?\b", line, re.I):
                results["failed"].append(line.strip()[:80])

    def _format_summary(self, results: dict, verbose: bool, raw_output: str) -> str:
        parts = [
            f"Tests: {results['pass_count']} passed, {results['fail_count']} failed, "
            f"{results['skip_count']} skipped (total: {results['total']}) "
            f"in {results['elapsed_seconds']}s",
        ]

        if results["failed"]:
            parts.append("\n❌ Failed tests:")
            for t in results["failed"][:20]:
                parts.append(f"  • {t}")

        if results["errors"]:
            parts.append("\n📋 Error details:")
            for e in results["errors"][:5]:
                if "traceback" in e:
                    parts.append(f"  [{e['test']}]\n    {e['traceback'][:300]}")
                else:
                    parts.append(f"  [{e['test']}] {e.get('error', '')[:200]}")

        if verbose or (results["fail_count"] > 0 and not results["errors"]):
            parts.append(f"\n--- Raw output (last 2000 chars) ---\n{raw_output[-2000:]}")

        return "\n".join(parts)
