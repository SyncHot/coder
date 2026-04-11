"""Lint Tool — run linters and parse output into structured issues.

Supports Python (ruff/flake8/pylint), JavaScript/TypeScript (eslint),
Go (golangci-lint), and generic regex-based output parsing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class LintTool(Tool):
    """Run linters and return structured issues per file/line."""

    def __init__(self, cwd: str | Path = ""):
        self._cwd = str(cwd) if cwd else ""

    @property
    def name(self) -> str:
        return "lint"

    @property
    def description(self) -> str:
        return (
            "Run code linters and get structured results. Auto-detects linter "
            "(ruff/flake8/eslint/golangci-lint). Returns issues grouped by file "
            "with line numbers, severity, and rule codes."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "linter": {
                    "type": "string",
                    "enum": ["auto", "ruff", "flake8", "pylint", "eslint", "golangci-lint", "mypy"],
                    "description": "Linter to use (default: auto-detect).",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to lint.",
                },
                "fix": {
                    "type": "boolean",
                    "description": "Auto-fix issues where possible (default: false).",
                },
                "config": {
                    "type": "string",
                    "description": "Config file path override.",
                },
            },
            "required": [],
        }

    async def execute(self, **kwargs) -> ToolResult:
        linter = kwargs.get("linter", "auto")
        path = kwargs.get("path", ".")
        fix = kwargs.get("fix", False)
        config = kwargs.get("config", "")

        if linter == "auto":
            linter = await self._detect_linter()

        cmd = self._build_command(linter, path, fix, config)
        if not cmd:
            return ToolResult(success=False, error=f"Cannot build command for: {linter}")

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._cwd or None,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            output = stdout.decode(errors="replace")
        except asyncio.TimeoutError:
            return ToolResult(success=False, error="Linter timed out after 120s")
        except Exception as e:
            return ToolResult(success=False, error=f"Failed to run linter: {e}")

        # Parse issues
        issues = self._parse_output(linter, output)
        summary = self._format_summary(linter, issues, output, proc.returncode == 0)

        return ToolResult(
            success=proc.returncode == 0 or len(issues) == 0,
            output=summary,
            artifacts={"issues": issues, "issue_count": len(issues), "linter": linter},
        )

    async def _detect_linter(self) -> str:
        cwd = Path(self._cwd) if self._cwd else Path.cwd()

        # Check for Python linters
        if (cwd / "pyproject.toml").exists() or (cwd / "setup.py").exists():
            # Prefer ruff
            try:
                proc = await asyncio.create_subprocess_exec(
                    "ruff", "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
                if proc.returncode == 0:
                    return "ruff"
            except FileNotFoundError:
                pass
            return "flake8"

        # JavaScript/TypeScript
        if (cwd / "package.json").exists():
            return "eslint"

        # Go
        if (cwd / "go.mod").exists():
            return "golangci-lint"

        return "ruff"  # default

    def _build_command(self, linter: str, path: str, fix: bool, config: str) -> str:
        if linter == "ruff":
            cmd = f"ruff check --output-format=json"
            if fix:
                cmd += " --fix"
            if config:
                cmd += f" --config={config}"
            cmd += f" {path}"
            return cmd

        elif linter == "flake8":
            cmd = f"flake8 --format=json"
            if config:
                cmd += f" --config={config}"
            cmd += f" {path}"
            return cmd

        elif linter == "pylint":
            cmd = f"pylint --output-format=json"
            if config:
                cmd += f" --rcfile={config}"
            cmd += f" {path}"
            return cmd

        elif linter == "mypy":
            cmd = f"mypy --no-color-output"
            if config:
                cmd += f" --config-file={config}"
            cmd += f" {path}"
            return cmd

        elif linter == "eslint":
            cmd = f"npx eslint --format=json"
            if fix:
                cmd += " --fix"
            if config:
                cmd += f" --config={config}"
            cmd += f" {path}"
            return cmd

        elif linter == "golangci-lint":
            cmd = f"golangci-lint run --out-format=json"
            if fix:
                cmd += " --fix"
            if config:
                cmd += f" --config={config}"
            if path and path != ".":
                cmd += f" {path}/..."
            return cmd

        return ""

    def _parse_output(self, linter: str, output: str) -> list[dict]:
        issues = []

        if linter == "ruff":
            issues = self._parse_ruff(output)
        elif linter in ("flake8", "pylint"):
            issues = self._parse_json_linter(output)
        elif linter == "eslint":
            issues = self._parse_eslint(output)
        elif linter == "golangci-lint":
            issues = self._parse_golangci(output)
        elif linter == "mypy":
            issues = self._parse_mypy(output)
        else:
            issues = self._parse_generic(output)

        return issues

    def _parse_ruff(self, output: str) -> list[dict]:
        try:
            data = json.loads(output)
            issues = []
            for item in data:
                issues.append({
                    "file": item.get("filename", ""),
                    "line": item.get("location", {}).get("row", 0),
                    "column": item.get("location", {}).get("column", 0),
                    "code": item.get("code", ""),
                    "message": item.get("message", ""),
                    "severity": "error" if item.get("code", "").startswith("E") else "warning",
                })
            return issues
        except json.JSONDecodeError:
            return self._parse_generic(output)

    def _parse_json_linter(self, output: str) -> list[dict]:
        try:
            data = json.loads(output)
            issues = []
            if isinstance(data, list):
                for item in data:
                    issues.append({
                        "file": item.get("path", item.get("filename", "")),
                        "line": item.get("line", item.get("row", 0)),
                        "column": item.get("column", item.get("col", 0)),
                        "code": item.get("code", item.get("symbol", "")),
                        "message": item.get("message", item.get("msg", "")),
                        "severity": item.get("type", "warning"),
                    })
            return issues
        except json.JSONDecodeError:
            return self._parse_generic(output)

    def _parse_eslint(self, output: str) -> list[dict]:
        try:
            data = json.loads(output)
            issues = []
            for file_result in data:
                filepath = file_result.get("filePath", "")
                for msg in file_result.get("messages", []):
                    issues.append({
                        "file": filepath,
                        "line": msg.get("line", 0),
                        "column": msg.get("column", 0),
                        "code": msg.get("ruleId", ""),
                        "message": msg.get("message", ""),
                        "severity": "error" if msg.get("severity", 1) == 2 else "warning",
                    })
            return issues
        except json.JSONDecodeError:
            return self._parse_generic(output)

    def _parse_golangci(self, output: str) -> list[dict]:
        try:
            data = json.loads(output)
            issues = []
            for item in data.get("Issues", []):
                pos = item.get("Pos", {})
                issues.append({
                    "file": pos.get("Filename", ""),
                    "line": pos.get("Line", 0),
                    "column": pos.get("Column", 0),
                    "code": item.get("FromLinter", ""),
                    "message": item.get("Text", ""),
                    "severity": item.get("Severity", "warning"),
                })
            return issues
        except json.JSONDecodeError:
            return self._parse_generic(output)

    def _parse_mypy(self, output: str) -> list[dict]:
        issues = []
        for line in output.split("\n"):
            m = re.match(r"^(.+?):(\d+):(\d+):\s*(error|warning|note):\s*(.+?)(?:\s+\[(.+?)\])?$", line)
            if m:
                issues.append({
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "column": int(m.group(3)),
                    "code": m.group(6) or "",
                    "message": m.group(5),
                    "severity": m.group(4),
                })
        return issues

    def _parse_generic(self, output: str) -> list[dict]:
        """Parse common linter output format: file:line:col: message"""
        issues = []
        for line in output.split("\n"):
            m = re.match(r"^(.+?):(\d+):(\d+):\s*(.+)$", line)
            if m:
                issues.append({
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "column": int(m.group(3)),
                    "code": "",
                    "message": m.group(4),
                    "severity": "warning",
                })
        return issues

    def _format_summary(self, linter: str, issues: list[dict], raw: str, clean: bool) -> str:
        if clean and not issues:
            return f"{linter}: No issues found. Code is clean!"

        # Group by file
        by_file: dict[str, list[dict]] = {}
        for issue in issues:
            f = issue.get("file", "unknown")
            by_file.setdefault(f, []).append(issue)

        errors = sum(1 for i in issues if i.get("severity") == "error")
        warnings = len(issues) - errors

        parts = [f"{linter}: {len(issues)} issues ({errors} errors, {warnings} warnings) in {len(by_file)} files\n"]

        for filepath, file_issues in sorted(by_file.items())[:20]:
            parts.append(f"  {filepath}:")
            for i in file_issues[:10]:
                severity_icon = "E" if i["severity"] == "error" else "W"
                code = f"[{i['code']}] " if i["code"] else ""
                parts.append(f"    L{i['line']:4d}:{i['column']:<3d} {severity_icon} {code}{i['message']}")
            if len(file_issues) > 10:
                remaining = len(file_issues) - 10
                parts.append(f"    ... ({remaining} more)")

        if len(by_file) > 20:
            remaining = len(by_file) - 20
            parts.append(f"\n  ... ({remaining} more files)")

        return "\n".join(parts)
