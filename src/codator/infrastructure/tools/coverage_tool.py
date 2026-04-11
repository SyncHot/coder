"""Code coverage tool — run coverage, parse reports, identify untested code."""

import asyncio
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class CoverageTool:
    name = "coverage"
    description = (
        "Run code coverage analysis. Supports pytest-cov, istanbul/nyc, go cover, "
        "cargo-tarpaulin. Parses LCOV/Cobertura/JSON reports and identifies untested code paths."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "report", "uncovered", "diff"],
                "description": "Action: run coverage, show report, list uncovered lines, or show diff coverage",
            },
            "path": {
                "type": "string",
                "description": "Project path or coverage report file path",
            },
            "framework": {
                "type": "string",
                "enum": ["pytest", "jest", "go", "cargo", "auto"],
                "description": "Test framework to use (auto-detected if not specified)",
                "default": "auto",
            },
            "threshold": {
                "type": "number",
                "description": "Minimum coverage percentage threshold (for pass/fail)",
                "default": 0,
            },
            "file_filter": {
                "type": "string",
                "description": "Filter results to specific file pattern (glob)",
            },
            "report_format": {
                "type": "string",
                "enum": ["summary", "detailed", "json"],
                "description": "Output format",
                "default": "summary",
            },
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs["path"]
        framework = kwargs.get("framework", "auto")

        if framework == "auto":
            framework = self._detect_framework(path)

        try:
            if action == "run":
                return await self._run_coverage(path, framework, kwargs)
            elif action == "report":
                return await self._parse_report(path, kwargs)
            elif action == "uncovered":
                return await self._find_uncovered(path, framework, kwargs)
            elif action == "diff":
                return await self._diff_coverage(path, framework, kwargs)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _detect_framework(self, path: str) -> str:
        if os.path.isdir(path):
            files = os.listdir(path)
            if "pytest.ini" in files or "pyproject.toml" in files or "setup.py" in files:
                return "pytest"
            if "package.json" in files:
                return "jest"
            if "go.mod" in files:
                return "go"
            if "Cargo.toml" in files:
                return "cargo"
        elif path.endswith(".py"):
            return "pytest"
        return "pytest"

    async def _run_command(self, cmd: list, cwd: str = None) -> tuple:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        stdout, stderr = await proc.communicate()
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _run_coverage(self, path: str, framework: str, kwargs: dict) -> ToolResult:
        threshold = kwargs.get("threshold", 0)
        cwd = path if os.path.isdir(path) else os.path.dirname(path)

        if framework == "pytest":
            cmd = ["python", "-m", "pytest", "--cov=.", "--cov-report=term-missing",
                   "--cov-report=json:coverage.json", "-q", "--tb=no"]
            if threshold:
                cmd.append(f"--cov-fail-under={threshold}")
        elif framework == "jest":
            cmd = ["npx", "jest", "--coverage", "--coverageReporters=json-summary", "--silent"]
        elif framework == "go":
            cmd = ["go", "test", "-coverprofile=coverage.out", "-covermode=atomic", "./..."]
        elif framework == "cargo":
            cmd = ["cargo", "tarpaulin", "--out", "Json", "--output-dir", "."]
        else:
            return ToolResult(success=False, error=f"Unsupported framework: {framework}")

        stdout, stderr, rc = await self._run_command(cmd, cwd)

        # Parse coverage percentage from output
        coverage_pct = self._extract_coverage_pct(stdout + stderr, framework)

        result_output = f"Coverage: {coverage_pct:.1f}%\n\n{stdout}"
        if threshold and coverage_pct < threshold:
            return ToolResult(
                success=False,
                output=result_output,
                error=f"Coverage {coverage_pct:.1f}% below threshold {threshold}%",
                artifacts={"coverage_percent": coverage_pct, "threshold": threshold},
            )

        return ToolResult(
            success=True,
            output=result_output[:5000],
            artifacts={"coverage_percent": coverage_pct, "framework": framework},
        )

    def _extract_coverage_pct(self, output: str, framework: str) -> float:
        if framework == "pytest":
            match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", output)
            if match:
                return float(match.group(1))
        elif framework == "go":
            match = re.search(r"coverage:\s+([\d.]+)%", output)
            if match:
                return float(match.group(1))
        # Generic percentage pattern
        match = re.search(r"(\d+\.?\d*)%", output)
        return float(match.group(1)) if match else 0.0

    async def _parse_report(self, path: str, kwargs: dict) -> ToolResult:
        """Parse an existing coverage report file."""
        report_format = kwargs.get("report_format", "summary")
        file_filter = kwargs.get("file_filter", "")

        if not os.path.isfile(path):
            # Try common report locations
            candidates = [
                os.path.join(path, "coverage.json"),
                os.path.join(path, "coverage.xml"),
                os.path.join(path, "coverage/lcov.info"),
                os.path.join(path, "htmlcov/status.json"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    path = c
                    break
            else:
                return ToolResult(success=False, error=f"No coverage report found at {path}")

        if path.endswith(".json"):
            return self._parse_json_report(path, file_filter, report_format)
        elif path.endswith(".xml"):
            return self._parse_cobertura(path, file_filter, report_format)
        elif "lcov" in path:
            return self._parse_lcov(path, file_filter, report_format)
        else:
            return ToolResult(success=False, error=f"Unknown report format: {path}")

    def _parse_json_report(self, path: str, file_filter: str, report_format: str) -> ToolResult:
        with open(path) as f:
            data = json.load(f)

        # pytest-cov JSON format
        if "totals" in data:
            totals = data["totals"]
            pct = totals.get("percent_covered", 0)
            files_data = data.get("files", {})

            if file_filter:
                files_data = {k: v for k, v in files_data.items() if file_filter in k}

            if report_format == "json":
                return ToolResult(success=True, output=json.dumps(data, indent=2)[:5000])

            lines = [f"Total Coverage: {pct:.1f}%\n"]
            for fname, fdata in sorted(files_data.items(), key=lambda x: x[1].get("summary", {}).get("percent_covered", 100)):
                fpct = fdata.get("summary", {}).get("percent_covered", 0)
                lines.append(f"  {fname}: {fpct:.1f}%")
                if report_format == "detailed":
                    missing = fdata.get("missing_lines", [])
                    if missing:
                        lines.append(f"    Missing lines: {missing[:20]}")

            return ToolResult(success=True, output="\n".join(lines)[:5000], artifacts={"total_percent": pct})
        return ToolResult(success=True, output=json.dumps(data, indent=2)[:3000])

    def _parse_cobertura(self, path: str, file_filter: str, report_format: str) -> ToolResult:
        tree = ET.parse(path)
        root = tree.getroot()
        line_rate = float(root.get("line-rate", 0)) * 100
        branch_rate = float(root.get("branch-rate", 0)) * 100

        lines = [f"Line Coverage: {line_rate:.1f}%", f"Branch Coverage: {branch_rate:.1f}%\n"]

        for pkg in root.findall(".//package"):
            for cls in pkg.findall(".//class"):
                filename = cls.get("filename", "")
                if file_filter and file_filter not in filename:
                    continue
                cls_rate = float(cls.get("line-rate", 0)) * 100
                lines.append(f"  {filename}: {cls_rate:.1f}%")

        return ToolResult(
            success=True,
            output="\n".join(lines)[:5000],
            artifacts={"line_coverage": line_rate, "branch_coverage": branch_rate},
        )

    def _parse_lcov(self, path: str, file_filter: str, report_format: str) -> ToolResult:
        with open(path) as f:
            content = f.read()

        files = {}
        current_file = None
        total_hit = 0
        total_found = 0

        for line in content.split("\n"):
            if line.startswith("SF:"):
                current_file = line[3:]
            elif line.startswith("LF:"):
                found = int(line[3:])
                total_found += found
                if current_file:
                    files.setdefault(current_file, {})["found"] = found
            elif line.startswith("LH:"):
                hit = int(line[3:])
                total_hit += hit
                if current_file:
                    files[current_file]["hit"] = hit

        pct = (total_hit / total_found * 100) if total_found else 0
        lines = [f"Total: {pct:.1f}% ({total_hit}/{total_found} lines)\n"]

        for fname, data in sorted(files.items()):
            if file_filter and file_filter not in fname:
                continue
            fpct = (data.get("hit", 0) / data.get("found", 1)) * 100
            lines.append(f"  {fname}: {fpct:.1f}%")

        return ToolResult(success=True, output="\n".join(lines)[:5000], artifacts={"total_percent": pct})

    async def _find_uncovered(self, path: str, framework: str, kwargs: dict) -> ToolResult:
        """Find uncovered lines in the project."""
        file_filter = kwargs.get("file_filter", "")
        cwd = path if os.path.isdir(path) else os.path.dirname(path)

        if framework == "pytest":
            cmd = ["python", "-m", "pytest", "--cov=.", "--cov-report=term-missing", "-q", "--tb=no"]
            stdout, stderr, rc = await self._run_command(cmd, cwd)

            # Parse missing lines from output
            lines = []
            for line in stdout.split("\n"):
                if "%" in line and "TOTAL" not in line:
                    if file_filter and file_filter not in line:
                        continue
                    parts = line.split()
                    if len(parts) >= 4:
                        filename = parts[0]
                        missing = parts[-1] if not parts[-1].endswith("%") else ""
                        if missing:
                            lines.append(f"{filename}: missing lines {missing}")

            return ToolResult(
                success=True,
                output="\n".join(lines) if lines else "All code is covered!",
                artifacts={"uncovered_files": len(lines)},
            )
        elif framework == "go":
            cmd = ["go", "test", "-coverprofile=coverage.out", "./..."]
            await self._run_command(cmd, cwd)
            cmd2 = ["go", "tool", "cover", "-func=coverage.out"]
            stdout, _, _ = await self._run_command(cmd2, cwd)
            # Filter uncovered
            uncovered = [l for l in stdout.split("\n") if "0.0%" in l]
            if file_filter:
                uncovered = [l for l in uncovered if file_filter in l]
            return ToolResult(
                success=True,
                output="\n".join(uncovered[:50]) if uncovered else "All functions covered!",
                artifacts={"uncovered_functions": len(uncovered)},
            )

        return ToolResult(success=False, error=f"Uncovered analysis not implemented for {framework}")

    async def _diff_coverage(self, path: str, framework: str, kwargs: dict) -> ToolResult:
        """Show coverage for recently changed lines (git diff)."""
        cwd = path if os.path.isdir(path) else os.path.dirname(path)

        # Get changed lines from git
        cmd = ["git", "diff", "--unified=0", "HEAD~1"]
        stdout, _, _ = await self._run_command(cmd, cwd)

        changed_files = {}
        current_file = None
        for line in stdout.split("\n"):
            if line.startswith("+++ b/"):
                current_file = line[6:]
                changed_files[current_file] = []
            elif line.startswith("@@") and current_file:
                match = re.search(r"\+(\d+)(?:,(\d+))?", line)
                if match:
                    start = int(match.group(1))
                    count = int(match.group(2) or 1)
                    changed_files[current_file].extend(range(start, start + count))

        if not changed_files:
            return ToolResult(success=True, output="No changes detected in git diff")

        # Run coverage
        if framework == "pytest":
            cmd = ["python", "-m", "pytest", "--cov=.", "--cov-report=json:coverage.json", "-q", "--tb=no"]
            await self._run_command(cmd, cwd)

            report_path = os.path.join(cwd, "coverage.json")
            if os.path.isfile(report_path):
                with open(report_path) as f:
                    data = json.load(f)

                lines = ["Diff Coverage:\n"]
                total_changed = 0
                total_covered = 0
                for fname, changed_lines in changed_files.items():
                    fdata = data.get("files", {}).get(fname, {})
                    missing = set(fdata.get("missing_lines", []))
                    covered = [l for l in changed_lines if l not in missing]
                    total_changed += len(changed_lines)
                    total_covered += len(covered)
                    pct = (len(covered) / len(changed_lines) * 100) if changed_lines else 100
                    lines.append(f"  {fname}: {pct:.0f}% ({len(covered)}/{len(changed_lines)} new lines covered)")

                diff_pct = (total_covered / total_changed * 100) if total_changed else 100
                lines.insert(1, f"Overall diff coverage: {diff_pct:.1f}%\n")

                return ToolResult(success=True, output="\n".join(lines), artifacts={"diff_coverage_percent": diff_pct})

        return ToolResult(success=False, error="Could not calculate diff coverage")
