"""Log analyzer tool — parse logs, find patterns, correlate events."""

import asyncio
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class LogAnalyzerTool:
    name = "log_analyzer"
    description = (
        "Analyze log files: parse structured/unstructured logs, find error patterns, "
        "extract timestamps, count occurrences, correlate events. "
        "Supports journalctl, Docker logs, nginx/apache, and custom log formats."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["errors", "pattern", "stats", "tail", "between", "correlate"],
                "description": "Analysis action to perform",
            },
            "source": {
                "type": "string",
                "description": "Log file path, service name (for journalctl), or container name (for docker logs)",
            },
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for (for pattern action)",
            },
            "since": {
                "type": "string",
                "description": "Start time filter (ISO format or relative like '1h', '30m')",
            },
            "until": {
                "type": "string",
                "description": "End time filter",
            },
            "lines": {
                "type": "integer",
                "description": "Number of lines to return",
                "default": 50,
            },
            "severity": {
                "type": "string",
                "enum": ["error", "warning", "info", "debug", "all"],
                "description": "Filter by log severity",
                "default": "all",
            },
            "group_by": {
                "type": "string",
                "enum": ["message", "source", "hour", "minute"],
                "description": "Group results by field (for stats action)",
            },
        },
        "required": ["action", "source"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        source = kwargs["source"]
        try:
            if action == "errors":
                return await self._find_errors(source, kwargs)
            elif action == "pattern":
                return await self._search_pattern(source, kwargs)
            elif action == "stats":
                return await self._statistics(source, kwargs)
            elif action == "tail":
                return await self._tail(source, kwargs)
            elif action == "between":
                return await self._between(source, kwargs)
            elif action == "correlate":
                return await self._correlate(source, kwargs)
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _run_command(self, cmd, cwd=None, timeout=30):
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return "", "Timed out", -1
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _get_log_content(self, source: str, lines: int = 1000, since: str = "") -> str:
        """Get log content from various sources."""
        if os.path.isfile(source):
            with open(source, "r", errors="ignore") as f:
                all_lines = f.readlines()
            return "".join(all_lines[-lines:])

        # Try journalctl
        cmd = ["journalctl", "-u", source, "-n", str(lines), "--no-pager"]
        if since:
            cmd.extend(["--since", since])
        stdout, _, rc = await self._run_command(cmd)
        if rc == 0 and stdout.strip():
            return stdout

        # Try docker logs
        cmd = ["docker", "logs", "--tail", str(lines), source]
        stdout, _, rc = await self._run_command(cmd)
        if rc == 0:
            return stdout

        return ""

    async def _find_errors(self, source: str, kwargs: dict) -> ToolResult:
        """Find error-level messages in logs."""
        lines_count = kwargs.get("lines", 100)
        content = await self._get_log_content(source, lines_count * 10, kwargs.get("since", ""))

        if not content:
            return ToolResult(success=False, error="Could not read logs from: " + source)

        error_patterns = [
            r'(?i)\b(ERROR|FATAL|CRITICAL|EXCEPTION|PANIC)\b',
            r'(?i)Traceback \(most recent call last\)',
            r'(?i)(?:segmentation fault|core dump|out of memory)',
            r'(?i)(?:failed|failure)\b.*(?:exit|status|code)',
        ]

        errors = []
        for line in content.split("\n"):
            for pattern in error_patterns:
                if re.search(pattern, line):
                    errors.append(line.strip())
                    break

        unique_errors = Counter(errors)
        output_lines = ["Errors found: " + str(len(errors)) + "\n"]
        output_lines.append("Unique error patterns (" + str(len(unique_errors)) + "):\n")
        for error, count in unique_errors.most_common(30):
            output_lines.append("  [" + str(count) + "x] " + error[:200])

        return ToolResult(
            success=True,
            output="\n".join(output_lines)[:5000],
            artifacts={"total_errors": len(errors), "unique_patterns": len(unique_errors)},
        )

    async def _search_pattern(self, source: str, kwargs: dict) -> ToolResult:
        """Search for a specific pattern in logs."""
        pattern = kwargs.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, error="pattern is required")

        lines_count = kwargs.get("lines", 50)
        content = await self._get_log_content(source, 5000, kwargs.get("since", ""))

        if not content:
            return ToolResult(success=False, error="Could not read logs from: " + source)

        matches = []
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return ToolResult(success=False, error="Invalid regex: " + str(e))

        for line in content.split("\n"):
            if regex.search(line):
                matches.append(line.strip())

        output = "Pattern: " + pattern + "\nMatches: " + str(len(matches)) + "\n\n"
        output += "\n".join(matches[-lines_count:])

        return ToolResult(success=True, output=output[:5000], artifacts={"matches": len(matches)})

    async def _statistics(self, source: str, kwargs: dict) -> ToolResult:
        """Generate log statistics."""
        group_by = kwargs.get("group_by", "message")
        content = await self._get_log_content(source, 5000, kwargs.get("since", ""))

        if not content:
            return ToolResult(success=False, error="Could not read logs from: " + source)

        lines = content.strip().split("\n")
        total = len(lines)

        # Count by severity
        severity_counts = Counter()
        for line in lines:
            if re.search(r'(?i)\bERROR\b', line):
                severity_counts["ERROR"] += 1
            elif re.search(r'(?i)\bWARN(?:ING)?\b', line):
                severity_counts["WARNING"] += 1
            elif re.search(r'(?i)\bINFO\b', line):
                severity_counts["INFO"] += 1
            elif re.search(r'(?i)\bDEBUG\b', line):
                severity_counts["DEBUG"] += 1
            else:
                severity_counts["OTHER"] += 1

        output_lines = [
            "Log Statistics:",
            "  Total lines: " + str(total),
            "\nBy severity:",
        ]
        for sev, count in severity_counts.most_common():
            pct = count / max(total, 1) * 100
            output_lines.append("  " + sev + ": " + str(count) + " (" + f"{pct:.1f}" + "%)")

        # Group by message pattern (normalize numbers/hashes)
        if group_by == "message":
            normalized = Counter()
            for line in lines:
                # Remove timestamps and normalize
                norm = re.sub(r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[.\d]*\S*', '<TIME>', line)
                norm = re.sub(r'\b\d+\b', '<N>', norm)
                norm = re.sub(r'\b[0-9a-f]{8,}\b', '<HEX>', norm)
                normalized[norm[:100]] += 1

            output_lines.append("\nTop message patterns:")
            for msg, count in normalized.most_common(15):
                output_lines.append("  [" + str(count) + "x] " + msg)

        return ToolResult(success=True, output="\n".join(output_lines)[:5000], artifacts={"total_lines": total, "severity": dict(severity_counts)})

    async def _tail(self, source: str, kwargs: dict) -> ToolResult:
        """Get last N lines of logs."""
        lines_count = kwargs.get("lines", 50)
        severity = kwargs.get("severity", "all")
        content = await self._get_log_content(source, lines_count * 5, kwargs.get("since", ""))

        if not content:
            return ToolResult(success=False, error="Could not read logs from: " + source)

        log_lines = content.strip().split("\n")

        if severity != "all":
            pattern = r'(?i)\b' + severity + r'\b'
            log_lines = [l for l in log_lines if re.search(pattern, l)]

        result = "\n".join(log_lines[-lines_count:])
        return ToolResult(success=True, output=result[:5000], artifacts={"lines_shown": min(lines_count, len(log_lines))})

    async def _between(self, source: str, kwargs: dict) -> ToolResult:
        """Get logs between two timestamps."""
        since = kwargs.get("since", "")
        until = kwargs.get("until", "")

        if not since:
            return ToolResult(success=False, error="since is required for between action")

        if os.path.isfile(source):
            content = await self._get_log_content(source, 10000)
            # Try to parse timestamps and filter
            lines = content.split("\n")
            filtered = []
            ts_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})')
            for line in lines:
                match = ts_pattern.search(line)
                if match:
                    ts_str = match.group(1).replace("T", " ")
                    if ts_str >= since and (not until or ts_str <= until):
                        filtered.append(line)
                elif filtered:  # Multi-line log entry
                    filtered.append(line)

            return ToolResult(success=True, output="\n".join(filtered[-100:])[:5000], artifacts={"lines": len(filtered)})

        # Use journalctl for service
        cmd = ["journalctl", "-u", source, "--since", since, "--no-pager"]
        if until:
            cmd.extend(["--until", until])
        stdout, _, rc = await self._run_command(cmd)

        if rc != 0:
            return ToolResult(success=False, error="Could not get logs between timestamps")

        return ToolResult(success=True, output=stdout[-5000:], artifacts={"lines": stdout.count("\n")})

    async def _correlate(self, source: str, kwargs: dict) -> ToolResult:
        """Correlate events across log entries (find related entries by request ID, timestamp proximity)."""
        pattern = kwargs.get("pattern", "")
        if not pattern:
            return ToolResult(success=False, error="pattern required for correlate (e.g., request ID)")

        content = await self._get_log_content(source, 10000, kwargs.get("since", ""))
        if not content:
            return ToolResult(success=False, error="Could not read logs")

        lines = content.split("\n")
        correlated = [l for l in lines if pattern in l]

        if not correlated:
            return ToolResult(success=True, output="No entries matching: " + pattern)

        output = "Correlated entries for '" + pattern + "' (" + str(len(correlated)) + " entries):\n\n"
        output += "\n".join(correlated[-50:])

        return ToolResult(success=True, output=output[:5000], artifacts={"correlated_entries": len(correlated)})
