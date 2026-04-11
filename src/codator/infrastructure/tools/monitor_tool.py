"""System monitoring tool — CPU, RAM, disk, network, service health checks."""

import asyncio
import json
import os
import re
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


class MonitorTool:
    name = "monitor"
    description = (
        "Monitor system resources and service health: CPU/RAM/disk/network usage, "
        "process list, service uptime, HTTP health checks, and port status. "
        "Can also query Prometheus endpoints for metrics."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["system", "processes", "health_check", "ports", "network", "prometheus", "logs"],
                "description": "Monitoring action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for health_check or Prometheus endpoint",
            },
            "service_name": {
                "type": "string",
                "description": "Service name for systemd status or log filtering",
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top processes to show",
                "default": 15,
            },
            "port": {
                "type": "integer",
                "description": "Specific port to check",
            },
            "query": {
                "type": "string",
                "description": "Prometheus PromQL query",
            },
            "lines": {
                "type": "integer",
                "description": "Number of log lines to show",
                "default": 50,
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "system":
                return await self._system_stats()
            elif action == "processes":
                return await self._top_processes(kwargs.get("top_n", 15))
            elif action == "health_check":
                return await self._health_check(kwargs.get("url", ""))
            elif action == "ports":
                return await self._check_ports(kwargs.get("port"))
            elif action == "network":
                return await self._network_stats()
            elif action == "prometheus":
                return await self._query_prometheus(kwargs.get("url", ""), kwargs.get("query", ""))
            elif action == "logs":
                return await self._service_logs(kwargs.get("service_name", ""), kwargs.get("lines", 50))
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _run_command(self, cmd: list, timeout: int = 30) -> tuple:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return "", "Command timed out", -1
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _system_stats(self) -> ToolResult:
        """Get comprehensive system stats."""
        stats = {}

        # CPU usage
        try:
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            idle1 = int(parts[4])
            total1 = sum(int(x) for x in parts[1:])
            await asyncio.sleep(0.5)
            with open("/proc/stat") as f:
                line = f.readline()
            parts = line.split()
            idle2 = int(parts[4])
            total2 = sum(int(x) for x in parts[1:])
            cpu_pct = 100 * (1 - (idle2 - idle1) / max(total2 - total1, 1))
            stats["cpu_percent"] = round(cpu_pct, 1)
        except (IOError, ValueError):
            pass

        # Load average
        try:
            with open("/proc/loadavg") as f:
                load = f.read().split()[:3]
            stats["load_avg"] = f"{load[0]} {load[1]} {load[2]}"
        except IOError:
            pass

        # Memory
        try:
            with open("/proc/meminfo") as f:
                meminfo = {}
                for line in f:
                    parts = line.split()
                    meminfo[parts[0].rstrip(":")] = int(parts[1])
            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", 0)
            used = total - available
            stats["ram_total_gb"] = round(total / 1024 / 1024, 1)
            stats["ram_used_gb"] = round(used / 1024 / 1024, 1)
            stats["ram_percent"] = round(used / max(total, 1) * 100, 1)
            stats["swap_total_gb"] = round(meminfo.get("SwapTotal", 0) / 1024 / 1024, 1)
            stats["swap_used_gb"] = round((meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0)) / 1024 / 1024, 1)
        except (IOError, ValueError):
            pass

        # Disk
        import shutil
        total, used, free = shutil.disk_usage("/")
        stats["disk_total_gb"] = round(total / 1024**3, 1)
        stats["disk_used_gb"] = round(used / 1024**3, 1)
        stats["disk_free_gb"] = round(free / 1024**3, 1)
        stats["disk_percent"] = round(used / total * 100, 1)

        # Uptime
        try:
            with open("/proc/uptime") as f:
                uptime_sec = float(f.read().split()[0])
            days = int(uptime_sec // 86400)
            hours = int((uptime_sec % 86400) // 3600)
            stats["uptime"] = f"{days}d {hours}h"
        except (IOError, ValueError):
            pass

        lines = ["System Status:"]
        for k, v in stats.items():
            lines.append(f"  {k}: {v}")

        # Warning indicators
        warnings = []
        if stats.get("cpu_percent", 0) > 80:
            warnings.append("⚠️  CPU usage high")
        if stats.get("ram_percent", 0) > 85:
            warnings.append("⚠️  RAM usage high")
        if stats.get("disk_percent", 0) > 90:
            warnings.append("⚠️  Disk space low")

        if warnings:
            lines.append("\nWarnings:")
            lines.extend(f"  {w}" for w in warnings)

        return ToolResult(success=True, output="\n".join(lines), artifacts=stats)

    async def _top_processes(self, top_n: int) -> ToolResult:
        """Get top processes by CPU and memory."""
        stdout, _, rc = await self._run_command(
            ["ps", "aux", "--sort=-%cpu"]
        )
        if rc != 0:
            return ToolResult(success=False, error="Failed to get process list")

        lines = stdout.strip().split("\n")
        header = lines[0]
        processes = lines[1:top_n + 1]

        output = f"Top {top_n} processes (by CPU):\n{header}\n" + "\n".join(processes)
        return ToolResult(success=True, output=output)

    async def _health_check(self, url: str) -> ToolResult:
        """HTTP health check with response time."""
        if not url:
            return ToolResult(success=False, error="url is required for health_check")

        start = time.time()
        cmd = ["curl", "-s", "-o", "/dev/null", "-w",
               "%{http_code}|%{time_total}|%{size_download}|%{ssl_verify_result}",
               "-m", "10", url]
        stdout, stderr, rc = await self._run_command(cmd)
        elapsed = time.time() - start

        if rc != 0:
            return ToolResult(
                success=False,
                output=f"❌ {url} — unreachable ({elapsed:.2f}s)\n{stderr}",
                artifacts={"status": "down", "url": url},
            )

        parts = stdout.split("|")
        http_code = parts[0] if parts else "0"
        response_time = parts[1] if len(parts) > 1 else "?"
        size = parts[2] if len(parts) > 2 else "?"

        is_healthy = http_code.startswith("2") or http_code.startswith("3")
        icon = "✅" if is_healthy else "❌"

        output = f"{icon} {url}\n  Status: {http_code}\n  Response time: {response_time}s\n  Size: {size} bytes"
        return ToolResult(
            success=is_healthy,
            output=output,
            artifacts={"status_code": int(http_code), "response_time": float(response_time), "url": url},
        )

    async def _check_ports(self, port: int = None) -> ToolResult:
        """Check listening ports."""
        cmd = ["ss", "-tlnp"]
        stdout, _, rc = await self._run_command(cmd)

        if rc != 0:
            # Fallback to netstat
            stdout, _, rc = await self._run_command(["netstat", "-tlnp"])

        if port:
            lines = [l for l in stdout.split("\n") if f":{port}" in l]
            if lines:
                return ToolResult(success=True, output=f"Port {port} is OPEN:\n" + "\n".join(lines))
            else:
                return ToolResult(success=True, output=f"Port {port} is NOT listening")

        return ToolResult(success=True, output=f"Listening ports:\n{stdout[:3000]}")

    async def _network_stats(self) -> ToolResult:
        """Network interface statistics."""
        stdout, _, rc = await self._run_command(["ip", "-s", "link"])
        if rc != 0:
            stdout, _, _ = await self._run_command(["ifconfig"])

        # Also get connections count
        stdout2, _, _ = await self._run_command(["ss", "-s"])

        output = f"Network Interfaces:\n{stdout[:2000]}\n\nConnection Summary:\n{stdout2[:1000]}"
        return ToolResult(success=True, output=output)

    async def _query_prometheus(self, url: str, query: str) -> ToolResult:
        """Query a Prometheus endpoint."""
        if not url:
            url = "http://localhost:9090"
        if not query:
            return ToolResult(success=False, error="query (PromQL) is required")

        api_url = f"{url}/api/v1/query"
        cmd = ["curl", "-s", f"{api_url}?query={query}"]
        stdout, stderr, rc = await self._run_command(cmd)

        if rc != 0:
            return ToolResult(success=False, error=f"Prometheus query failed: {stderr}")

        try:
            data = json.loads(stdout)
            if data.get("status") != "success":
                return ToolResult(success=False, error=f"Query error: {data.get('error', 'unknown')}")

            results = data.get("data", {}).get("result", [])
            lines = [f"Query: {query}\nResults ({len(results)}):\n"]
            for r in results[:20]:
                metric = r.get("metric", {})
                value = r.get("value", [None, "?"])
                label = metric.get("__name__", "") or json.dumps(metric)
                lines.append(f"  {label}: {value[1]}")

            return ToolResult(success=True, output="\n".join(lines), artifacts={"result_count": len(results)})
        except json.JSONDecodeError:
            return ToolResult(success=False, error=f"Invalid response from Prometheus: {stdout[:200]}")

    async def _service_logs(self, service_name: str, lines: int) -> ToolResult:
        """Get recent logs for a systemd service."""
        if not service_name:
            return ToolResult(success=False, error="service_name is required for logs action")

        cmd = ["journalctl", "-u", service_name, "-n", str(lines), "--no-pager"]
        stdout, stderr, rc = await self._run_command(cmd)

        if rc != 0:
            # Try docker logs
            cmd2 = ["docker", "logs", "--tail", str(lines), service_name]
            stdout, stderr, rc = await self._run_command(cmd2)

        if rc != 0:
            return ToolResult(success=False, error=f"Cannot get logs for '{service_name}': {stderr}")

        return ToolResult(success=True, output=stdout[:5000])
