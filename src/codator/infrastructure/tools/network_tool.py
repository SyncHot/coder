"""Network diagnostics tool — ports, DNS, TCP probes, connectivity checks."""

import asyncio
import os
import re
import socket
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


class NetworkTool:
    name = "network"
    description = (
        "Network diagnostics: check port connectivity, DNS resolution, TCP probes, "
        "ping hosts, check SSL certificates, and trace routes. "
        "Useful for debugging deployment and connectivity issues."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["port_check", "dns", "ping", "ssl_check", "trace", "scan_ports", "connectivity"],
                "description": "Network action to perform",
            },
            "host": {
                "type": "string",
                "description": "Target hostname or IP address",
            },
            "port": {
                "type": "integer",
                "description": "Port number for port_check",
            },
            "ports": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "List of ports to scan (for scan_ports)",
            },
            "timeout": {
                "type": "number",
                "description": "Connection timeout in seconds",
                "default": 5,
            },
            "count": {
                "type": "integer",
                "description": "Number of pings to send",
                "default": 4,
            },
        },
        "required": ["action", "host"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        host = kwargs["host"]
        try:
            if action == "port_check":
                return await self._port_check(host, kwargs)
            elif action == "dns":
                return await self._dns_lookup(host)
            elif action == "ping":
                return await self._ping(host, kwargs)
            elif action == "ssl_check":
                return await self._ssl_check(host, kwargs)
            elif action == "trace":
                return await self._traceroute(host)
            elif action == "scan_ports":
                return await self._scan_ports(host, kwargs)
            elif action == "connectivity":
                return await self._connectivity_check(host, kwargs)
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _run_command(self, cmd, timeout=30):
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return "", "Timed out", -1
        return stdout.decode(), stderr.decode(), proc.returncode

    async def _port_check(self, host: str, kwargs: dict) -> ToolResult:
        """Check if a specific port is open."""
        port = kwargs.get("port", 80)
        timeout = kwargs.get("timeout", 5)

        start = time.time()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout
            )
            elapsed = time.time() - start
            writer.close()
            await writer.wait_closed()
            return ToolResult(
                success=True,
                output="OPEN " + host + ":" + str(port) + " (connected in " + f"{elapsed*1000:.1f}" + "ms)",
                artifacts={"status": "open", "latency_ms": round(elapsed * 1000, 1)},
            )
        except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
            elapsed = time.time() - start
            status = "refused" if "Refused" in str(e) else "timeout"
            return ToolResult(
                success=False,
                output="CLOSED " + host + ":" + str(port) + " (" + status + " after " + f"{elapsed*1000:.0f}" + "ms)",
                artifacts={"status": status},
            )

    async def _dns_lookup(self, host: str) -> ToolResult:
        """Perform DNS resolution."""
        lines = ["DNS Lookup: " + host + "\n"]

        # A records
        try:
            result = socket.getaddrinfo(host, None, socket.AF_INET)
            ips = list(set(r[4][0] for r in result))
            lines.append("  A records: " + ", ".join(ips))
        except socket.gaierror as e:
            lines.append("  A records: FAILED (" + str(e) + ")")
            return ToolResult(success=False, output="\n".join(lines))

        # AAAA records
        try:
            result6 = socket.getaddrinfo(host, None, socket.AF_INET6)
            ips6 = list(set(r[4][0] for r in result6))
            if ips6:
                lines.append("  AAAA records: " + ", ".join(ips6))
        except socket.gaierror:
            pass

        # Reverse DNS
        for ip in ips[:3]:
            try:
                hostname, _, _ = socket.gethostbyaddr(ip)
                lines.append("  Reverse DNS (" + ip + "): " + hostname)
            except socket.herror:
                pass

        # MX records via dig
        stdout, _, rc = await self._run_command(["dig", "+short", "MX", host])
        if rc == 0 and stdout.strip():
            lines.append("  MX records: " + stdout.strip().replace("\n", ", "))

        # NS records
        stdout, _, rc = await self._run_command(["dig", "+short", "NS", host])
        if rc == 0 and stdout.strip():
            lines.append("  NS records: " + stdout.strip().replace("\n", ", "))

        return ToolResult(success=True, output="\n".join(lines), artifacts={"ips": ips})

    async def _ping(self, host: str, kwargs: dict) -> ToolResult:
        """Ping a host."""
        count = kwargs.get("count", 4)
        cmd = ["ping", "-c", str(count), "-W", "3", host]
        stdout, stderr, rc = await self._run_command(cmd, timeout=count * 5 + 5)

        if rc != 0:
            return ToolResult(success=False, output="Ping failed: " + host + "\n" + stderr)

        # Parse stats
        match = re.search(r'(\d+) packets transmitted, (\d+) received.*?(\d+)% packet loss', stdout)
        stats = {}
        if match:
            stats["transmitted"] = int(match.group(1))
            stats["received"] = int(match.group(2))
            stats["loss_percent"] = int(match.group(3))

        rtt_match = re.search(r'rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)', stdout)
        if rtt_match:
            stats["rtt_min_ms"] = float(rtt_match.group(1))
            stats["rtt_avg_ms"] = float(rtt_match.group(2))
            stats["rtt_max_ms"] = float(rtt_match.group(3))

        return ToolResult(success=rc == 0, output=stdout[:2000], artifacts=stats)

    async def _ssl_check(self, host: str, kwargs: dict) -> ToolResult:
        """Check SSL certificate details."""
        port = kwargs.get("port", 443)
        cmd = ["openssl", "s_client", "-connect", host + ":" + str(port), "-servername", host]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(b"Q\n"), timeout=10)
        output = stdout.decode() + stderr.decode()

        # Extract certificate info
        info = {}
        subject_match = re.search(r'subject=(.+)', output)
        if subject_match:
            info["subject"] = subject_match.group(1).strip()

        issuer_match = re.search(r'issuer=(.+)', output)
        if issuer_match:
            info["issuer"] = issuer_match.group(1).strip()

        dates_match = re.search(r'notAfter=(.+)', output)
        if dates_match:
            info["expires"] = dates_match.group(1).strip()

        verify_match = re.search(r'Verify return code: (\d+) \((.+?)\)', output)
        if verify_match:
            info["verify_code"] = int(verify_match.group(1))
            info["verify_msg"] = verify_match.group(2)

        lines = ["SSL Certificate: " + host + ":" + str(port) + "\n"]
        for k, v in info.items():
            lines.append("  " + k + ": " + str(v))

        is_valid = info.get("verify_code", 1) == 0
        icon = "valid" if is_valid else "INVALID"
        lines.insert(1, "  Status: " + icon)

        return ToolResult(success=is_valid, output="\n".join(lines), artifacts=info)

    async def _traceroute(self, host: str) -> ToolResult:
        """Trace route to host."""
        cmd = ["traceroute", "-m", "20", "-w", "2", host]
        stdout, stderr, rc = await self._run_command(cmd, timeout=60)

        if rc != 0 and not stdout:
            # Try tracepath as fallback
            cmd = ["tracepath", host]
            stdout, stderr, rc = await self._run_command(cmd, timeout=60)

        if not stdout:
            return ToolResult(success=False, error="Traceroute failed: " + stderr)

        return ToolResult(success=True, output=stdout[:3000])

    async def _scan_ports(self, host: str, kwargs: dict) -> ToolResult:
        """Scan multiple ports."""
        ports = kwargs.get("ports", [22, 80, 443, 3000, 5000, 5432, 6379, 8080, 8443, 9090])
        timeout = kwargs.get("timeout", 3)

        results = []
        for port in ports:
            start = time.time()
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=timeout
                )
                elapsed = time.time() - start
                writer.close()
                await writer.wait_closed()
                results.append((port, "OPEN", elapsed))
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
                elapsed = time.time() - start
                results.append((port, "CLOSED", elapsed))

        lines = ["Port scan: " + host + "\n"]
        open_ports = []
        for port, status, elapsed in results:
            icon = "OPEN" if status == "OPEN" else "closed"
            lines.append("  " + str(port).ljust(6) + " " + icon + " (" + f"{elapsed*1000:.0f}" + "ms)")
            if status == "OPEN":
                open_ports.append(port)

        lines.append("\nOpen ports: " + str(len(open_ports)) + "/" + str(len(ports)))

        return ToolResult(success=True, output="\n".join(lines), artifacts={"open_ports": open_ports})

    async def _connectivity_check(self, host: str, kwargs: dict) -> ToolResult:
        """Full connectivity diagnostic."""
        results = []

        # DNS
        try:
            ips = socket.gethostbyname_ex(host)[2]
            results.append(("DNS", True, "Resolved to " + ", ".join(ips)))
        except socket.gaierror as e:
            results.append(("DNS", False, str(e)))
            lines = ["Connectivity check: " + host + "\n"]
            lines.append("  DNS: FAILED - " + str(e))
            return ToolResult(success=False, output="\n".join(lines))

        # Ping
        stdout, _, rc = await self._run_command(["ping", "-c", "1", "-W", "3", host])
        results.append(("Ping", rc == 0, "OK" if rc == 0 else "Failed"))

        # HTTP
        for port in [80, 443]:
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=5
                )
                writer.close()
                await writer.wait_closed()
                results.append(("TCP:" + str(port), True, "Connected"))
            except (asyncio.TimeoutError, ConnectionRefusedError, OSError) as e:
                results.append(("TCP:" + str(port), False, str(type(e).__name__)))

        lines = ["Connectivity check: " + host + "\n"]
        all_ok = True
        for check, ok, msg in results:
            icon = "OK" if ok else "FAIL"
            lines.append("  " + check.ljust(10) + " " + icon + " — " + msg)
            if not ok:
                all_ok = False

        return ToolResult(success=all_ok, output="\n".join(lines))
