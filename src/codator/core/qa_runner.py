"""QA Runner — comprehensive automated testing for Ethos OS NAS.

Systematically tests all Ethos applications via API, browser, and network
diagnostics, then creates tickets for any issues found.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from codator.infrastructure.tools.ethos_ticket_tool import EthosClient

logger = logging.getLogger(__name__)


# ── All known Ethos API endpoints by app ────────────────────────────────────

ETHOS_APP_ENDPOINTS: dict[str, list[str]] = {
    "auth": ["/auth/verify"],
    "dashboard": ["/dashboard/summary"],
    "system": ["/system/info"],
    "apps": ["/apps"],
    "users": ["/users/list", "/users/groups", "/users/password-policy"],
    "storage": [
        "/storage/drives", "/storage/health", "/storage/pool/list",
        "/storage/samba/shares", "/storage/samba/status",
        "/storage/nfs/exports", "/storage/nfs/status",
        "/storage/ftp/status", "/storage/sftp/status",
        "/storage/webdav/shares", "/storage/webdav/status",
        "/storage/smart/schedule", "/storage/app-usage",
        "/storage/maintenance/status", "/storage/maintenance/history?limit=20",
    ],
    "files": ["/files/list?path=/home"],
    "network": [
        "/network/interfaces", "/network/bonds",
        "/network/wifi/status", "/network/wifi/saved",
        "/network/ap/status",
    ],
    "docker": [
        "/docker/status", "/docker/containers",
        "/docker/images", "/docker/projects", "/docker/system",
    ],
    "backup": [
        "/backup/status", "/backup/profiles", "/backup/history",
        "/backup/snapshots", "/backup/scheduled-backups",
        "/backup/btrfs-snapshots",
    ],
    "services": ["/services/list"],
    "packages": [
        "/packages/installed", "/packages/stats", "/packages/upgradable",
    ],
    "firewall": [
        "/firewall/status", "/firewall/rules", "/firewall/banned",
    ],
    "fail2ban": ["/fail2ban/status"],
    "notifications": [
        "/notifications", "/notifications/config", "/notifications/history",
    ],
    "power": ["/power/status"],
    "updates": [
        "/update/check", "/update/config", "/update/rootfs-info",
    ],
    "resources": ["/resources/all"],
    "hardware": ["/hardware/profile"],
    "tickets": ["/tickets/projects"],
    "downloads": ["/downloads/list", "/downloads/stats", "/downloads/config"],
    "surveillance": ["/surveillance/status", "/surveillance/cameras"],
    "dlna": ["/dlna/status"],
    "antivirus": ["/antivirus/status"],
    "ups": ["/ups/status"],
    "wireguard": ["/wireguard/status"],
    "cron": ["/cron/jobs"],
    "gallery": [
        "/gallery/stats", "/gallery/albums", "/gallery/folders",
    ],
    "photos_ai": ["/photos-ai/pkg-status", "/photos-ai/stats"],
    "video_station": [
        "/video-station/pkg-status", "/video-station/home",
    ],
    "radio_music": ["/radio-music/playback-state", "/radio-music/playlists"],
    "aichat": ["/aichat/models/active"],
    "printer": ["/printer/status", "/printer/printers"],
    "domains": [],
    "websites": [],
    "vm": ["/vm/status", "/vm/machines"],
    "builder": ["/builder/status", "/builder/info"],
    "security_advisor": [],
    "doc_anonymizer": ["/doc-anonymizer/status"],
    "med_assistant": ["/med-assistant/status"],
    "rollback": ["/rollback/snapshots", "/rollback/auto"],
    "ethos_packages": ["/ethos-packages"],
    "remote_log": ["/remote-log/config"],
    "cloud_backup": [],
}

SECURITY_HEADERS = [
    "X-Content-Type-Options",
    "X-Frame-Options",
    "Strict-Transport-Security",
    "Content-Security-Policy",
    "Referrer-Policy",
    "Permissions-Policy",
]


@dataclass
class QAFinding:
    """A single QA finding / bug."""

    category: str  # network, auth, api, ui, security, performance
    severity: str  # critical, high, medium, low
    app: str  # which Ethos app
    title: str
    description: str
    endpoint: str = ""
    response_time: float = 0.0
    status_code: int = 0
    details: dict = field(default_factory=dict)

    @property
    def ticket_labels(self) -> list[str]:
        labels = ["qa-automated", self.category]
        if self.app:
            labels.append(self.app)
        return labels


@dataclass
class QAReport:
    """Aggregated QA results."""

    target: str
    started_at: float = 0.0
    finished_at: float = 0.0
    findings: list[QAFinding] = field(default_factory=list)
    api_tested: int = 0
    api_passed: int = 0
    api_failed: int = 0
    api_errors: int = 0

    @property
    def duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def summary(self) -> str:
        sev_counts = {}
        for f in self.findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1

        lines = [
            f"═══ QA Report: {self.target} ═══",
            f"Duration: {self.duration:.1f}s",
            f"API endpoints tested: {self.api_tested} "
            f"(✓{self.api_passed} ✗{self.api_failed} ⚠{self.api_errors})",
            f"Findings: {len(self.findings)}",
        ]
        for sev in ("critical", "high", "medium", "low"):
            if cnt := sev_counts.get(sev, 0):
                lines.append(f"  {sev.upper()}: {cnt}")

        if self.findings:
            lines.append("\n── Top Findings ──")
            for f in sorted(
                self.findings,
                key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    x.severity, 4
                ),
            )[:20]:
                lines.append(
                    f"  [{f.severity.upper()}] [{f.category}] {f.app}: {f.title}"
                )
        return "\n".join(lines)


class QARunner:
    """Comprehensive QA test runner for Ethos OS NAS."""

    def __init__(
        self,
        client: EthosClient,
        project_name: str = "Ethos",
        on_finding: Any = None,
    ):
        self._client = client
        self._project_name = project_name
        self._on_finding = on_finding  # async callback(finding)
        self._report = QAReport(target=client.base_url)

    @property
    def report(self) -> QAReport:
        return self._report

    async def _add_finding(self, finding: QAFinding) -> None:
        self._report.findings.append(finding)
        logger.warning(
            "QA finding: [%s][%s] %s — %s",
            finding.severity,
            finding.category,
            finding.app,
            finding.title,
        )
        if self._on_finding:
            try:
                await self._on_finding(finding)
            except Exception as e:
                logger.error("Finding callback failed: %s", e)

    # ── Main entry point ────────────────────────────────────────────────

    async def run_all(self, *, skip: set[str] | None = None) -> QAReport:
        """Run all QA test categories. Returns the report."""
        skip = skip or set()
        self._report.started_at = time.time()

        suites = [
            ("network", self.test_network),
            ("auth", self.test_auth),
            ("security", self.test_security_headers),
            ("api", self.test_all_api_endpoints),
            ("performance", self.test_performance),
        ]

        for name, func in suites:
            if name in skip:
                logger.info("Skipping QA suite: %s", name)
                continue
            logger.info("Running QA suite: %s", name)
            try:
                await func()
            except Exception as e:
                logger.error("QA suite '%s' crashed: %s", name, e)
                await self._add_finding(
                    QAFinding(
                        category="infrastructure",
                        severity="high",
                        app="qa-runner",
                        title=f"QA suite '{name}' crashed",
                        description=str(e),
                    )
                )

        self._report.finished_at = time.time()
        return self._report

    # ── Network tests ───────────────────────────────────────────────────

    async def test_network(self) -> None:
        """Test connectivity, SSL, and common ports."""
        import socket
        import ssl

        from urllib.parse import urlparse

        parsed = urlparse(self._client.base_url)
        host = parsed.hostname or ""
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        # DNS resolution
        try:
            socket.getaddrinfo(host, None)
        except socket.gaierror as e:
            await self._add_finding(
                QAFinding(
                    category="network",
                    severity="critical",
                    app="dns",
                    title=f"DNS resolution failed for {host}",
                    description=str(e),
                )
            )
            return

        # TCP connectivity
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=10
            )
            writer.close()
            await writer.wait_closed()
        except Exception as e:
            await self._add_finding(
                QAFinding(
                    category="network",
                    severity="critical",
                    app="tcp",
                    title=f"Cannot connect to {host}:{port}",
                    description=str(e),
                )
            )
            return

        # SSL certificate check
        if parsed.scheme == "https":
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = True
                ctx.verify_mode = ssl.CERT_REQUIRED
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port, ssl=ctx), timeout=10
                )
                writer.close()
                await writer.wait_closed()
            except ssl.SSLCertVerificationError as e:
                await self._add_finding(
                    QAFinding(
                        category="network",
                        severity="medium",
                        app="ssl",
                        title="SSL certificate validation failed",
                        description=str(e),
                    )
                )
            except Exception:
                pass  # non-SSL errors already covered

        # Port scan for common NAS services
        common_ports = {
            22: "SSH",
            80: "HTTP",
            443: "HTTPS",
            445: "SMB",
            139: "NetBIOS",
            548: "AFP",
            2049: "NFS",
            8080: "HTTP-Alt",
            9090: "Webmin",
            5001: "DLNA",
            8096: "Jellyfin",
            51820: "WireGuard",
        }
        open_ports = []
        for p, svc in common_ports.items():
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, p), timeout=2
                )
                writer.close()
                await writer.wait_closed()
                open_ports.append(f"{p}/{svc}")
            except Exception:
                pass

        logger.info("Open ports on %s: %s", host, ", ".join(open_ports))

    # ── Auth tests ──────────────────────────────────────────────────────

    async def test_auth(self) -> None:
        """Test authentication flows."""
        # Test with invalid credentials
        async with httpx.AsyncClient(
            verify=False, timeout=15
        ) as http:
            try:
                resp = await http.post(
                    f"{self._client.base_url}/api/auth/login",
                    json={"username": "invalid_user_xyz", "password": "wrongpass"},
                )
                data = resp.json()
                if resp.status_code == 200 and data.get("token"):
                    await self._add_finding(
                        QAFinding(
                            category="auth",
                            severity="critical",
                            app="auth",
                            title="Invalid credentials accepted",
                            description="Server issued a token for invalid username/password",
                            status_code=resp.status_code,
                        )
                    )
            except Exception:
                pass

            # Test access without token
            try:
                resp = await http.get(
                    f"{self._client.base_url}/api/dashboard/summary"
                )
                if resp.status_code == 200:
                    await self._add_finding(
                        QAFinding(
                            category="auth",
                            severity="critical",
                            app="auth",
                            title="API accessible without authentication",
                            description="/api/dashboard/summary returned 200 with no token",
                            endpoint="/dashboard/summary",
                            status_code=200,
                        )
                    )
            except Exception:
                pass

            # Test with garbage token
            try:
                resp = await http.get(
                    f"{self._client.base_url}/api/dashboard/summary",
                    headers={"Authorization": "Bearer garbage_token_12345"},
                )
                if resp.status_code == 200:
                    await self._add_finding(
                        QAFinding(
                            category="auth",
                            severity="critical",
                            app="auth",
                            title="API accessible with invalid token",
                            description="/api/dashboard/summary returned 200 with garbage bearer token",
                            endpoint="/dashboard/summary",
                            status_code=200,
                        )
                    )
            except Exception:
                pass

    # ── Security headers ────────────────────────────────────────────────

    async def test_security_headers(self) -> None:
        """Check HTTP security headers on main page."""
        async with httpx.AsyncClient(
            verify=False, timeout=15
        ) as http:
            try:
                resp = await http.get(self._client.base_url)
            except Exception as e:
                await self._add_finding(
                    QAFinding(
                        category="security",
                        severity="medium",
                        app="web",
                        title="Cannot fetch main page for header check",
                        description=str(e),
                    )
                )
                return

            missing = []
            for header in SECURITY_HEADERS:
                if header.lower() not in {
                    k.lower() for k in resp.headers.keys()
                }:
                    missing.append(header)

            if missing:
                await self._add_finding(
                    QAFinding(
                        category="security",
                        severity="medium",
                        app="web",
                        title=f"Missing security headers: {', '.join(missing)}",
                        description=(
                            "The following recommended HTTP security headers are missing "
                            f"from the main page response: {', '.join(missing)}. "
                            "These headers help protect against XSS, clickjacking, "
                            "MIME sniffing, and other attacks."
                        ),
                        endpoint="/",
                        status_code=resp.status_code,
                        details={"missing_headers": missing},
                    )
                )

            # Check cookie flags
            cookies = resp.headers.get_list("set-cookie")
            for cookie in cookies:
                issues = []
                cl = cookie.lower()
                if "httponly" not in cl:
                    issues.append("missing HttpOnly")
                if "secure" not in cl:
                    issues.append("missing Secure")
                if "samesite" not in cl:
                    issues.append("missing SameSite")
                if issues:
                    name = cookie.split("=")[0].strip()
                    await self._add_finding(
                        QAFinding(
                            category="security",
                            severity="medium",
                            app="web",
                            title=f"Cookie '{name}' missing flags: {', '.join(issues)}",
                            description=f"Cookie: {cookie[:120]}",
                        )
                    )

    # ── API endpoint tests ──────────────────────────────────────────────

    async def test_all_api_endpoints(self) -> None:
        """Smoke-test every known Ethos API endpoint."""
        if not self._client.is_authenticated:
            await self._client.login()

        # Flatten endpoints
        all_endpoints: list[tuple[str, str]] = []
        for app, paths in ETHOS_APP_ENDPOINTS.items():
            for path in paths:
                all_endpoints.append((app, path))

        # Test in batches of 5 (concurrency limiter)
        sem = asyncio.Semaphore(5)

        async def test_one(app: str, path: str) -> None:
            async with sem:
                self._report.api_tested += 1
                start = time.time()
                try:
                    data, err = await self._client.api_safe(
                        path, timeout=20
                    )
                    elapsed = time.time() - start

                    if err:
                        self._report.api_errors += 1
                        # Only report non-404 errors as findings
                        if "404" not in err and "405" not in err:
                            await self._add_finding(
                                QAFinding(
                                    category="api",
                                    severity="medium",
                                    app=app,
                                    title=f"API error: {path}",
                                    description=err[:500],
                                    endpoint=path,
                                    response_time=elapsed,
                                )
                            )
                    else:
                        self._report.api_passed += 1
                        # Check for error responses in the data
                        if isinstance(data, dict) and data.get("error"):
                            self._report.api_failed += 1
                            self._report.api_passed -= 1
                            await self._add_finding(
                                QAFinding(
                                    category="api",
                                    severity="low",
                                    app=app,
                                    title=f"API returned error: {path}",
                                    description=str(data["error"])[:500],
                                    endpoint=path,
                                    response_time=elapsed,
                                )
                            )
                except Exception as e:
                    self._report.api_errors += 1
                    await self._add_finding(
                        QAFinding(
                            category="api",
                            severity="medium",
                            app=app,
                            title=f"API exception: {path}",
                            description=f"{type(e).__name__}: {e}",
                            endpoint=path,
                        )
                    )

        tasks = [test_one(app, path) for app, path in all_endpoints]
        await asyncio.gather(*tasks)

    # ── Performance tests ───────────────────────────────────────────────

    async def test_performance(self) -> None:
        """Check for slow endpoints."""
        slow_threshold = 5.0  # seconds

        critical_endpoints = [
            ("/dashboard/summary", "dashboard"),
            ("/files/list?path=/home", "files"),
            ("/storage/drives", "storage"),
            ("/services/list", "services"),
            ("/resources/all", "resources"),
        ]

        for path, app in critical_endpoints:
            start = time.time()
            data, err = await self._client.api_safe(path, timeout=30)
            elapsed = time.time() - start

            if elapsed > slow_threshold:
                await self._add_finding(
                    QAFinding(
                        category="performance",
                        severity="medium" if elapsed < 10 else "high",
                        app=app,
                        title=f"Slow endpoint: {path} ({elapsed:.1f}s)",
                        description=(
                            f"Response time {elapsed:.1f}s exceeds "
                            f"{slow_threshold}s threshold."
                        ),
                        endpoint=path,
                        response_time=elapsed,
                    )
                )

    # ── Ticket creation ─────────────────────────────────────────────────

    async def create_tickets(
        self, project_name: str = "Ethos"
    ) -> list[dict]:
        """Create Ethos tickets for all findings."""
        from codator.infrastructure.tools.ethos_ticket_tool import EthosTicketTool

        tool = EthosTicketTool(client=self._client)

        # Resolve project
        try:
            project_id = await tool._resolve_project_id(project_name)
        except ValueError:
            logger.error("Project '%s' not found — cannot create tickets", project_name)
            return []

        created = []
        for finding in self._report.findings:
            result = await tool.execute(
                action="create",
                project=project_name,
                title=f"[QA-{finding.severity.upper()}] {finding.title}",
                description=self._format_finding_body(finding),
                type="bug" if finding.category != "performance" else "task",
                priority=finding.severity,
                column="Backlog",
                labels=finding.ticket_labels,
            )
            if result.success:
                created.append(result.artifacts)
                logger.info("Created ticket: %s", finding.title)
            else:
                logger.error("Failed to create ticket: %s — %s", finding.title, result.error)

        return created

    @staticmethod
    def _format_finding_body(finding: QAFinding) -> str:
        """Format a finding as a Markdown ticket description."""
        lines = [
            f"## {finding.title}",
            "",
            f"**Category:** {finding.category}",
            f"**Severity:** {finding.severity}",
            f"**App:** {finding.app}",
        ]
        if finding.endpoint:
            lines.append(f"**Endpoint:** `{finding.endpoint}`")
        if finding.status_code:
            lines.append(f"**Status Code:** {finding.status_code}")
        if finding.response_time:
            lines.append(f"**Response Time:** {finding.response_time:.2f}s")
        lines.extend(["", "### Description", "", finding.description])
        if finding.details:
            lines.extend(["", "### Details", "", f"```json\n{finding.details}\n```"])
        lines.extend([
            "",
            "---",
            "*Created by codator QA Runner (automated)*",
        ])
        return "\n".join(lines)
