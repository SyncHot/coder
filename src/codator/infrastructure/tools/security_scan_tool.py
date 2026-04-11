"""Security scanning tool — secrets detection, dependency audit, SAST patterns."""

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


# Common secret patterns
SECRET_PATTERNS = [
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([A-Za-z0-9_\-]{20,})', "API Key"),
    (r'(?i)(secret|password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\']{8,})', "Secret/Password"),
    (r'(?i)(token)\s*[=:]\s*["\']?([A-Za-z0-9_\-\.]{20,})', "Token"),
    (r'(ghp_[A-Za-z0-9]{36})', "GitHub Personal Access Token"),
    (r'(gho_[A-Za-z0-9]{36})', "GitHub OAuth Token"),
    (r'(sk-[A-Za-z0-9]{48})', "OpenAI API Key"),
    (r'(AKIA[0-9A-Z]{16})', "AWS Access Key ID"),
    (r'(?i)(aws[_-]?secret[_-]?access[_-]?key)\s*[=:]\s*["\']?([A-Za-z0-9/+=]{40})', "AWS Secret Key"),
    (r'(-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----)', "Private Key"),
    (r'(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})', "JWT Token"),
    (r'(?i)(slack[_-]?(?:token|webhook))\s*[=:]\s*["\']?(xox[bpas]-[A-Za-z0-9\-]+)', "Slack Token"),
    (r'(SG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43})', "SendGrid API Key"),
    (r'(?i)(database[_-]?url|db[_-]?url)\s*[=:]\s*["\']?((?:postgres|mysql|mongodb)://[^\s"\']+)', "Database URL"),
]

IGNORE_PATHS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", "dist", "build"}


class SecurityScanTool:
    name = "security_scan"
    description = (
        "Scan for security issues: leaked secrets/credentials in code, "
        "vulnerable dependencies (pip-audit/npm audit/trivy), and common SAST patterns. "
        "Supports Python, JavaScript, Go, Rust, and Docker projects."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["secrets", "dependencies", "sast", "full"],
                "description": "Scan type: secrets detection, dependency audit, SAST patterns, or full scan",
            },
            "path": {
                "type": "string",
                "description": "Project directory to scan",
            },
            "fix": {
                "type": "boolean",
                "description": "Attempt to auto-fix (for dependencies: upgrade vulnerable packages)",
                "default": False,
            },
            "severity": {
                "type": "string",
                "enum": ["critical", "high", "medium", "low", "all"],
                "description": "Minimum severity to report",
                "default": "medium",
            },
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Glob patterns to exclude from scanning",
            },
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs["path"]

        if not os.path.isdir(path):
            return ToolResult(success=False, error=f"Directory not found: {path}")

        try:
            if action == "secrets":
                return await self._scan_secrets(path, kwargs)
            elif action == "dependencies":
                return await self._scan_dependencies(path, kwargs)
            elif action == "sast":
                return await self._scan_sast(path, kwargs)
            elif action == "full":
                results = []
                r1 = await self._scan_secrets(path, kwargs)
                r2 = await self._scan_dependencies(path, kwargs)
                r3 = await self._scan_sast(path, kwargs)
                combined = f"=== SECRETS ===\n{r1.output}\n\n=== DEPENDENCIES ===\n{r2.output}\n\n=== SAST ===\n{r3.output}"
                total_issues = (r1.artifacts.get("findings", 0) +
                               r2.artifacts.get("vulnerabilities", 0) +
                               r3.artifacts.get("issues", 0))
                return ToolResult(
                    success=total_issues == 0,
                    output=combined[:8000],
                    artifacts={"total_issues": total_issues},
                )
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

    async def _scan_secrets(self, path: str, kwargs: dict) -> ToolResult:
        """Scan for leaked secrets in source code."""
        exclude = set(kwargs.get("exclude_patterns", []))
        findings = []

        # Try gitleaks first (if available)
        try:
            stdout, stderr, rc = await self._run_command(
                ["gitleaks", "detect", "--source", path, "--report-format", "json", "--no-git", "-r", "/dev/stdout"]
            )
            if stdout.strip():
                leaks = json.loads(stdout)
                for leak in leaks:
                    findings.append({
                        "file": leak.get("File", ""),
                        "line": leak.get("StartLine", 0),
                        "type": leak.get("RuleID", "unknown"),
                        "match": leak.get("Match", "")[:50] + "...",
                        "severity": "high",
                    })
                return ToolResult(
                    success=len(findings) == 0,
                    output=self._format_secret_findings(findings),
                    artifacts={"findings": len(findings), "tool": "gitleaks"},
                )
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        # Fallback: manual regex scanning
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE_PATHS]
            rel_root = os.path.relpath(root, path)

            for fname in files:
                if fname.endswith((".pyc", ".so", ".o", ".bin", ".png", ".jpg", ".gif", ".ico")):
                    continue
                filepath = os.path.join(root, fname)
                rel_path = os.path.join(rel_root, fname)

                if any(re.match(p, rel_path) for p in exclude):
                    continue

                try:
                    with open(filepath, "r", errors="ignore") as f:
                        content = f.read(100_000)  # Limit file size
                except (IOError, OSError):
                    continue

                for line_num, line in enumerate(content.split("\n"), 1):
                    for pattern, secret_type in SECRET_PATTERNS:
                        if re.search(pattern, line):
                            # Skip if in a comment or test file
                            if line.strip().startswith(("#", "//", "*", "/*")):
                                continue
                            findings.append({
                                "file": rel_path,
                                "line": line_num,
                                "type": secret_type,
                                "match": line.strip()[:80],
                                "severity": "high",
                            })
                            break

        return ToolResult(
            success=len(findings) == 0,
            output=self._format_secret_findings(findings),
            artifacts={"findings": len(findings), "tool": "regex"},
        )

    def _format_secret_findings(self, findings: list) -> str:
        if not findings:
            return "✅ No secrets detected"
        lines = [f"⚠️  Found {len(findings)} potential secret(s):\n"]
        for f in findings[:30]:
            lines.append(f"  [{f['severity'].upper()}] {f['file']}:{f['line']} — {f['type']}")
            lines.append(f"    {f['match']}")
        if len(findings) > 30:
            lines.append(f"\n  ... and {len(findings) - 30} more")
        return "\n".join(lines)

    async def _scan_dependencies(self, path: str, kwargs: dict) -> ToolResult:
        """Scan for vulnerable dependencies."""
        fix = kwargs.get("fix", False)
        severity = kwargs.get("severity", "medium")
        results = []

        # Python: pip-audit
        if os.path.isfile(os.path.join(path, "requirements.txt")) or os.path.isfile(os.path.join(path, "pyproject.toml")):
            stdout, stderr, rc = await self._run_command(["pip-audit", "--format=json", "-r", "requirements.txt"], cwd=path)
            if rc == 0 or stdout:
                try:
                    vulns = json.loads(stdout) if stdout.strip() else []
                    results.append(("Python (pip-audit)", vulns))
                except json.JSONDecodeError:
                    results.append(("Python (pip-audit)", [{"raw": stdout + stderr}]))

            if fix and os.path.isfile(os.path.join(path, "requirements.txt")):
                await self._run_command(["pip-audit", "--fix", "-r", "requirements.txt"], cwd=path)

        # JavaScript: npm audit
        if os.path.isfile(os.path.join(path, "package.json")):
            audit_cmd = ["npm", "audit", "--json"]
            stdout, stderr, rc = await self._run_command(audit_cmd, cwd=path)
            try:
                data = json.loads(stdout) if stdout.strip() else {}
                vulns = data.get("vulnerabilities", {})
                results.append(("JavaScript (npm audit)", vulns))
            except json.JSONDecodeError:
                results.append(("JavaScript (npm audit)", [{"raw": stdout[:500]}]))

            if fix:
                await self._run_command(["npm", "audit", "fix"], cwd=path)

        # Try trivy for container/broader scanning
        stdout, stderr, rc = await self._run_command(
            ["trivy", "fs", "--format", "json", "--severity", severity.upper(), path]
        )
        if rc == 0 and stdout:
            try:
                data = json.loads(stdout)
                trivy_results = data.get("Results", [])
                results.append(("Trivy", trivy_results))
            except json.JSONDecodeError:
                pass

        # Format output
        total_vulns = 0
        lines = []
        for tool_name, vulns in results:
            if isinstance(vulns, dict):
                count = len(vulns)
            elif isinstance(vulns, list):
                count = len(vulns)
            else:
                count = 0
            total_vulns += count
            lines.append(f"\n{tool_name}: {count} issue(s)")
            if isinstance(vulns, list):
                for v in vulns[:10]:
                    if isinstance(v, dict):
                        name = v.get("name", v.get("PkgName", "unknown"))
                        sev = v.get("fix_versions", v.get("Severity", ""))
                        lines.append(f"  • {name} — {sev}")

        if not results:
            return ToolResult(success=True, output="No dependency scanning tools available (install pip-audit, npm, or trivy)")

        output = f"{'✅ No vulnerabilities' if total_vulns == 0 else f'⚠️  {total_vulns} vulnerability(ies) found'}\n" + "\n".join(lines)
        return ToolResult(
            success=total_vulns == 0,
            output=output[:5000],
            artifacts={"vulnerabilities": total_vulns, "fix_applied": fix},
        )

    async def _scan_sast(self, path: str, kwargs: dict) -> ToolResult:
        """Static Application Security Testing — common vulnerability patterns."""
        issues = []

        # Try bandit for Python
        if any(f.endswith(".py") for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))):
            stdout, stderr, rc = await self._run_command(
                ["bandit", "-r", path, "-f", "json", "-ll"]
            )
            if stdout.strip():
                try:
                    data = json.loads(stdout)
                    for result in data.get("results", []):
                        issues.append({
                            "file": result.get("filename", ""),
                            "line": result.get("line_number", 0),
                            "severity": result.get("issue_severity", "MEDIUM"),
                            "confidence": result.get("issue_confidence", ""),
                            "issue": result.get("issue_text", ""),
                            "cwe": result.get("issue_cwe", {}).get("id", ""),
                        })
                except json.JSONDecodeError:
                    pass

        # Manual SAST patterns (language-agnostic)
        sast_patterns = [
            (r'eval\s*\(', "Dangerous eval() usage", "high"),
            (r'exec\s*\(', "Dangerous exec() usage", "high"),
            (r'subprocess\.call\s*\([^)]*shell\s*=\s*True', "Shell injection risk", "high"),
            (r'os\.system\s*\(', "OS command injection risk", "medium"),
            (r'pickle\.loads?\s*\(', "Insecure deserialization", "high"),
            (r'yaml\.load\s*\([^)]*(?!Loader)', "Unsafe YAML loading", "medium"),
            (r'(?i)innerHTML\s*=', "Potential XSS via innerHTML", "medium"),
            (r'(?i)document\.write\s*\(', "Potential XSS via document.write", "medium"),
            (r'SQL.*\+.*(?:request|input|params|query)', "Potential SQL injection", "high"),
            (r'(?i)verify\s*=\s*False', "SSL verification disabled", "medium"),
            (r'(?i)chmod\s+777', "Overly permissive file permissions", "medium"),
            (r'(?i)bind\s*\(\s*["\']0\.0\.0\.0', "Binding to all interfaces", "low"),
        ]

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE_PATHS]
            for fname in files:
                if not fname.endswith((".py", ".js", ".ts", ".go", ".rb", ".php", ".java")):
                    continue
                filepath = os.path.join(root, fname)
                rel_path = os.path.relpath(filepath, path)
                try:
                    with open(filepath, "r", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            for pattern, desc, sev in sast_patterns:
                                if re.search(pattern, line):
                                    issues.append({
                                        "file": rel_path,
                                        "line": line_num,
                                        "severity": sev,
                                        "issue": desc,
                                    })
                                    break
                except (IOError, OSError):
                    continue

        if not issues:
            return ToolResult(success=True, output="✅ No SAST issues detected", artifacts={"issues": 0})

        lines = [f"⚠️  Found {len(issues)} SAST issue(s):\n"]
        for issue in sorted(issues, key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("severity", "low"), 3))[:40]:
            lines.append(f"  [{issue['severity'].upper()}] {issue['file']}:{issue.get('line', '?')} — {issue['issue']}")

        return ToolResult(
            success=False,
            output="\n".join(lines)[:5000],
            artifacts={"issues": len(issues)},
        )
