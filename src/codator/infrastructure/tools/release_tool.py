"""Release management tool — semver, changelog, git tags, releases."""

import asyncio
import os
import re
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


class ReleaseTool:
    name = "release"
    description = (
        "Manage releases: bump semantic version, auto-generate changelog from commits, "
        "create git tags, draft GitHub/Gitea releases. Follows conventional commits format."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["current_version", "bump", "changelog", "tag", "create_release", "list_tags"],
                "description": "Release action to perform",
            },
            "path": {
                "type": "string",
                "description": "Project directory",
            },
            "bump_type": {
                "type": "string",
                "enum": ["major", "minor", "patch", "auto"],
                "description": "Version bump type (auto detects from commits)",
                "default": "auto",
            },
            "version": {
                "type": "string",
                "description": "Explicit version string (overrides bump_type)",
            },
            "since": {
                "type": "string",
                "description": "Generate changelog since this tag/commit",
            },
            "title": {
                "type": "string",
                "description": "Release title",
            },
            "draft": {
                "type": "boolean",
                "description": "Create as draft release",
                "default": False,
            },
            "prerelease": {
                "type": "boolean",
                "description": "Mark as pre-release",
                "default": False,
            },
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs.get("path", ".")

        try:
            if action == "current_version":
                return await self._current_version(path)
            elif action == "bump":
                return await self._bump_version(path, kwargs)
            elif action == "changelog":
                return await self._generate_changelog(path, kwargs)
            elif action == "tag":
                return await self._create_tag(path, kwargs)
            elif action == "create_release":
                return await self._create_release(path, kwargs)
            elif action == "list_tags":
                return await self._list_tags(path)
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
        return stdout.decode().strip(), stderr.decode().strip(), proc.returncode

    async def _current_version(self, path: str) -> ToolResult:
        """Detect current version from various sources."""
        version = None
        source = ""

        # Check pyproject.toml
        pyproject = os.path.join(path, "pyproject.toml")
        if os.path.isfile(pyproject):
            with open(pyproject) as f:
                content = f.read()
            match = re.search(r'version\s*=\s*["\']([^"\']+)', content)
            if match:
                version = match.group(1)
                source = "pyproject.toml"

        # Check package.json
        if not version:
            pkg_json = os.path.join(path, "package.json")
            if os.path.isfile(pkg_json):
                import json
                with open(pkg_json) as f:
                    data = json.load(f)
                version = data.get("version")
                source = "package.json"

        # Check Cargo.toml
        if not version:
            cargo = os.path.join(path, "Cargo.toml")
            if os.path.isfile(cargo):
                with open(cargo) as f:
                    content = f.read()
                match = re.search(r'version\s*=\s*"([^"]+)"', content)
                if match:
                    version = match.group(1)
                    source = "Cargo.toml"

        # Check git tags
        if not version:
            stdout, _, rc = await self._run_command(["git", "describe", "--tags", "--abbrev=0"], cwd=path)
            if rc == 0 and stdout:
                version = stdout.lstrip("v")
                source = "git tag"

        if not version:
            return ToolResult(success=False, error="Could not detect version. No pyproject.toml, package.json, Cargo.toml, or git tags found.")

        return ToolResult(
            success=True,
            output=f"Current version: {version} (from {source})",
            artifacts={"version": version, "source": source},
        )

    async def _bump_version(self, path: str, kwargs: dict) -> ToolResult:
        """Bump version based on semver rules."""
        bump_type = kwargs.get("bump_type", "auto")
        explicit_version = kwargs.get("version")

        # Get current version
        result = await self._current_version(path)
        if not result.success:
            return result

        current = result.artifacts["version"]
        source = result.artifacts["source"]

        if explicit_version:
            new_version = explicit_version
        elif bump_type == "auto":
            new_version = await self._auto_bump(path, current)
        else:
            new_version = self._semver_bump(current, bump_type)

        # Update version file
        updated = await self._update_version_file(path, source, current, new_version)

        return ToolResult(
            success=True,
            output=f"Bumped version: {current} → {new_version} (in {source})\n{updated}",
            artifacts={"old_version": current, "new_version": new_version, "source": source},
        )

    def _semver_bump(self, version: str, bump_type: str) -> str:
        match = re.match(r'(\d+)\.(\d+)\.(\d+)', version)
        if not match:
            return version
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if bump_type == "major":
            return f"{major + 1}.0.0"
        elif bump_type == "minor":
            return f"{major}.{minor + 1}.0"
        else:
            return f"{major}.{minor}.{patch + 1}"

    async def _auto_bump(self, path: str, current: str) -> str:
        """Auto-detect bump type from conventional commits."""
        stdout, _, _ = await self._run_command(
            ["git", "log", "--oneline", "--format=%s", f"v{current}..HEAD"], cwd=path
        )
        if not stdout:
            stdout, _, _ = await self._run_command(
                ["git", "log", "--oneline", "--format=%s", "-20"], cwd=path
            )

        commits = stdout.split("\n") if stdout else []

        has_breaking = any("BREAKING" in c or "!" in c.split(":")[0] for c in commits if ":" in c)
        has_feat = any(c.startswith("feat") for c in commits)

        if has_breaking:
            return self._semver_bump(current, "major")
        elif has_feat:
            return self._semver_bump(current, "minor")
        else:
            return self._semver_bump(current, "patch")

    async def _update_version_file(self, path: str, source: str, old: str, new: str) -> str:
        if source == "pyproject.toml":
            filepath = os.path.join(path, "pyproject.toml")
            with open(filepath) as f:
                content = f.read()
            content = content.replace(f'version = "{old}"', f'version = "{new}"')
            content = content.replace(f"version = '{old}'", f"version = '{new}'")
            with open(filepath, "w") as f:
                f.write(content)
            return f"Updated {filepath}"
        elif source == "package.json":
            import json
            filepath = os.path.join(path, "package.json")
            with open(filepath) as f:
                data = json.load(f)
            data["version"] = new
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return f"Updated {filepath}"
        elif source == "Cargo.toml":
            filepath = os.path.join(path, "Cargo.toml")
            with open(filepath) as f:
                content = f.read()
            content = content.replace(f'version = "{old}"', f'version = "{new}"')
            with open(filepath, "w") as f:
                f.write(content)
            return f"Updated {filepath}"
        return ""

    async def _generate_changelog(self, path: str, kwargs: dict) -> ToolResult:
        """Generate changelog from conventional commits."""
        since = kwargs.get("since", "")

        if since:
            cmd = ["git", "log", "--format=%H|%s|%an|%ai", f"{since}..HEAD"]
        else:
            cmd = ["git", "log", "--format=%H|%s|%an|%ai", "-50"]

        stdout, _, rc = await self._run_command(cmd, cwd=path)
        if rc != 0:
            return ToolResult(success=False, error="Failed to get git log")

        # Categorize commits
        categories = {
            "feat": ("✨ Features", []),
            "fix": ("🐛 Bug Fixes", []),
            "perf": ("⚡ Performance", []),
            "refactor": ("♻️ Refactoring", []),
            "docs": ("📚 Documentation", []),
            "test": ("✅ Tests", []),
            "ci": ("🔧 CI/CD", []),
            "chore": ("🔨 Chores", []),
            "other": ("📝 Other", []),
        }

        breaking_changes = []

        for line in stdout.split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            sha, subject, author, date = parts

            # Parse conventional commit
            match = re.match(r'^(\w+)(?:\(.+?\))?(!)?:\s*(.+)', subject)
            if match:
                ctype = match.group(1)
                breaking = match.group(2)
                msg = match.group(3)
                if breaking:
                    breaking_changes.append(msg)
            else:
                ctype = "other"
                msg = subject

            category = ctype if ctype in categories else "other"
            categories[category][1].append(f"- {msg} ({sha[:7]})")

        # Generate markdown
        today = datetime.now().strftime("%Y-%m-%d")
        lines = [f"## Changelog ({today})\n"]

        if breaking_changes:
            lines.append("### ⚠️ BREAKING CHANGES")
            for bc in breaking_changes:
                lines.append(f"- {bc}")
            lines.append("")

        for key, (title, commits) in categories.items():
            if commits:
                lines.append(f"### {title}")
                lines.extend(commits)
                lines.append("")

        changelog = "\n".join(lines)
        return ToolResult(
            success=True,
            output=changelog,
            artifacts={"total_commits": sum(len(c[1]) for c in categories.values()), "breaking": len(breaking_changes)},
        )

    async def _create_tag(self, path: str, kwargs: dict) -> ToolResult:
        """Create a git tag."""
        version = kwargs.get("version", "")
        if not version:
            # Auto-detect next version
            result = await self._current_version(path)
            if result.success:
                version = "v" + self._semver_bump(result.artifacts["version"], "patch")
            else:
                return ToolResult(success=False, error="Cannot determine version for tag")

        if not version.startswith("v"):
            version = f"v{version}"

        title = kwargs.get("title", f"Release {version}")

        cmd = ["git", "tag", "-a", version, "-m", title]
        _, stderr, rc = await self._run_command(cmd, cwd=path)

        if rc != 0:
            return ToolResult(success=False, error=f"Failed to create tag: {stderr}")

        return ToolResult(success=True, output=f"Created tag: {version} — \"{title}\"", artifacts={"tag": version})

    async def _create_release(self, path: str, kwargs: dict) -> ToolResult:
        """Create a GitHub/Gitea release."""
        version = kwargs.get("version", "")
        title = kwargs.get("title", "")
        draft = kwargs.get("draft", False)
        prerelease = kwargs.get("prerelease", False)

        if not version:
            result = await self._current_version(path)
            if result.success:
                version = "v" + result.artifacts["version"]
            else:
                return ToolResult(success=False, error="Cannot determine version")

        if not version.startswith("v"):
            version = f"v{version}"

        if not title:
            title = f"Release {version}"

        # Generate changelog for release notes
        changelog_result = await self._generate_changelog(path, {"since": ""})
        notes = changelog_result.output if changelog_result.success else ""

        # Try gh CLI first
        cmd = ["gh", "release", "create", version, "--title", title, "--notes", notes]
        if draft:
            cmd.append("--draft")
        if prerelease:
            cmd.append("--prerelease")

        stdout, stderr, rc = await self._run_command(cmd, cwd=path)

        if rc != 0:
            return ToolResult(
                success=False,
                error=f"Failed to create release (gh CLI): {stderr}\nMake sure 'gh' is installed and authenticated.",
                output=f"Tag: {version}\nTitle: {title}\nNotes:\n{notes[:2000]}",
            )

        return ToolResult(
            success=True,
            output=f"Release created: {version}\nURL: {stdout}",
            artifacts={"version": version, "url": stdout},
        )

    async def _list_tags(self, path: str) -> ToolResult:
        """List existing git tags."""
        stdout, _, rc = await self._run_command(
            ["git", "tag", "--sort=-creatordate", "--format=%(refname:short) %(creatordate:short) %(subject)"],
            cwd=path,
        )
        if rc != 0:
            return ToolResult(success=False, error="Failed to list tags")

        tags = stdout.strip().split("\n") if stdout.strip() else []
        return ToolResult(
            success=True,
            output=f"Tags ({len(tags)}):\n" + "\n".join(f"  {t}" for t in tags[:30]),
            artifacts={"count": len(tags)},
        )
