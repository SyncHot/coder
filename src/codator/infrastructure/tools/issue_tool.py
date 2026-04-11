"""Issue Tool — create and manage issues/tickets in Git hosting platforms."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import Any

import httpx

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class IssueTool(Tool):
    """Create and list issues on GitHub, Gitea, or via gh CLI.

    Supports multiple backends:
    - gh CLI (auto-detected, works with GitHub)
    - Gitea API (for self-hosted)
    - GitHub API (direct, with token)
    """

    def __init__(
        self,
        backend: str = "auto",  # "auto", "gh", "gitea", "github"
        gitea_url: str = "",
        gitea_token: str = "",
        github_token: str = "",
        default_repo: str = "",  # "owner/repo"
        default_labels: list[str] | None = None,
    ):
        self._backend = backend
        self._gitea_url = gitea_url.rstrip("/")
        self._gitea_token = gitea_token
        self._github_token = github_token
        self._default_repo = default_repo
        self._default_labels = default_labels or []
        self._gh_available: bool | None = None

    @property
    def name(self) -> str:
        return "issue"

    @property
    def description(self) -> str:
        return (
            "Create, list, and manage issues/tickets in a git hosting platform. "
            "Actions: create, list, comment, close. "
            "Supports GitHub (gh CLI), Gitea API, and GitHub API."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["create", "list", "comment", "close"],
                    "description": "Action to perform.",
                },
                "title": {
                    "type": "string",
                    "description": "Issue title (required for create).",
                },
                "body": {
                    "type": "string",
                    "description": "Issue body/description.",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to apply.",
                },
                "repo": {
                    "type": "string",
                    "description": "Repository (owner/repo). Uses default if not specified.",
                },
                "issue_number": {
                    "type": "integer",
                    "description": "Issue number (for comment/close).",
                },
                "comment_body": {
                    "type": "string",
                    "description": "Comment text (for comment action).",
                },
            },
            "required": ["action"],
        }

    def _check_gh(self) -> bool:
        """Check if gh CLI is available and authenticated."""
        if self._gh_available is not None:
            return self._gh_available
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, text=True, timeout=10,
            )
            self._gh_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._gh_available = False
        return self._gh_available

    async def _create_gh(self, title: str, body: str, labels: list[str], repo: str) -> ToolResult:
        """Create issue via gh CLI."""
        cmd = ["gh", "issue", "create", "--title", title, "--body", body]
        if repo:
            cmd.extend(["--repo", repo])
        for label in labels:
            cmd.extend(["--label", label])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                url = result.stdout.strip()
                return ToolResult(success=True, output=f"Issue created: {url}")
            return ToolResult(success=False, error=f"gh failed: {result.stderr.strip()}")
        except Exception as exc:
            return ToolResult(success=False, error=f"gh CLI error: {exc}")

    async def _create_gitea(self, title: str, body: str, labels: list[str], repo: str) -> ToolResult:
        """Create issue via Gitea API."""
        if not self._gitea_url or not self._gitea_token:
            return ToolResult(success=False, error="Gitea URL and token required.")
        if not repo:
            return ToolResult(success=False, error="Repository (owner/repo) required.")

        url = f"{self._gitea_url}/api/v1/repos/{repo}/issues"
        headers = {"Authorization": f"token {self._gitea_token}"}
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            # Gitea needs label IDs — try to resolve by name
            label_ids = await self._resolve_gitea_labels(repo, labels)
            if label_ids:
                payload["labels"] = label_ids

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return ToolResult(
                success=True,
                output=f"Issue #{data['number']} created: {data['html_url']}",
                artifacts={"issue_number": data["number"], "url": data["html_url"]},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"Gitea API error: {exc}")

    async def _resolve_gitea_labels(self, repo: str, label_names: list[str]) -> list[int]:
        """Resolve label names to IDs via Gitea API."""
        url = f"{self._gitea_url}/api/v1/repos/{repo}/labels"
        headers = {"Authorization": f"token {self._gitea_token}"}
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                all_labels = resp.json()
            name_to_id = {l["name"].lower(): l["id"] for l in all_labels}
            return [name_to_id[n.lower()] for n in label_names if n.lower() in name_to_id]
        except Exception:
            return []

    async def _create_github_api(self, title: str, body: str, labels: list[str], repo: str) -> ToolResult:
        """Create issue via GitHub REST API."""
        if not self._github_token:
            return ToolResult(success=False, error="GitHub token required.")
        if not repo:
            return ToolResult(success=False, error="Repository (owner/repo) required.")

        url = f"https://api.github.com/repos/{repo}/issues"
        headers = {
            "Authorization": f"Bearer {self._github_token}",
            "Accept": "application/vnd.github+json",
        }
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
            return ToolResult(
                success=True,
                output=f"Issue #{data['number']} created: {data['html_url']}",
                artifacts={"issue_number": data["number"], "url": data["html_url"]},
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"GitHub API error: {exc}")

    async def _list_issues(self, repo: str) -> ToolResult:
        """List open issues."""
        if self._check_gh():
            cmd = ["gh", "issue", "list", "--state", "open", "--limit", "20"]
            if repo:
                cmd.extend(["--repo", repo])
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    return ToolResult(success=True, output=result.stdout.strip() or "No open issues.")
                return ToolResult(success=False, error=result.stderr.strip())
            except Exception as exc:
                return ToolResult(success=False, error=str(exc))
        return ToolResult(success=False, error="No issue backend available for listing.")

    async def execute(self, **kwargs) -> ToolResult:
        """Execute an issue action.

        Parameters
        ----------
        action : str
            One of: create, list, comment, close.
        """
        action = kwargs.get("action", "create")
        title = kwargs.get("title", "")
        body = kwargs.get("body", "")
        labels = kwargs.get("labels", []) + self._default_labels
        repo = kwargs.get("repo", "") or self._default_repo
        issue_number = kwargs.get("issue_number")

        if action == "create":
            if not title:
                return ToolResult(success=False, error="Title is required for creating an issue.")

            # Auto-detect backend
            backend = self._backend
            if backend == "auto":
                if self._gitea_url and self._gitea_token:
                    backend = "gitea"
                elif self._check_gh():
                    backend = "gh"
                elif self._github_token:
                    backend = "github"
                else:
                    return ToolResult(
                        success=False,
                        error="No issue backend configured. Set up gh CLI, Gitea, or GitHub token.",
                    )

            if backend == "gh":
                return await self._create_gh(title, body, labels, repo)
            elif backend == "gitea":
                return await self._create_gitea(title, body, labels, repo)
            elif backend == "github":
                return await self._create_github_api(title, body, labels, repo)
            else:
                return ToolResult(success=False, error=f"Unknown backend: {backend}")

        elif action == "list":
            return await self._list_issues(repo)

        elif action == "comment":
            if not issue_number:
                return ToolResult(success=False, error="issue_number required for comment.")
            comment_body = kwargs.get("comment_body", "")
            if self._check_gh():
                cmd = ["gh", "issue", "comment", str(issue_number), "--body", comment_body]
                if repo:
                    cmd.extend(["--repo", repo])
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        return ToolResult(success=True, output="Comment added.")
                    return ToolResult(success=False, error=result.stderr.strip())
                except Exception as exc:
                    return ToolResult(success=False, error=str(exc))
            return ToolResult(success=False, error="Comment not supported on this backend yet.")

        elif action == "close":
            if not issue_number:
                return ToolResult(success=False, error="issue_number required for close.")
            if self._check_gh():
                cmd = ["gh", "issue", "close", str(issue_number)]
                if repo:
                    cmd.extend(["--repo", repo])
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    if result.returncode == 0:
                        return ToolResult(success=True, output=f"Issue #{issue_number} closed.")
                    return ToolResult(success=False, error=result.stderr.strip())
                except Exception as exc:
                    return ToolResult(success=False, error=str(exc))
            return ToolResult(success=False, error="Close not supported on this backend yet.")

        return ToolResult(success=False, error=f"Unknown action: {action}")
