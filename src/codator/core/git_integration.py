"""Git integration — workspace context from git diff and log."""

from __future__ import annotations

import logging
from pathlib import Path

import git

logger = logging.getLogger(__name__)


class GitContext:
    """Extracts development context from a Git repository."""

    def __init__(self, repo_path: str = "."):
        self._repo_path = Path(repo_path).resolve()
        self._repo: git.Repo | None = None
        self._init_repo()

    def _init_repo(self):
        try:
            self._repo = git.Repo(self._repo_path, search_parent_directories=True)
        except git.InvalidGitRepositoryError:
            logger.info("No git repository found at %s", self._repo_path)
        except Exception as exc:
            logger.warning("Git init failed: %s", exc)

    @property
    def is_available(self) -> bool:
        return self._repo is not None

    def get_work_context(self, max_diff_lines: int = 200) -> str:
        """Build a priority context string from current git state.

        Returns a formatted string containing:
          - Current branch
          - Staged + unstaged changes (truncated diff)
          - Recent commit messages (last 5)
        """
        if not self._repo:
            return "[Git not available — no repository detected]"

        parts: list[str] = []

        # Branch
        try:
            branch = self._repo.active_branch.name
        except TypeError:
            branch = "HEAD (detached)"
        parts.append(f"Branch: {branch}")

        # Status summary
        changed = [item.a_path for item in self._repo.index.diff(None)]
        staged = [item.a_path for item in self._repo.index.diff("HEAD")]
        untracked = self._repo.untracked_files[:20]

        if staged:
            parts.append(f"Staged ({len(staged)}): {', '.join(staged[:10])}")
        if changed:
            parts.append(f"Modified ({len(changed)}): {', '.join(changed[:10])}")
        if untracked:
            parts.append(f"Untracked ({len(untracked)}): {', '.join(untracked[:10])}")

        # Diff (staged + unstaged combined)
        try:
            diff_text = self._repo.git.diff("--stat", "--patch", "--no-color")
            diff_lines = diff_text.split("\n")
            if len(diff_lines) > max_diff_lines:
                diff_text = "\n".join(diff_lines[:max_diff_lines])
                diff_text += f"\n... [truncated, {len(diff_lines) - max_diff_lines} lines omitted]"
            if diff_text.strip():
                parts.append(f"\n--- Git Diff ---\n{diff_text}")
        except git.GitCommandError:
            pass

        # Recent commits
        try:
            commits = list(self._repo.iter_commits(max_count=5))
            if commits:
                log_lines = [
                    f"  {c.hexsha[:8]} {c.summary}" for c in commits
                ]
                parts.append(f"\nRecent commits:\n" + "\n".join(log_lines))
        except Exception:
            pass

        return "\n".join(parts)

    def get_file_at_head(self, file_path: str) -> str | None:
        """Read a file's content from the latest commit (HEAD)."""
        if not self._repo:
            return None
        try:
            blob = self._repo.head.commit.tree / file_path
            return blob.data_stream.read().decode("utf-8", errors="replace")
        except (KeyError, ValueError):
            return None

    def get_changed_files(self) -> list[str]:
        """List files changed since HEAD (staged + unstaged)."""
        if not self._repo:
            return []
        files = set()
        files.update(item.a_path for item in self._repo.index.diff(None))
        files.update(item.a_path for item in self._repo.index.diff("HEAD"))
        return sorted(files)
