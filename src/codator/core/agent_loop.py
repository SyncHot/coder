"""Plan-Act-Verify agentic cycle for structured coding tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from codator.infrastructure.tools.terminal_tool import TerminalTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset(
    {"read_file", "edit_file", "create_file", "run_command", "delete_file",
     "analyze"}
)


@dataclass
class AgentStep:
    index: int
    action: str
    target: str
    description: str
    status: str = "pending"


@dataclass
class AgentPlan:
    task: str
    steps: list[AgentStep]
    reasoning: str


@dataclass
class ActionResult:
    step: AgentStep
    success: bool
    output: str
    error: str


@dataclass
class VerifyResult:
    success: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    plan: AgentPlan
    actions: list[ActionResult]
    verification: VerifyResult
    heal_iterations: int
    final_success: bool


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
You are a coding assistant that produces structured plans.
Given a task and optional project context, return a JSON object with:
{
  "task": "<restate the task concisely>",
  "reasoning": "<brief explanation of your approach>",
  "steps": [
    {"action": "<read_file|edit_file|create_file|run_command|delete_file|analyze>",
     "target": "<file path or shell command>",
     "description": "<what this step does>"}
  ]
}

Rules:
- Only use the six allowed actions.
- **ALWAYS start by reading the relevant files first** before editing or running commands.
- If the task is a question, review, or analysis (e.g. "what can be improved",
  "check this file", "review the code"), use only read_file and analyze actions.
  Do NOT edit or run commands for analytical tasks.
- The "analyze" action takes a file path as target and a description of what to
  look for. It is read-only and produces observations — no changes.
- **Use the project file tree provided in context to find correct file paths.**
  Never guess file paths — always refer to the actual files listed in the project context.
- File paths must be relative to the project root.
- For edit_file the description MUST be a JSON string:
  {"file": "path", "old": "exact text to find", "new": "replacement text"}
  IMPORTANT: The "old" field must be copied EXACTLY from the file you read —
  including indentation, whitespace, and line endings. Do NOT paraphrase or
  summarise the old text. Copy it character-for-character from the read_file output.
- Keep plans minimal — only steps that are necessary.
- Return ONLY valid JSON, no markdown fences.
"""

_HEAL_SYSTEM = """\
You are a coding assistant that fixes errors.
Given the original task, the errors encountered, and optional project context,
produce a new JSON plan to fix the errors. Use the same JSON schema as the
planning phase:
{
  "task": "<fix description>",
  "reasoning": "<what went wrong and how to fix>",
  "steps": [...]
}

Return ONLY valid JSON, no markdown fences.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlanActVerifyAgent:
    """Structured Plan → Act → Verify agent backed by Ollama."""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:14b-instruct-q6_K",
        project_root: str = ".",
        max_heal_iterations: int = 3,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._project_root = Path(project_root).resolve()
        self._max_heal = max_heal_iterations
        self._terminal = TerminalTool(
            working_dir=str(self._project_root),
            timeout=60,
            require_confirm=True,
        )

    # -- helpers -------------------------------------------------------------

    def _safe_path(self, relative: str) -> Path:
        """Resolve *relative* inside project root; raise on traversal."""
        resolved = (self._project_root / relative).resolve()
        if not str(resolved).startswith(str(self._project_root)):
            raise ValueError(
                f"Path traversal blocked: {relative!r} escapes project root"
            )
        return resolved

    async def _ollama_chat(self, system: str, user: str,
                          *, force_json: bool = True) -> str:
        """Call Ollama POST /api/chat and return content.

        Uses a long read timeout (10 min) to support large quantised models
        (e.g. 32b-q3) that may generate slowly.
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if force_json:
            payload["format"] = "json"
        # connect fast, but allow up to 10 min for model generation
        timeout = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=timeout
        ) as client:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        return data["message"]["content"]

    @staticmethod
    def _parse_plan(raw: str, task: str) -> AgentPlan:
        """Parse JSON response into an AgentPlan, validating actions."""
        obj = json.loads(raw)
        steps: list[AgentStep] = []
        for i, s in enumerate(obj.get("steps", [])):
            action = s.get("action", "").strip()
            if not action or action not in VALID_ACTIONS:
                logger.warning("Skipping step %d with invalid action: %r", i, action)
                continue
            steps.append(
                AgentStep(
                    index=i,
                    action=action,
                    target=s.get("target", ""),
                    description=s.get("description", ""),
                )
            )
        return AgentPlan(
            task=obj.get("task", task),
            steps=steps,
            reasoning=obj.get("reasoning", ""),
        )

    # -- plan ----------------------------------------------------------------

    async def plan(
        self, task: str, project_context: str = ""
    ) -> AgentPlan:
        """Ask the model for a structured plan to accomplish *task*."""
        user_msg = f"Task:\n{task}"
        if project_context:
            user_msg += f"\n\nProject context:\n{project_context}"

        logger.info("Planning for task: %s", task)
        raw = await self._ollama_chat(_PLAN_SYSTEM, user_msg)
        logger.debug("Raw plan response: %s", raw)
        plan = self._parse_plan(raw, task)
        logger.info(
            "Plan created with %d steps — %s", len(plan.steps), plan.reasoning
        )
        return plan

    # -- act -----------------------------------------------------------------

    async def act(self, step: AgentStep) -> ActionResult:
        """Execute a single plan step and return the outcome."""
        logger.info("Executing step %d: %s %s", step.index, step.action, step.target)
        try:
            handler = {
                "read_file": self._act_read_file,
                "edit_file": self._act_edit_file,
                "create_file": self._act_create_file,
                "run_command": self._act_run_command,
                "delete_file": self._act_delete_file,
                "analyze": self._act_analyze,
            }.get(step.action)
            if handler is None:
                raise ValueError(f"Unknown action: {step.action!r}")
            output = await handler(step)
            step.status = "done"
            return ActionResult(step=step, success=True, output=output, error="")
        except Exception as exc:
            logger.error("Step %d failed: %s", step.index, exc)
            step.status = "failed"
            return ActionResult(step=step, success=False, output="", error=str(exc))

    async def _act_read_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        return path.read_text(encoding="utf-8")

    async def _act_edit_file(self, step: AgentStep) -> str:
        # Handle both JSON string and dict descriptions
        if isinstance(step.description, dict):
            edit_info = step.description
        else:
            edit_info = json.loads(step.description)
        file_rel = edit_info.get("file", step.target)
        old_text: str = edit_info["old"]
        new_text: str = edit_info["new"]

        path = self._safe_path(file_rel)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # backup before editing
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

        content = path.read_text(encoding="utf-8")

        # 1. Try exact match
        if old_text in content:
            content = content.replace(old_text, new_text, 1)
            path.write_text(content, encoding="utf-8")
            return f"Edited {file_rel} (exact match, backup at {backup.name})"

        # 2. Try whitespace-normalised match
        match_pos = self._fuzzy_find(content, old_text)
        if match_pos is not None:
            start, end = match_pos
            # Preserve indentation of the first matched line
            matched_block = content[start:end]
            matched_lines = matched_block.splitlines(keepends=True)
            new_lines = new_text.splitlines(keepends=True)
            if matched_lines and new_lines:
                import re
                orig_indent = re.match(r"(\s*)", matched_lines[0]).group(1)
                new_indent = re.match(r"(\s*)", new_lines[0]).group(1)
                if orig_indent and not new_indent:
                    # LLM omitted indentation — reindent new_text
                    new_text = "".join(
                        orig_indent + l if l.strip() else l
                        for l in new_lines
                    )
            content = content[:start] + new_text + content[end:]
            path.write_text(content, encoding="utf-8")
            return f"Edited {file_rel} (fuzzy match, backup at {backup.name})"

        raise ValueError(
            f"Text to replace not found in {file_rel} (neither exact nor fuzzy)"
        )

    @staticmethod
    def _fuzzy_find(content: str, needle: str) -> tuple[int, int] | None:
        """Find the best approximate match of *needle* inside *content*.

        Returns (start, end) character indices in *content* or None.
        Uses character-level similarity on sliding windows of lines.
        """
        import difflib

        needle_lines = needle.strip().splitlines()
        if not needle_lines:
            return None

        content_lines = content.splitlines(keepends=True)
        if not content_lines:
            return None

        # Normalised needle text for comparison (stripped lines joined)
        needle_norm = "\n".join(l.strip() for l in needle_lines)
        n = len(needle_lines)

        best_ratio = 0.0
        best_span: tuple[int, int] | None = None

        # Try windows of n-2 .. n+2 lines to handle off-by-a-few
        for delta in range(-2, 3):
            m = n + delta
            if m < 1 or m > len(content_lines):
                continue
            for i in range(len(content_lines) - m + 1):
                window_norm = "\n".join(
                    l.strip() for l in content_lines[i:i + m]
                )
                sm = difflib.SequenceMatcher(
                    None, needle_norm, window_norm, autojunk=False
                )
                ratio = sm.ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_span = (i, i + m)

        # Require at least 55% character-level similarity
        if best_ratio < 0.55 or best_span is None:
            return None

        start_line, end_line = best_span
        start_pos = sum(len(l) for l in content_lines[:start_line])
        end_pos = sum(len(l) for l in content_lines[:end_line])
        logger.info(
            "Fuzzy match: ratio=%.2f, lines %d-%d in file",
            best_ratio, start_line, end_line,
        )
        return start_pos, end_pos

    async def _act_create_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(step.description, encoding="utf-8")
        return f"Created {step.target}"

    async def _act_run_command(self, step: AgentStep) -> str:
        result = await self._terminal.run_command(step.target, timeout=60)
        if not result.success:
            raise RuntimeError(
                f"Command failed: {result.error or result.output}"
            )
        return (result.output or "") + (result.error or "")

    async def _act_delete_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        # backup before deleting
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        path.unlink()
        return f"Deleted {step.target} (backup at {backup.name})"

    async def _act_analyze(self, step: AgentStep) -> str:
        """Read-only analysis: read the file and return its content for review."""
        path = self._safe_path(step.target)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        content = path.read_text(encoding="utf-8")
        # Truncate very large files for analysis
        if len(content) > 15000:
            content = content[:15000] + f"\n... [truncated, total {len(content)} chars]"
        return f"=== {step.target} ===\n{content}"

    # -- verify --------------------------------------------------------------

    async def verify(self, language: str = "python") -> VerifyResult:
        """Run linting and tests to verify project health."""
        errors: list[str] = []
        warnings: list[str] = []

        if language == "python":
            errors, warnings = await self._verify_python()
        else:
            logger.warning("No verification rules for language %r", language)

        success = len(errors) == 0
        return VerifyResult(success=success, errors=errors, warnings=warnings)

    async def _verify_python(self) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        # --- ruff ---
        ruff_result = await self._run_quiet("ruff check .")
        if ruff_result is not None:
            for line in ruff_result.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if any(
                    stripped.startswith(p)
                    for p in ("Found", "All checks", "[")
                ):
                    continue
                # ruff marks warnings with codes starting with W or D
                if ": W" in stripped or ": D" in stripped:
                    warnings.append(stripped)
                else:
                    errors.append(stripped)

        # --- pytest (only if tests/ exists) ---
        tests_dir = self._project_root / "tests"
        if tests_dir.is_dir():
            pytest_result = await self._run_quiet(
                "python -m pytest --tb=short -q"
            )
            if pytest_result is not None:
                for line in pytest_result.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if "FAILED" in stripped or "ERROR" in stripped:
                        errors.append(stripped)
                    elif "warning" in stripped.lower():
                        warnings.append(stripped)

        return errors, warnings

    async def _run_quiet(self, cmd: str) -> str | None:
        """Run *cmd* via sandboxed TerminalTool and return combined output."""
        try:
            result = await self._terminal.run_command(cmd, timeout=120)
            return (result.output or "") + (result.error or "")
        except Exception as exc:
            logger.warning("Verification command %r failed: %s", cmd, exc)
            return None

    # -- self-heal -----------------------------------------------------------

    async def self_heal(
        self,
        errors: list[str],
        original_task: str,
        project_context: str = "",
    ) -> AgentPlan:
        """Ask the model to produce a fix plan for *errors*."""
        user_msg = (
            f"Original task:\n{original_task}\n\n"
            f"Errors:\n" + "\n".join(errors)
        )
        if project_context:
            user_msg += f"\n\nProject context:\n{project_context}"

        logger.info("Self-healing: %d errors to fix", len(errors))
        raw = await self._ollama_chat(_HEAL_SYSTEM, user_msg)
        return self._parse_plan(raw, f"fix: {original_task}")

    # -- full cycle ----------------------------------------------------------

    async def run(
        self,
        task: str,
        project_context: str = "",
        on_step: Callable[[str, str], object] | None = None,
        use_git_transaction: bool = True,
    ) -> AgentResult:
        """Execute the full Plan → Act → Verify (→ Heal) cycle.

        When *use_git_transaction* is True and the project is a git repo,
        all changes are made on a temporary branch.  On success the branch
        is merged back; on failure it is rolled back automatically.
        """

        async def _notify(description: str, status: str) -> None:
            if on_step is not None:
                result = on_step(description, status)
                if asyncio.iscoroutine(result):
                    await result

        # ---- Git transaction setup -----------------------------------------
        original_branch: str | None = None
        agent_branch: str | None = None

        if use_git_transaction:
            original_branch, agent_branch = self._git_create_branch()

        try:
            agent_result = await self._run_inner(
                task, project_context, _notify,
            )
        except Exception:
            if agent_branch and original_branch:
                self._git_rollback(original_branch, agent_branch)
            raise

        # ---- Git transaction commit/rollback -------------------------------
        if agent_branch and original_branch:
            if agent_result.final_success:
                self._git_merge(original_branch, agent_branch)
                await _notify("Git: merged agent branch", "done")
            else:
                self._git_rollback(original_branch, agent_branch)
                await _notify("Git: rolled back agent branch", "done")

        return agent_result

    async def _run_inner(
        self,
        task: str,
        project_context: str,
        _notify: Callable,
    ) -> AgentResult:
        """Core agent loop (extracted for git transaction wrapping)."""
        # ---- Plan ----------------------------------------------------------
        await _notify("Planning…", "started")
        current_plan = await self.plan(task, project_context)
        await _notify(f"Plan ready — {len(current_plan.steps)} steps", "done")

        all_actions: list[ActionResult] = []
        heal_iterations = 0

        # Detect analysis-only plans (no edits/creates/deletes/commands)
        _mutating_actions = {"edit_file", "create_file", "delete_file", "run_command"}
        is_analysis_only = not any(
            s.action in _mutating_actions for s in current_plan.steps
        )

        for iteration in range(1 + self._max_heal):
            # ---- Act -------------------------------------------------------
            for step in current_plan.steps:
                await _notify(step.description, "running")
                result = await self.act(step)
                all_actions.append(result)
                status = "done" if result.success else "failed"
                await _notify(step.description, status)
                if not result.success:
                    logger.warning(
                        "Step %d failed: %s", step.index, result.error
                    )

            # ---- Verify (skip for read-only analysis) ----------------------
            if is_analysis_only:
                # Gather all analysis outputs as the "result"
                analysis_output = "\n".join(
                    a.output for a in all_actions if a.success and a.output
                )
                verification = VerifyResult(success=True)
                await _notify("Analysis complete", "done")

                # Generate a summary response from the model (plain text, not JSON)
                summary_prompt = (
                    f"Based on your analysis of the code, answer the user's "
                    f"original question:\n\n{task}\n\n"
                    f"Here is what you found:\n{analysis_output[:8000]}"
                )
                try:
                    summary = await self._ollama_chat(
                        "You are a helpful coding assistant. Provide a clear, "
                        "actionable review of the code. Be specific about what "
                        "is good and what should be improved. "
                        "Respond in the same language as the user's question.",
                        summary_prompt,
                        force_json=False,
                    )
                    all_actions.append(ActionResult(
                        step=AgentStep(
                            index=len(all_actions),
                            action="analyze",
                            target="summary",
                            description="Analysis summary",
                            status="done",
                        ),
                        success=True, output=summary, error="",
                    ))
                except Exception as exc:
                    logger.warning("Summary generation failed: %s", exc)

                return AgentResult(
                    plan=current_plan,
                    actions=all_actions,
                    verification=verification,
                    heal_iterations=0,
                    final_success=True,
                )

            await _notify("Verifying…", "running")
            verification = await self.verify()
            await _notify(
                "Verification " + ("passed" if verification.success else "failed"),
                "done" if verification.success else "failed",
            )

            if verification.success:
                return AgentResult(
                    plan=current_plan,
                    actions=all_actions,
                    verification=verification,
                    heal_iterations=heal_iterations,
                    final_success=True,
                )

            # ---- Heal (if budget remains) ----------------------------------
            if iteration < self._max_heal:
                heal_iterations += 1
                await _notify(
                    f"Self-healing (attempt {heal_iterations})…", "running"
                )
                try:
                    current_plan = await self.self_heal(
                        verification.errors, task, project_context
                    )
                    await _notify(
                        f"Heal plan ready — {len(current_plan.steps)} steps", "done"
                    )
                except (json.JSONDecodeError, ValueError, KeyError) as exc:
                    logger.warning("Heal plan parsing failed: %s", exc)
                    await _notify(
                        f"Self-healing failed (bad response): {exc}", "failed"
                    )
                    break  # stop heal loop — no valid plan to retry

        # exhausted heal budget
        return AgentResult(
            plan=current_plan,
            actions=all_actions,
            verification=verification,
            heal_iterations=heal_iterations,
            final_success=False,
        )

    # ---- Git transaction helpers -------------------------------------------

    def _git_create_branch(self) -> tuple[str | None, str | None]:
        """Create agent branch. Returns (original_branch, agent_branch)."""
        import subprocess
        import time

        try:
            original = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self._project_root),
                capture_output=True, text=True, timeout=5,
            )
            if original.returncode != 0:
                return None, None
            orig_branch = original.stdout.strip()

            agent_branch = f"codator/agent-{int(time.time())}"
            subprocess.run(
                ["git", "checkout", "-b", agent_branch],
                cwd=str(self._project_root),
                capture_output=True, text=True, timeout=5,
                check=True,
            )
            logger.info("Created agent branch: %s", agent_branch)
            return orig_branch, agent_branch
        except Exception as exc:
            logger.debug("Git branch creation failed: %s", exc)
            return None, None

    def _git_merge(
        self, original_branch: str, agent_branch: str,
    ) -> None:
        """Merge agent branch back and clean up."""
        import subprocess

        try:
            # Commit any uncommitted changes on agent branch
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(self._project_root),
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "commit", "-m",
                 f"codator agent: auto-commit from {agent_branch}",
                 "--allow-empty"],
                cwd=str(self._project_root),
                capture_output=True, timeout=5,
            )
            # Switch back and merge
            subprocess.run(
                ["git", "checkout", original_branch],
                cwd=str(self._project_root),
                capture_output=True, timeout=5, check=True,
            )
            subprocess.run(
                ["git", "merge", "--no-ff", agent_branch,
                 "-m", f"Merge codator agent work ({agent_branch})"],
                cwd=str(self._project_root),
                capture_output=True, timeout=10, check=True,
            )
            subprocess.run(
                ["git", "branch", "-d", agent_branch],
                cwd=str(self._project_root),
                capture_output=True, timeout=5,
            )
            logger.info("Merged agent branch %s into %s",
                        agent_branch, original_branch)
        except Exception as exc:
            logger.error("Git merge failed: %s", exc)

    def _git_rollback(
        self, original_branch: str, agent_branch: str,
    ) -> None:
        """Discard agent branch and return to original."""
        import subprocess

        try:
            # Discard all changes
            subprocess.run(
                ["git", "checkout", "--force", original_branch],
                cwd=str(self._project_root),
                capture_output=True, timeout=5,
            )
            subprocess.run(
                ["git", "branch", "-D", agent_branch],
                cwd=str(self._project_root),
                capture_output=True, timeout=5,
            )
            logger.info("Rolled back agent branch %s", agent_branch)
        except Exception as exc:
            logger.error("Git rollback failed: %s", exc)
