"""Plan-Act-Verify agentic cycle for structured coding tasks."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset(
    {"read_file", "edit_file", "create_file", "run_command", "delete_file"}
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
    {"action": "<read_file|edit_file|create_file|run_command|delete_file>",
     "target": "<file path or shell command>",
     "description": "<what this step does>"}
  ]
}

Rules:
- Only use the five allowed actions.
- File paths must be relative to the project root.
- For edit_file the description MUST be a JSON string:
  {"file": "path", "old": "text to find", "new": "replacement text"}
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

    # -- helpers -------------------------------------------------------------

    def _safe_path(self, relative: str) -> Path:
        """Resolve *relative* inside project root; raise on traversal."""
        resolved = (self._project_root / relative).resolve()
        if not str(resolved).startswith(str(self._project_root)):
            raise ValueError(
                f"Path traversal blocked: {relative!r} escapes project root"
            )
        return resolved

    async def _ollama_chat(self, system: str, user: str) -> str:
        """Call Ollama POST /api/chat with format:'json' and return content."""
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": "json",
        }
        async with httpx.AsyncClient(
            base_url=self._base_url, timeout=120.0
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
            action = s.get("action", "")
            if action not in VALID_ACTIONS:
                raise ValueError(f"Invalid action in step {i}: {action!r}")
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
        if old_text not in content:
            raise ValueError(
                f"Text to replace not found in {file_rel}"
            )
        content = content.replace(old_text, new_text, 1)
        path.write_text(content, encoding="utf-8")
        return f"Edited {file_rel} (backup at {backup.name})"

    async def _act_create_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(step.description, encoding="utf-8")
        return f"Created {step.target}"

    async def _act_run_command(self, step: AgentStep) -> str:
        proc = await asyncio.create_subprocess_shell(
            step.target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._project_root),
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        except TimeoutError:
            proc.kill()
            await proc.communicate()
            raise TimeoutError(
                f"Command timed out after 60s: {step.target!r}"
            )
        output = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command exited {proc.returncode}:\n{err or output}"
            )
        return output + err

    async def _act_delete_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        # backup before deleting
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        path.unlink()
        return f"Deleted {step.target} (backup at {backup.name})"

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
        """Run *cmd* and return combined output, or None on execution error."""
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(self._project_root),
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120)
            return stdout.decode(errors="replace")
        except (TimeoutError, OSError) as exc:
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
    ) -> AgentResult:
        """Execute the full Plan → Act → Verify (→ Heal) cycle."""

        async def _notify(description: str, status: str) -> None:
            if on_step is not None:
                result = on_step(description, status)
                if asyncio.iscoroutine(result):
                    await result

        # ---- Plan ----------------------------------------------------------
        await _notify("Planning…", "started")
        current_plan = await self.plan(task, project_context)
        await _notify(f"Plan ready — {len(current_plan.steps)} steps", "done")

        all_actions: list[ActionResult] = []
        heal_iterations = 0

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

            # ---- Verify ----------------------------------------------------
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
                current_plan = await self.self_heal(
                    verification.errors, task, project_context
                )
                await _notify(
                    f"Heal plan ready — {len(current_plan.steps)} steps", "done"
                )

        # exhausted heal budget
        return AgentResult(
            plan=current_plan,
            actions=all_actions,
            verification=verification,
            heal_iterations=heal_iterations,
            final_success=False,
        )
