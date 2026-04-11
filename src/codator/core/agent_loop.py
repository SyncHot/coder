"""Plan-Act-Verify agentic cycle for structured coding tasks."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import shutil
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

import httpx

from codator.infrastructure.tools.terminal_tool import TerminalTool
from codator.infrastructure.tools.file_tool import GrepTool, GlobTool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Robust JSON parsing for LLM output
# ---------------------------------------------------------------------------

def safe_parse_json(raw: str) -> Any:
    """Parse JSON from LLM output, tolerating common generation errors.

    Strategy (in order):
    1. Standard json.loads
    2. Fix invalid backslash escapes (\\n in code blocks, Windows paths, etc.)
    3. Fix unescaped quotes inside JSON string values (LLMs forget to escape
       ``"`` when embedding code containing double quotes)
    4. dirtyjson (lenient parser that handles trailing commas, unquoted keys, etc.)
    5. Extract first JSON object/array via regex, then retry steps 1-4
    """
    text = PlanActVerifyAgent._strip_json_fences(raw)

    # 1. Standard parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fix invalid backslash escapes — LLMs often produce \n, \t inside
    #    code blocks that are NOT valid JSON escapes, or Windows-style paths
    #    like C:\Users.  Replace unrecognised \X sequences with \\X.
    _VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
    def _fix_escapes(s: str) -> str:
        result: list[str] = []
        i = 0
        while i < len(s):
            if s[i] == '\\' and i + 1 < len(s):
                next_ch = s[i + 1]
                if next_ch in _VALID_JSON_ESCAPES:
                    result.append(s[i:i+2])
                    i += 2
                else:
                    # Invalid escape — double the backslash
                    result.append('\\\\')
                    i += 1
            else:
                result.append(s[i])
                i += 1
        return "".join(result)

    fixed = _fix_escapes(text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 3. Fix unescaped quotes — LLMs embed code like replace(",", ".") in
    #    JSON strings, sometimes escaping some " but not others.  A " inside
    #    a *value* string is only a real terminator if it is followed by valid
    #    JSON structure (,"key": or } or ]).  Key strings are always short
    #    and trusted.
    def _fix_unescaped_quotes(s: str) -> str:
        out: list[str] = []
        i = 0
        in_string = False
        expect_key = False  # next string will be a key
        is_key = False      # current string is a key

        while i < len(s):
            c = s[i]

            if in_string:
                if c == '\\' and i + 1 < len(s):
                    out.append(s[i:i + 2])
                    i += 2
                    continue

                if c == '"':
                    if is_key:
                        # Keys are simple identifiers — trust closing quote
                        in_string = False
                        out.append(c)
                    elif _is_real_value_end(s, i):
                        in_string = False
                        out.append(c)
                    else:
                        out.append('\\"')
                    i += 1
                    continue

                out.append(c)
                i += 1
            else:
                if c in '{,':
                    expect_key = True
                elif c == ':':
                    expect_key = False
                elif c == '"':
                    in_string = True
                    is_key = expect_key
                    expect_key = False

                out.append(c)
                i += 1

        return "".join(out)

    def _is_real_value_end(s: str, pos: int) -> bool:
        """Return True if the ``"`` at *pos* ends a JSON value string.

        Accepts only:
        - end of input
        - ``}`` or ``]`` (end of object/array)
        - ``,`` then ``"somekey"`` then ``:`` (next key-value pair)
        """
        j = pos + 1
        while j < len(s) and s[j] in ' \t\n\r':
            j += 1

        if j >= len(s):
            return True

        ch = s[j]
        if ch in '}]':
            return True
        if ch == ',':
            # Expect: , <ws> "key" <ws> :
            k = j + 1
            while k < len(s) and s[k] in ' \t\n\r':
                k += 1
            if k < len(s) and s[k] == '"':
                # Find closing quote of the key (skip escaped quotes)
                k += 1
                while k < len(s):
                    if s[k] == '\\' and k + 1 < len(s):
                        k += 2
                        continue
                    if s[k] == '"':
                        # Found end of key — check for ':'
                        k += 1
                        while k < len(s) and s[k] in ' \t\n\r':
                            k += 1
                        return k < len(s) and s[k] == ':'
                    k += 1
            return False

        return False

    for src in (text, fixed):
        try:
            return json.loads(_fix_unescaped_quotes(src))
        except json.JSONDecodeError:
            pass

    # 4. dirtyjson (lenient)
    try:
        import dirtyjson
        return dirtyjson.loads(text)
    except Exception:
        pass

    # 5. Extract first JSON object/array via regex
    m = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
    if m:
        extracted = m.group(1)
        for attempt in (extracted, _fix_escapes(extracted)):
            try:
                return json.loads(attempt)
            except json.JSONDecodeError:
                pass
            try:
                return json.loads(_fix_unescaped_quotes(attempt))
            except json.JSONDecodeError:
                pass
        try:
            import dirtyjson
            return dirtyjson.loads(extracted)
        except Exception:
            pass

    # All strategies failed — raise with the original error for diagnostics
    return json.loads(text)  # will raise JSONDecodeError


# ---------------------------------------------------------------------------
# LLM newline normalization — fix double/under-escaped newlines in code
# ---------------------------------------------------------------------------

def _unescape_collapsed_code(text: str) -> str:
    r"""Convert literal ``\n`` to real newlines in LLM-generated code.

    LLMs often double-escape newlines in JSON edit descriptions, producing
    code collapsed to a single line with literal ``\n``.  This converts
    ``\n`` → real newline ONLY when **outside** Python string literals,
    preserving ``\n`` inside strings where it belongs (e.g. ``'\n'.join(…)``).

    Also handles ``\t`` outside strings → real tab.
    """
    if "\\n" not in text and "\\t" not in text:
        return text

    result: list[str] = []
    i = 0
    in_str: str | None = None  # None, or the quote delimiter ("'", '"', "'''", '"""')

    while i < len(text):
        # --- triple-quote boundaries (must check before single-quote) ---
        tri = text[i : i + 3]
        if tri in ("'''", '"""'):
            if in_str is None:
                in_str = tri
            elif in_str == tri:
                in_str = None
            result.append(tri)
            i += 3
            continue

        c = text[i]

        # --- escape sequences ---
        if c == "\\" and i + 1 < len(text):
            nc = text[i + 1]
            if nc == "n":
                # \n outside string → real newline; inside → keep literal
                result.append("\\n" if in_str else "\n")
                i += 2
                continue
            if nc == "t":
                result.append("\\t" if in_str else "\t")
                i += 2
                continue
            if nc in ("\\", "'", '"'):
                # Escaped backslash or quote — keep as-is, skip both chars
                result.append(c + nc)
                i += 2
                continue
            # Other \X (e.g. \d in regex) — keep as-is
            result.append(c + nc)
            i += 2
            continue

        # --- string delimiter toggle (single-char quotes) ---
        if c in ("'", '"'):
            if in_str is None:
                in_str = c
            elif in_str == c:
                in_str = None
            # else: different quote type inside a string → just a character

        result.append(c)
        i += 1

    return "".join(result)


def _escape_broken_strings(text: str) -> str:
    r"""Fix real newlines inside Python string literals → ``\n``.

    The reverse of ``_unescape_collapsed_code``: when an LLM *under*-escapes
    in JSON (uses ``\n`` instead of ``\\n`` for Python's escape), json.loads
    turns them into real newlines **inside** string literals — a syntax error
    in Python.  This converts them back to ``\n`` and strips the spurious
    indentation whitespace that follows.

    Only applies to single/double-quoted strings (triple-quoted strings
    legitimately contain real newlines).
    """
    if "\n" not in text:
        return text

    result: list[str] = []
    i = 0
    in_str: str | None = None

    while i < len(text):
        # --- triple-quote boundaries ---
        tri = text[i : i + 3]
        if tri in ("'''", '"""'):
            if in_str is None:
                in_str = tri
            elif in_str == tri:
                in_str = None
            result.append(tri)
            i += 3
            continue

        c = text[i]

        # --- escape sequences — skip both chars, don't toggle state ---
        if c == "\\" and i + 1 < len(text):
            result.append(c + text[i + 1])
            i += 2
            continue

        # --- string delimiter toggle (single-char quotes) ---
        if c in ("'", '"'):
            if in_str is None:
                in_str = c
            elif in_str == c:
                in_str = None

        # --- real newline inside single/double-quoted string → \n ---
        if c == "\n" and in_str is not None and len(in_str) == 1:
            result.append("\\n")
            i += 1
            # Skip indentation whitespace injected by the code structure
            while i < len(text) and text[i] in (" ", "\t"):
                i += 1
            continue

        result.append(c)
        i += 1

    return "".join(result)


_COLLAPSED_STMT_RE = re.compile(
    r"(?<=[\)\]\}\w])"  # preceded by ), ], }, or word char
    r"([ ]{4,})"  # 4+ spaces (suggests indent, not single space)
    r"((?:return|import|from|if|elif|else|for|while|def|class|"
    r"try|except|finally|raise|with|yield|assert|break|continue|pass)\b)",
)


def _split_collapsed_statements(text: str) -> str:
    r"""Insert newlines where Python statements are concatenated on one line.

    LLMs sometimes drop the ``\n`` between two statements but keep the
    indentation whitespace, producing e.g.::

        vtt = x            return Response(vtt)

    This detects ``<code>    <keyword>`` (4+ spaces between code and a
    Python keyword) and inserts a newline before the indentation block.
    """
    return _COLLAPSED_STMT_RE.sub(r"\n\1\2", text)


def normalize_edit_text(text: str) -> str:
    """Apply newline normalization passes to LLM-generated code.

    1. ``_unescape_collapsed_code``:  literal ``\\n`` → real newlines outside strings
    2. ``_escape_broken_strings``:    real newlines  → ``\\n`` inside string literals
    3. ``_split_collapsed_statements``:  detect 4+ space gaps before keywords → newlines
    """
    result = _escape_broken_strings(_unescape_collapsed_code(text))
    return _split_collapsed_statements(result)


# ---------------------------------------------------------------------------
# Environment error classification for smart verification
# ---------------------------------------------------------------------------

# Patterns that indicate environment/infra problems, NOT code bugs
_ENV_ERROR_PATTERNS = [
    "PermissionError",
    "Permission denied",
    "Errno 13",
    "collection error",
    "CollectionError",
    "ERROR collecting",
    "import file mismatch",
    "no module named",
    "ModuleNotFoundError",
    "ImportError",
    "FileNotFoundError: [Errno 2]",
    "OSError: [Errno",
    "socket.error",
    "ConnectionRefusedError",
    "TimeoutError",
]


def classify_verification_errors(
    errors: list[str],
) -> tuple[list[str], list[str]]:
    """Split errors into (code_errors, env_errors).

    Environment errors (permissions, missing modules, connection issues)
    are separated from genuine code/syntax bugs.
    """
    code_errors: list[str] = []
    env_errors: list[str] = []
    for err in errors:
        err_lower = err.lower()
        if any(pat.lower() in err_lower for pat in _ENV_ERROR_PATTERNS):
            env_errors.append(err)
        # pytest collection error summary lines: "ERROR path/to/test.py"
        elif re.match(r"^ERROR\s+\S+\.py\s*$", err.strip()):
            env_errors.append(err)
        else:
            code_errors.append(err)
    return code_errors, env_errors


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

VALID_ACTIONS = frozenset(
    {"read_file", "edit_file", "create_file", "run_command", "delete_file",
     "analyze", "grep", "glob"}
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


@dataclass
class Proposal:
    """A single improvement/change proposed by the agent after analysis."""
    index: int
    title: str
    description: str
    file: str
    priority: str = "medium"  # low / medium / high


class StepScratchpad:
    """Rolling context accumulator — tracks what the agent has done.

    Provides compact summaries of step results so the LLM can reason
    about prior actions in heal and edit-generation calls without
    needing full conversation history.
    """

    def __init__(self, max_entries: int = 20) -> None:
        self._entries: list[str] = []
        self._max = max_entries

    def record(self, step: AgentStep, success: bool,
               output: str = "", error: str = "") -> None:
        """Record a step result as a compact one-liner."""
        icon = "✓" if success else "✗"
        detail = output[:120] if success else error[:120]
        # Strip newlines for compact display
        detail = detail.replace("\n", " ").strip()
        entry = f"{icon} {step.action}({step.target}): {detail}"
        self._entries.append(entry)
        # FIFO — drop oldest when over budget
        if len(self._entries) > self._max:
            self._entries = self._entries[-self._max:]

    def to_context(self, last_n: int = 10) -> str:
        """Return last N entries as context string for LLM prompts."""
        if not self._entries:
            return ""
        entries = self._entries[-last_n:]
        return "AGENT PROGRESS SO FAR:\n" + "\n".join(entries)

    def status_report(self, changed_files: list[str] | None = None,
                      errors: list[str] | None = None,
                      next_step: str = "") -> str:
        """Compact STATUS line for progress tracking."""
        files_str = ", ".join(changed_files) if changed_files else "none"
        err_str = str(len(errors)) if errors else "0"
        return (
            f"STATUS: [Modified: {files_str}] | "
            f"[Errors: {err_str}] | "
            f"[Next: {next_step or 'done'}]"
        )

    def clear(self) -> None:
        """Reset the scratchpad for a new plan execution."""
        self._entries.clear()

    @property
    def entries(self) -> list[str]:
        return list(self._entries)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = """\
Precise coding planner. Produce MINIMAL step sequences — no filler, no redundancy.

Think before acting (internal reasoning goes in "reasoning" field):
- THOUGHT: What files are involved? What's the data flow?
- CONSTRAINT: Never edit without reading first. Never guess paths.
- STRATEGY: Fewest steps that achieve the goal correctly.

Return JSON:
{
  "task": "<concise restatement>",
  "reasoning": "<your thought process — what you checked, why this approach>",
  "steps": [
    {"action": "<read_file|edit_file|create_file|run_command|delete_file|analyze|grep|glob>",
     "target": "<file path, shell command, regex pattern, or glob pattern>",
     "description": "<what this step achieves>"}
  ]
}

Hard rules:
- 8 allowed actions ONLY: read_file, edit_file, create_file, run_command, delete_file, analyze, grep, glob.
- ALWAYS read_file BEFORE edit_file. No exceptions.
- Analysis tasks (questions, reviews, "what to improve") → read_file + analyze + grep/glob ONLY. No mutations.
- grep: target=regex, description can be JSON {"path":"dir","glob":"*.py"}.
- glob: target=pattern (e.g. "**/*.py"), description can be JSON {"path":"dir"}.
- Paths are RELATIVE to project root. Use the file tree in context — never invent paths.
- edit_file description: plain English explaining the change. Optionally JSON {"file":"path","old":"...","new":"..."}.
- Minimize steps. If 3 steps suffice, don't produce 7.
- Return ONLY valid JSON. No markdown fences. No commentary.
"""

_HEAL_SYSTEM = """\
Error recovery agent. Your prior approach failed. Fix it NOW — different strategy.

Schema:
{
  "task": "<fix description>",
  "reasoning": "<root cause analysis — WHY it failed, not just WHAT failed>",
  "steps": [...]
}

Protocol:
1. DIAGNOSE: Read the "AGENT PROGRESS" section. Identify the pattern of failure.
2. PIVOT: If the same file/approach failed twice → CHANGE STRATEGY ENTIRELY.
   Different function, different file, alternative algorithm.
3. EXECUTE: read_file FIRST (always), then edit_file with correct content.
4. SCOPE: Fix ONLY errors your edits introduced. Do NOT fix pre-existing lint
   warnings or unused imports that existed before your changes.

Anti-patterns (INSTANT FAILURE if you do these):
- Repeating an edit that returned "text not found" → FORBIDDEN.
- Editing a file listed in "FILES ALREADY MODIFIED" without re-reading it → FORBIDDEN.
- Producing the same plan as a previous failed attempt → FORBIDDEN.
- Installing a package that doesn't exist (use stdlib alternatives) → FORBIDDEN.
- Generating non-JSON output → FORBIDDEN.
- Fixing lint errors you did not introduce (F401 unused import etc.) → FORBIDDEN.

Valid actions: read_file, edit_file, create_file, run_command, delete_file, analyze, grep, glob.
Each step MUST have "action", "target", "description" fields.
Return ONLY valid JSON. No markdown fences.
"""

_PROPOSE_SYSTEM = """\
Senior code reviewer and programming partner. After analyzing the code, \
produce a list of concrete improvement proposals. Write each proposal as if \
explaining it to a colleague — natural language, not robotic bullet points.

Return JSON:
{
  "proposals": [
    {
      "title": "<short, descriptive title>",
      "description": "<explain what to change, why it matters, and how — mention \
specific function names, patterns, and tradeoffs in natural language>",
      "file": "<relative file path that needs to be changed>",
      "priority": "<high|medium|low>"
    }
  ]
}

Guidelines:
- Order by priority (high first).
- Be specific: mention function names, line ranges, patterns.
- Each proposal should be independently implementable.
- Descriptions should read like a senior engineer's review comment, not a ticket template.
- The "file" field MUST contain the relative path to the main file to change.
- Return ONLY valid JSON, no markdown fences.
- Respond in the same language as the user's question.
"""

_REPLAN_EDIT_SYSTEM = """\
You are a precise code editor. You will be given:
1. The ACTUAL content of a source file.
2. An INTENDED edit (old/new text that failed to match).

Your job: find the correct text span in the actual file that corresponds
to the intended "old" text, then return the corrected edit as JSON:
{"old": "<exact text from the file to replace>", "new": "<replacement text>"}

RULES:
- The "old" field MUST be copied character-for-character from the actual file
  content — including indentation, whitespace, and blank lines.
- The "new" field should achieve the same intent as the original replacement.
- For INSERTIONS: "old" must contain the existing lines around the insertion
  point, and "new" must contain those same lines with the new code inserted.
- Return ONLY the JSON object, no markdown fences or explanation.
- If the intended edit does not correspond to any part of the file, return:
  {"old": "", "new": "", "error": "no matching code found"}
"""

_IMPLEMENT_SYSTEM = """\
Implementation engine. Given proposals, produce the execution plan.

Return JSON:
{
  "task": "<implementation summary>",
  "reasoning": "<approach — what you'll change and why>",
  "steps": [
    {"action": "read_file", "target": "path", "description": "Read file before editing"},
    {"action": "edit_file", "target": "path", "description": "Detailed description: what to add/change and WHERE (which function, after which line/statement)"}
  ]
}

Rules:
- read_file BEFORE edit_file. Always.
- Use grep/glob when unsure of exact paths or positions.
- edit_file description MUST be specific: name the function, the exact location (e.g. "after the ffprobe check"), and what code to add/modify.
- For insertions: describe WHAT to insert and WHERE (between which existing lines).
- Paths relative to project root.
- Return ONLY valid JSON. No fences.
"""

_GENERATE_EDIT_SYSTEM = """\
Precise code editor. Given file content (with line numbers) + change description, return ONE edit as JSON.

Output format:
{"old": "<exact text copied from the file>", "new": "<replacement text>"}

CRITICAL RULES:
1. "old" MUST be copied CHARACTER-FOR-CHARACTER from the file — whitespace, indentation, blank lines must be exact.
2. "old" should include 2-3 context lines around the change point for unique matching.
3. "new" contains the same context lines with the change applied.
4. Do NOT include line numbers in "old" or "new" — they are only for reference.

FOR INSERTIONS (adding new code):
- Copy the surrounding lines (before and after the insertion point) into "old".
- In "new", include those SAME surrounding lines with the new code inserted between them.
- Example — to add 'mediainfo' after the 'ffprobe' line:
  {"old": "        'ffprobe': shutil.which('ffprobe') is not None,\\n    }", "new": "        'ffprobe': shutil.which('ffprobe') is not None,\\n        'mediainfo': shutil.which('mediainfo') is not None,\\n    }"}

FOR MODIFICATIONS (changing existing code):
- Copy the lines that need changing into "old".
- Put the modified version in "new".

If the change is not applicable: {"old": "", "new": "", "error": "reason"}
ONLY JSON. No markdown fences. No explanation.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class PlanActVerifyAgent:
    """Structured Plan → Act → Verify agent backed by Ollama.

    Supports optional **architect mode**: a reasoning model (e.g. deepseek-r1)
    handles planning/analysis while an editor model (e.g. qwen2.5-coder)
    handles code edits.  When ``architect_model`` is set, all LLM calls
    for planning, healing, analysis, and proposals use it instead of the
    default ``model``.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "qwen2.5-coder:14b-instruct-q6_K",
        project_root: str = ".",
        max_heal_iterations: int = 3,
        num_ctx: int = 32768,
        confirm_callback: Callable[[str, str], Any] | None = None,
        architect_model: str | None = None,
        architect_num_ctx: int | None = None,
    ) -> None:
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._architect_model = architect_model
        self._architect_num_ctx = architect_num_ctx or num_ctx
        self._project_root = Path(project_root).resolve()
        self._max_heal = max_heal_iterations
        self._num_ctx = num_ctx
        self._terminal = TerminalTool(
            working_dir=str(self._project_root),
            timeout=60,
            require_confirm=True,
            confirm_callback=confirm_callback,
        )
        self._grep = GrepTool(project_root=str(self._project_root))
        self._glob = GlobTool(project_root=str(self._project_root))

        # Persistent HTTP client for Ollama API (connection pooling)
        self._http_client: httpx.AsyncClient | None = None
        self._http_timeout = httpx.Timeout(
            connect=30.0, read=600.0, write=30.0, pool=30.0,
        )

        # Step context accumulator — tracks file contents read during plan
        # execution so that subsequent edit steps can access them without
        # re-reading and can include them in LLM prompts for better edits.
        self._step_file_cache: dict[str, str] = {}

        # Rolling scratchpad — compact log of what agent did for LLM context
        self._scratchpad = StepScratchpad(max_entries=20)

        # Lint baseline — captures pre-existing lint errors before agent edits
        # so verify() only reports NEW errors introduced by the agent.
        self._lint_baseline: set[str] = set()

    @property
    def architect_model(self) -> str | None:
        """Return the current architect model (None if not set)."""
        return self._architect_model

    @architect_model.setter
    def architect_model(self, value: str | None) -> None:
        self._architect_model = value

    # -- HTTP client lifecycle -----------------------------------------------

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Return the persistent HTTP client, creating one if needed."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self._base_url, timeout=self._http_timeout,
            )
        return self._http_client

    async def close(self) -> None:
        """Close the persistent HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

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
                          *, force_json: bool = True,
                          on_token: Callable[[str], Any] | None = None,
                          use_architect: bool = False) -> str:
        """Call Ollama POST /api/chat and return content.

        When *on_token* is provided, streams the response and calls the
        callback for each token chunk (allows live "thinking" display).
        Uses a long read timeout (10 min) to support large quantised models.

        When *use_architect* is True and an architect model is configured,
        routes the request to the architect (reasoning) model instead.
        """
        model = self._model
        num_ctx = self._num_ctx
        if use_architect and self._architect_model:
            model = self._architect_model
            num_ctx = self._architect_num_ctx
            logger.debug("Using architect model: %s (ctx: %d)", model, num_ctx)

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": on_token is not None,
            "options": {"num_ctx": num_ctx},
        }
        if force_json:
            payload["format"] = "json"

        client = await self._get_http_client()

        if on_token is None:
            # Non-streaming (original path)
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"]

        # Streaming path — yield tokens via callback
        chunks: list[str] = []
        async with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed JSON line in stream: %s", line[:200])
                    continue
                token = obj.get("message", {}).get("content", "")
                if token:
                    chunks.append(token)
                    result = on_token(token)
                    if asyncio.iscoroutine(result):
                        await result
        return "".join(chunks)

    @staticmethod
    def _strip_json_fences(raw: str) -> str:
        """Strip markdown code fences from JSON responses."""
        text = raw.strip()
        if text.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]
            # Remove closing fence
            if text.rstrip().endswith("```"):
                text = text.rstrip()
                text = text[:-3].rstrip()
        return text

    @staticmethod
    def _parse_plan(raw: str, task: str) -> AgentPlan:
        """Parse JSON response into an AgentPlan, validating actions.

        Tolerant of LLM field naming variations — small models often use
        'command', 'task', 'details' etc. instead of the canonical
        'action', 'target', 'description'.
        """
        obj = safe_parse_json(raw)
        steps: list[AgentStep] = []
        for i, s in enumerate(obj.get("steps", [])):
            # --- Extract action (try many field names) ---
            action = ""
            for key in ("action", "step", "command", "task", "type",
                        "operation", "tool"):
                val = s.get(key, "")
                if val and isinstance(val, str):
                    candidate = val.strip().lower()
                    words = candidate.split()
                    if len(words) <= 3:
                        action = candidate
                        break
                    # Long prose in action field — try to extract the
                    # first verb/word and use it as a short action hint
                    if len(words) > 3 and words[0]:
                        action = words[0]
                        break

            # Normalise common LLM aliases to valid actions
            _action_aliases = {
                "install": "run_command",
                "open": "read_file",
                "locate": "grep",
                "find": "glob",
                "search": "grep",
                "review": "analyze",
                "check": "analyze",
                "inspect": "analyze",
                "examine": "analyze",
                "identify": "analyze",
                "look": "read_file",
                "view": "read_file",
                "modify": "edit_file",
                "update": "edit_file",
                "change": "edit_file",
                "fix": "edit_file",
                "implement": "edit_file",
                "complete": "edit_file",
                "add": "edit_file",
                "optimize": "edit_file",
                "refactor": "edit_file",
                "replace": "edit_file",
                "resolve": "edit_file",
                "correct": "edit_file",
                "patch": "edit_file",
                "rewrite": "edit_file",
                "configure": "edit_file",
                "write": "create_file",
                "remove": "delete_file",
                "execute": "run_command",
                "shell": "run_command",
                "test": "run_command",
                "run": "run_command",
                "re-run": "run_command",
                "save": "",
                "commit": "",
                "navigate": "",
                "ensure": "analyze",
                "verify": "analyze",
                "validate": "analyze",
                "confirm": "analyze",
                "verify permissions": "run_command",
                "check test environment": "run_command",
                "update dependencies": "run_command",
            }
            action = _action_aliases.get(action, action)

            # If still no valid action, try inferring from prose fields
            if not action or action not in VALID_ACTIONS:
                inferred = PlanActVerifyAgent._infer_action_from_prose(s)
                if inferred:
                    action = inferred

            if not action or action not in VALID_ACTIONS:
                logger.warning("Skipping step %d with invalid action: %r", i, action)
                continue

            # --- Extract target (try many field names) ---
            target = ""
            for key in ("target", "path", "file", "file_path", "filename",
                        "pattern", "cmd", "directory"):
                val = s.get(key, "")
                if val and isinstance(val, str):
                    target = val
                    break
            # Fallback: extract file path from any field value
            if not target:
                target = PlanActVerifyAgent._extract_path_from_step(s)

            # --- Extract description (try many field names) ---
            description = ""
            for key in ("description", "details", "reasoning", "solution",
                        "explanation", "info", "what", "summary"):
                val = s.get(key, "")
                if val and isinstance(val, str):
                    description = val
                    break

            steps.append(
                AgentStep(
                    index=i,
                    action=action,
                    target=target,
                    description=description,
                )
            )
        return AgentPlan(
            task=obj.get("task", task),
            steps=steps,
            reasoning=obj.get("reasoning", ""),
        )

    @staticmethod
    def _infer_action_from_prose(step_dict: dict) -> str:
        """Try to infer a valid action from prose text in step fields.

        Small models often put descriptions in 'command' or 'task' fields
        instead of action names. This tries to extract a verb and map it.
        """
        # Gather all string values from the step
        texts = [v for v in step_dict.values() if isinstance(v, str)]
        combined = " ".join(texts).lower()

        _verb_map = [
            (r"\bread\b|\bopen\b|\bview\b|\blook at\b", "read_file"),
            (r"\banalyze\b|\banalysis\b|\breview\b|\binspect\b|\bidentify\b|\bensure\b|\bverify\b|\bvalidate\b|\bconfirm\b|\bcheck\b", "analyze"),
            (r"\bgrep\b|\bsearch\b|\bfind occurrences\b|\bscan\b", "grep"),
            (r"\bglob\b|\bfind files\b|\blist files\b|\blist directory\b", "glob"),
            (r"\bedit\b|\bmodify\b|\bchange\b|\bupdate\b|\breplace\b|\brefactor\b|\bfix\b|\bimplement\b|\bcomplete\b|\badd\b|\boptimize\b|\bresolve\b|\bcorrect\b|\bpatch\b|\brewrite\b|\bconfigure\b", "edit_file"),
            (r"\bcreate\b|\bwrite new\b|\bgenerate\b", "create_file"),
            (r"\bdelete\b|\bremove file\b", "delete_file"),
            (r"\brun\b|\bexecute\b|\binstall\b|\bchmod\b|\bpip\b|\bnpm\b|\btest\b|\bre-run\b|\bpermission\b", "run_command"),
        ]
        for pattern, action in _verb_map:
            if re.search(pattern, combined):
                return action
        return ""

    @staticmethod
    def _extract_path_from_step(step_dict: dict) -> str:
        """Try to extract a file path from any field in a step dict."""
        for val in step_dict.values():
            if not isinstance(val, str):
                continue
            # Match patterns like backend/file.py, src/module/file.ts
            m = re.search(
                r'(?:^|[\s`\'"])([a-zA-Z0-9_./-]+\.[a-zA-Z]{1,5})\b', val
            )
            if m:
                path = m.group(1)
                # Sanity check: must contain at least one directory separator
                # or be a known file extension
                if "/" in path or path.endswith((".py", ".js", ".ts", ".go",
                                                 ".rs", ".toml", ".json",
                                                 ".yaml", ".yml", ".md")):
                    return path
        return ""

    # -- plan ----------------------------------------------------------------

    async def plan(
        self, task: str, project_context: str = "",
        on_token: Callable[[str], Any] | None = None,
    ) -> AgentPlan:
        """Ask the model for a structured plan to accomplish *task*."""
        user_msg = f"Task:\n{task}"
        if project_context:
            user_msg += f"\n\nProject context:\n{project_context}"

        logger.info("Planning for task: %s", task)
        raw = await self._ollama_chat(_PLAN_SYSTEM, user_msg, on_token=on_token,
                                     use_architect=True)
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
                "grep": self._act_grep,
                "glob": self._act_glob,
            }.get(step.action)
            if handler is None:
                raise ValueError(f"Unknown action: {step.action!r}")
            output = await handler(step)
            step.status = "done"
            return ActionResult(step=step, success=True, output=output, error="")
        except PermissionError as exc:
            # Try to fix permissions automatically, then retry once
            target_path = step.target
            logger.warning(
                "PermissionError on step %d (%s) — attempting chmod fix",
                step.index, target_path,
            )
            try:
                chmod_result = await self._terminal.run_command(
                    f"chmod u+rw {target_path}", timeout=10,
                )
                if chmod_result.success:
                    logger.info("chmod succeeded, retrying step %d", step.index)
                    output = await handler(step)
                    step.status = "done"
                    return ActionResult(
                        step=step, success=True,
                        output=f"[auto-fixed permissions] {output}",
                        error="",
                    )
            except Exception as chmod_exc:
                logger.warning("chmod fix failed: %s", chmod_exc)
            step.status = "failed"
            return ActionResult(
                step=step, success=False, output="",
                error=(
                    f"PermissionError: {exc}. "
                    f"Try running: chmod u+rw {target_path}"
                ),
            )
        except Exception as exc:
            logger.error("Step %d failed: %s", step.index, exc)
            step.status = "failed"
            return ActionResult(step=step, success=False, output="", error=str(exc))

    async def _act_read_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        content = path.read_text(encoding="utf-8")
        # Cache for use by subsequent edit steps
        self._step_file_cache[step.target] = content
        return content

    async def _act_edit_file(self, step: AgentStep) -> str:
        # Handle both JSON string and dict descriptions
        edit_info = None
        if isinstance(step.description, dict):
            edit_info = step.description
        else:
            raw = step.description
            # Try parsing as JSON (the old-style format with old/new fields)
            try:
                parsed = safe_parse_json(raw)
                # Verify it actually has old/new fields — otherwise it's
                # just a JSON blob that the model produced (like {"path":"..."})
                if parsed.get("old") or parsed.get("old_text") or parsed.get("original"):
                    edit_info = parsed
            except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
                pass

            if edit_info is None:
                # Description is prose or malformed JSON — use LLM to
                # generate the precise edit from the file content
                logger.info(
                    "edit_file step has prose description for %s — "
                    "generating edit via LLM",
                    step.target,
                )
                edit_info = await self._generate_edit(step.target, raw)

        file_rel = edit_info.get("file", step.target)
        old_text = edit_info.get("old") or edit_info.get("old_text") or edit_info.get("original") or ""
        new_text = edit_info.get("new") or edit_info.get("new_text") or edit_info.get("replacement") or ""

        # Fix LLM double/under-escaped newlines (literal \n ↔ real newlines)
        old_text = normalize_edit_text(old_text)
        new_text = normalize_edit_text(new_text)

        if not old_text:
            raise ValueError("Missing 'old' field in edit_file description JSON")

        path = self._safe_path(file_rel)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # backup before editing
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)

        content = path.read_text(encoding="utf-8")

        def _write_and_cache(new_content: str, method: str) -> str:
            """Write file with pre-write syntax validation for Python files.

            If the file is .py, ast.parse() must pass before write. On failure
            the backup is restored and a structured error is raised so the agent
            can self-correct instead of writing broken code.
            """
            if file_rel.endswith(".py"):
                syntax_err = self._validate_python_syntax(new_content, file_rel)
                if syntax_err:
                    self._rollback_edit(file_rel, path, backup)
                    raise ValueError(
                        f"Pre-write validation failed — file NOT written.\n{syntax_err}"
                    )
            path.write_text(new_content, encoding="utf-8")
            # Update cache so subsequent edits to same file see current state
            self._step_file_cache[file_rel] = new_content
            return f"Edited {file_rel} ({method}, backup at {backup.name})"

        # 1. Try exact match
        if old_text in content:
            new_text = self._align_indentation(old_text, new_text)
            content = content.replace(old_text, new_text, 1)
            return _write_and_cache(content, "exact match")

        # 2. Try whitespace-normalised match
        match_pos = self._fuzzy_find(content, old_text)
        if match_pos is not None:
            start, end = match_pos
            matched_block = content[start:end]
            new_text = self._align_indentation(matched_block, new_text)
            content = content[:start] + new_text + content[end:]
            return _write_and_cache(content, "fuzzy match")

        # 3. LLM-assisted re-plan: ask the model to find the correct span
        logger.warning(
            "Exact and fuzzy match failed for %s — attempting LLM re-plan",
            file_rel,
        )
        try:
            corrected_old, corrected_new = await self._replan_edit(
                content, old_text, new_text, file_rel,
            )
            if corrected_old in content:
                corrected_new = self._align_indentation(
                    corrected_old, corrected_new,
                )
                content = content.replace(corrected_old, corrected_new, 1)
                return _write_and_cache(content, "LLM re-plan")
            # Corrected old still doesn't match — try fuzzy on the correction
            match_pos = self._fuzzy_find(content, corrected_old)
            if match_pos is not None:
                start, end = match_pos
                matched_block = content[start:end]
                corrected_new = self._align_indentation(
                    matched_block, corrected_new,
                )
                content = content[:start] + corrected_new + content[end:]
                return _write_and_cache(content, "LLM re-plan + fuzzy")
        except Exception as replan_exc:
            logger.warning("LLM re-plan failed for %s: %s", file_rel, replan_exc)

        raise ValueError(
            f"Text to replace not found in {file_rel} (neither exact nor fuzzy)"
        )

    @staticmethod
    def _fuzzy_find(content: str, needle: str) -> tuple[int, int] | None:
        """Find the best approximate match of *needle* inside *content*.

        Returns (start, end) character indices in *content* or None.
        Uses progressive thresholds: 70% → 55% → 40% with relative
        indentation normalization for better matching across indent levels.
        """
        import difflib

        needle_lines = needle.strip().splitlines()
        if not needle_lines:
            return None

        content_lines = content.splitlines(keepends=True)
        if not content_lines:
            return None

        n = len(needle_lines)

        def _normalize(lines: list[str]) -> str:
            """Normalize with relative indentation (aider-style)."""
            return "\n".join(l.strip() for l in lines)

        def _normalize_relative(lines: list[str]) -> str:
            """Normalize preserving relative indent structure."""
            if not lines:
                return ""
            stripped = [l.rstrip() for l in lines]
            # Find minimum indentation
            indents = [len(l) - len(l.lstrip()) for l in stripped if l.strip()]
            base = min(indents) if indents else 0
            return "\n".join(l[base:].rstrip() for l in stripped)

        # Try matching with multiple normalization strategies
        needle_norms = [
            _normalize(needle_lines),
            _normalize_relative(needle_lines),
        ]

        best_ratio = 0.0
        best_span: tuple[int, int] | None = None

        # Try windows of n-2 .. n+3 lines to handle off-by-a-few
        for delta in range(-2, 4):
            m = n + delta
            if m < 1 or m > len(content_lines):
                continue
            for i in range(len(content_lines) - m + 1):
                window_lines = content_lines[i:i + m]
                window_norms = [
                    _normalize(window_lines),
                    _normalize_relative(window_lines),
                ]
                # Try all normalization combinations
                for nn in needle_norms:
                    for wn in window_norms:
                        sm = difflib.SequenceMatcher(
                            None, nn, wn, autojunk=False
                        )
                        ratio = sm.ratio()
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_span = (i, i + m)

        # Progressive thresholds: try strict first, relax if needed
        # 70% = high confidence, 55% = moderate (e.g. variable renames),
        # 40% = structural match (same shape, different names)
        threshold = 0.70 if best_ratio >= 0.70 else (
            0.55 if best_ratio >= 0.55 else 0.40
        )

        if best_ratio < threshold or best_span is None:
            return None

        start_line, end_line = best_span
        start_pos = sum(len(l) for l in content_lines[:start_line])
        end_pos = sum(len(l) for l in content_lines[:end_line])
        logger.info(
            "Fuzzy match: ratio=%.2f (threshold=%.2f), lines %d-%d in file",
            best_ratio, threshold, start_line, end_line,
        )
        return start_pos, end_pos

    @staticmethod
    def _align_indentation(matched_text: str, new_text: str) -> str:
        """Ensure *new_text* has the same base indentation as *matched_text*.

        LLMs often produce replacement code with wrong base indentation.
        Uses common-line matching: finds lines present in both old and new,
        computes the indent delta from the first differing pair, then applies
        that correction uniformly to all lines in new_text.

        Falls back to first-line comparison and per-line fixups when
        common-line matching cannot determine a delta.
        """

        if "\n" not in new_text:
            return new_text

        lines = new_text.split("\n")
        non_empty = [(i, line) for i, line in enumerate(lines) if line.strip()]
        if len(non_empty) < 2:
            return new_text  # single meaningful line — nothing to align

        # Build indent map for matched_text: stripped_content → indent_len
        matched_indents: dict[str, int] = {}
        for mline in matched_text.split("\n"):
            s = mline.strip()
            if s and s not in matched_indents:
                matched_indents[s] = len(re.match(r"(\s*)", mline).group(1))

        # Strategy 1: common-line matching — find indent delta from a line
        # that exists in both old and new but has different indentation.
        delta: int | None = None
        for _, nline in non_empty:
            s = nline.strip()
            if s in matched_indents:
                actual = len(re.match(r"(\s*)", nline).group(1))
                expected = matched_indents[s]
                if actual != expected:
                    delta = actual - expected
                    break

        if delta is not None and delta != 0:
            if delta > 0:
                # Over-indented: shift all lines left
                result = []
                for line in lines:
                    if not line.strip():
                        result.append(line)
                    else:
                        cur = len(re.match(r"(\s*)", line).group(1))
                        result.append(" " * max(0, cur - delta) + line.lstrip())
                return "\n".join(result)
            else:
                # Under-indented: shift all lines right
                pad = " " * (-delta)
                return "\n".join(
                    pad + line if line.strip() else line for line in lines
                )

        # Strategy 2: first-line comparison fallback
        target_indent = re.match(r"(\s*)", matched_text).group(1)
        first_indent = re.match(r"(\s*)", non_empty[0][1]).group(1)
        fl_delta = len(first_indent) - len(target_indent)

        if fl_delta > 0:
            result = []
            for line in lines:
                if not line.strip():
                    result.append(line)
                else:
                    cur = len(re.match(r"(\s*)", line).group(1))
                    result.append(" " * max(0, cur - fl_delta) + line.lstrip())
            return "\n".join(result)

        if fl_delta < 0:
            pad = " " * (-fl_delta)
            return "\n".join(
                pad + line if line.strip() else line for line in lines
            )

        # Strategy 3 (Case B): first line correct but subsequent lines
        # under-indented (common when LLM drops indent on continuation lines)
        first_i = non_empty[0][0]
        if not target_indent:
            return new_text

        subsequent = non_empty[1:]
        min_sub_indent = min(
            len(re.match(r"(\s*)", line).group(1)) for _, line in subsequent
        )
        if min_sub_indent >= len(target_indent):
            return new_text  # all lines already have enough indentation

        sub_delta = target_indent[:len(target_indent) - min_sub_indent]
        result = []
        for i, line in enumerate(lines):
            if i == first_i or not line.strip():
                result.append(line)
            else:
                result.append(sub_delta + line)
        return "\n".join(result)

    async def _generate_edit(
        self, file_rel: str, change_description: str,
    ) -> dict:
        """Generate a precise {old, new} edit from a prose description.

        Uses the step file cache when available to avoid re-reading disk.
        Includes truncation warnings so the LLM knows if context is partial.
        """
        # Use cached content from a prior read_file step if available
        if file_rel in self._step_file_cache:
            file_content = self._step_file_cache[file_rel]
        else:
            path = self._safe_path(file_rel)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            file_content = path.read_text(encoding="utf-8")

        total_lines = file_content.count("\n") + 1
        max_chars = self._num_ctx * 3
        truncated = file_content[:max_chars]
        is_truncated = len(file_content) > max_chars
        shown_lines = truncated.count("\n") + 1

        # Add line numbers for LLM reference (NOT to be included in old/new)
        numbered_lines = []
        for i, line in enumerate(truncated.splitlines(keepends=True), 1):
            numbered_lines.append(f"{i:4d} | {line}")
        numbered_content = "".join(numbered_lines)

        # Build prompt with file content and truncation warning
        parts = [f"FILE: {file_rel} ({total_lines} lines)"]
        if is_truncated:
            parts.append(
                f"⚠️ FILE TRUNCATED: showing lines 1-{shown_lines} of {total_lines}. "
                f"If the code to change is not visible, return: "
                f'{{"old": "", "new": "", "error": "code beyond visible range"}}'
            )
        parts.append(
            "Below is the file with line numbers for reference. "
            "Do NOT include line numbers (e.g. '  42 | ') in your old/new text — "
            "copy only the actual code."
        )
        parts.append(f"```\n{numbered_content}\n```")

        # Include scratchpad so LLM knows what steps already executed
        scratchpad_ctx = self._scratchpad.to_context(last_n=5)
        if scratchpad_ctx:
            parts.append(f"\n{scratchpad_ctx}")

        parts.append(f"\nCHANGE REQUESTED:\n{change_description}")
        parts.append(
            f"\nProduce the edit as JSON: "
            f'{{\"old\": \"exact text from file\", \"new\": \"replacement text\"}}'
        )

        user_msg = "\n".join(parts)

        logger.info("Generating edit for %s via LLM", file_rel)
        raw = await self._ollama_chat(
            _GENERATE_EDIT_SYSTEM, user_msg, use_architect=True,
        )
        edit = safe_parse_json(raw)

        error = edit.get("error", "")
        if error:
            raise ValueError(
                f"LLM could not generate edit for {file_rel}: {error}"
            )

        old_text = edit.get("old", "")
        if not old_text:
            # Retry once with explicit insertion guidance
            logger.warning(
                "Empty 'old' from LLM for %s — retrying with insertion hint",
                file_rel,
            )
            retry_msg = (
                f"{user_msg}\n\n"
                "⚠️ PREVIOUS ATTEMPT RETURNED EMPTY 'old'. This is wrong.\n"
                "For INSERTIONS: copy the lines AROUND the insertion point into 'old', "
                "then put those same lines WITH the new code inserted between them "
                "into 'new'. You MUST return a non-empty 'old' field.\n"
                "For MODIFICATIONS: copy the existing lines that need changing.\n"
                "Return the JSON now."
            )
            raw = await self._ollama_chat(
                _GENERATE_EDIT_SYSTEM, retry_msg, use_architect=True,
            )
            edit = safe_parse_json(raw)
            old_text = edit.get("old", "")
            if not old_text:
                raise ValueError(
                    f"LLM generated empty 'old' field for {file_rel} (after retry)"
                )

        return edit

    async def _replan_edit(
        self, file_content: str, old_text: str, new_text: str, file_rel: str,
    ) -> tuple[str, str]:
        """Ask the LLM to correct an edit using actual file content.

        Returns (corrected_old, corrected_new).
        Raises ValueError if the LLM cannot find a match.
        """
        # Truncate very large files to stay within context
        max_chars = self._num_ctx * 3
        total_lines = file_content.count("\n") + 1
        truncated = file_content[:max_chars]
        is_truncated = len(file_content) > max_chars
        shown_lines = truncated.count("\n") + 1

        trunc_note = ""
        if is_truncated:
            trunc_note = (
                f"\n⚠️ FILE TRUNCATED: showing lines 1-{shown_lines} "
                f"of {total_lines}.\n"
            )

        user_msg = (
            f"FILE: {file_rel} ({total_lines} lines){trunc_note}\n"
            f"```\n{truncated}\n```\n\n"
            f"INTENDED EDIT (old text did NOT match the file):\n"
            f"old: ```\n{old_text}\n```\n"
            f"new: ```\n{new_text}\n```\n\n"
            f"Find the correct span in the actual file and return corrected JSON."
        )

        logger.info("Re-planning edit for %s via LLM", file_rel)
        raw = await self._ollama_chat(_REPLAN_EDIT_SYSTEM, user_msg,
                                     use_architect=True)
        corrected = safe_parse_json(raw)

        corrected_old = corrected.get("old", "")
        corrected_new = corrected.get("new", "")
        error = corrected.get("error", "")

        # Fix LLM double/under-escaped newlines
        corrected_old = normalize_edit_text(corrected_old)
        corrected_new = normalize_edit_text(corrected_new)

        if error or not corrected_old:
            raise ValueError(
                f"LLM re-plan could not find matching code in {file_rel}: {error}"
            )

        return corrected_old, corrected_new

    async def _act_create_file(self, step: AgentStep) -> str:
        path = self._safe_path(step.target)
        file_rel = step.target

        # Validate Python syntax before creating the file
        if file_rel.endswith(".py"):
            syntax_err = self._validate_python_syntax(step.description, file_rel)
            if syntax_err:
                raise ValueError(
                    f"Refusing to create {file_rel} — syntax is invalid.\n{syntax_err}"
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(step.description, encoding="utf-8")
        self._step_file_cache[file_rel] = step.description
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

    async def _act_grep(self, step: AgentStep) -> str:
        """Search for a pattern in the codebase."""
        kwargs: dict[str, Any] = {"pattern": step.target}
        if step.description:
            try:
                extra = json.loads(self._strip_json_fences(step.description))
                if isinstance(extra, dict):
                    kwargs.update(extra)
            except (json.JSONDecodeError, ValueError):
                pass  # description is just human-readable text, not JSON
        result = await self._grep.execute(**kwargs)
        if not result.success:
            raise RuntimeError(result.error or "Grep failed")
        return result.output or "No matches found."

    async def _act_glob(self, step: AgentStep) -> str:
        """Find files matching a glob pattern."""
        kwargs: dict[str, Any] = {"pattern": step.target}
        if step.description:
            try:
                extra = json.loads(self._strip_json_fences(step.description))
                if isinstance(extra, dict):
                    kwargs.update(extra)
            except (json.JSONDecodeError, ValueError):
                pass
        result = await self._glob.execute(**kwargs)
        if not result.success:
            raise RuntimeError(result.error or "Glob failed")
        return result.output or "No files matched."

    # -- pre-write validation ------------------------------------------------

    @staticmethod
    def _validate_python_syntax(content: str, file_path: str) -> str | None:
        """Validate Python source before writing to disk.

        Returns None if valid, or a structured error message aimed at an LLM
        agent so it can self-correct without guessing.
        """
        try:
            ast.parse(content, filename=file_path)
            return None
        except SyntaxError as exc:
            return PlanActVerifyAgent._parse_error_for_agent(exc, file_path, content)

    @staticmethod
    def _parse_error_for_agent(
        exc: SyntaxError, file_path: str, content: str,
    ) -> str:
        """Convert a SyntaxError into an actionable instruction for the LLM.

        Instead of raw tracebacks the agent can't act on, this produces:
        - Exact line number and column
        - The offending line with a caret marker
        - 3 lines of surrounding context so the agent can orient itself
        """
        lineno = exc.lineno or 0
        col = exc.offset or 0
        msg = exc.msg or "unknown syntax error"

        lines = content.splitlines()
        # Context window: 3 lines before, the error line, 3 lines after
        start = max(0, lineno - 4)
        end = min(len(lines), lineno + 3)
        context_lines = []
        for i in range(start, end):
            prefix = ">>>" if i == lineno - 1 else "   "
            context_lines.append(f"{prefix} {i + 1:4d} | {lines[i]}")
            if i == lineno - 1 and col > 0:
                context_lines.append(f"          {' ' * (col - 1)}^")

        context_block = "\n".join(context_lines)
        return (
            f"SYNTAX ERROR in {file_path} at line {lineno}, col {col}: {msg}\n"
            f"{context_block}\n"
            f"FIX: Check indentation, brackets, and string delimiters near line {lineno}."
        )

    def _rollback_edit(self, file_rel: str, path: Path, backup: Path) -> None:
        """Restore file from backup and invalidate cache.

        Called when pre-write validation fails — guarantees the file on disk
        stays in its last-known-good state.
        """
        if backup.exists():
            shutil.copy2(backup, path)
            logger.info("Rolled back %s from backup", file_rel)
        # Invalidate cache so subsequent steps re-read from disk
        self._step_file_cache.pop(file_rel, None)

    # -- verify --------------------------------------------------------------

    def _detect_language(self) -> str:
        """Detect primary project language from marker files."""
        markers = {
            "python": ["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile"],
            "javascript": ["package.json", "tsconfig.json"],
            "go": ["go.mod"],
            "rust": ["Cargo.toml"],
        }
        for lang, files in markers.items():
            for f in files:
                if (self._project_root / f).exists():
                    return lang
        return "python"  # default fallback

    async def verify(self, language: str = "", changed_files: list[str] | None = None) -> VerifyResult:
        """Run tiered verification to catch errors efficiently.

        Tier 1: Syntax check on changed files only (fast, <5s)
        Tier 2: Lint changed files only (medium, <30s)
        Tier 3: Full test suite (slow, skipped if no test dir)

        Uses smart error classification: environment errors (PermissionError,
        CollectionError, etc.) are reported as warnings, not hard failures.
        """
        if not language:
            language = self._detect_language()

        errors: list[str] = []
        warnings: list[str] = []

        # ---- Tier 1: Syntax check changed files (fast) ----
        if changed_files and language == "python":
            syntax_errors = await self._check_python_syntax(changed_files)
            if syntax_errors:
                # Fail fast — no point running lint/tests with syntax errors
                return VerifyResult(
                    success=False, errors=syntax_errors, warnings=[]
                )

        # ---- Tier 2: Targeted lint on changed files ----
        verifiers_targeted = {
            "python": self._verify_python_lint,
            "javascript": self._verify_javascript,
            "go": self._verify_go,
            "rust": self._verify_rust,
        }
        verifier_lint = verifiers_targeted.get(language)
        if verifier_lint:
            lint_errors, lint_warnings = await verifier_lint(
                changed_files=changed_files
            )
            errors.extend(lint_errors)
            warnings.extend(lint_warnings)

        # ---- Tier 3: Full test suite (only if lint passes) ----
        if not errors and language == "python":
            test_errors, test_warnings = await self._verify_python_tests()
            errors.extend(test_errors)
            warnings.extend(test_warnings)

        # Smart classification: separate env errors from code errors
        code_errors, env_errors = classify_verification_errors(errors)

        if env_errors:
            logger.info(
                "Environment errors detected (%d) — demoting to warnings",
                len(env_errors),
            )
            for e in env_errors:
                warnings.append(f"[ENV] {e}")

        success = len(code_errors) == 0
        return VerifyResult(success=success, errors=code_errors, warnings=warnings)

    @staticmethod
    def _find_ruff() -> str | None:
        """Locate ruff binary — checks codator's own venv first, then PATH.

        When codator is invoked via its venv Python but the target project
        doesn't have ruff installed, the plain 'ruff' command fails. This
        resolves ruff from codator's own bin directory.
        """
        # 1. Same directory as the running Python interpreter
        venv_bin = Path(sys.executable).parent / "ruff"
        if venv_bin.exists():
            return str(venv_bin)
        # 2. System PATH
        system_ruff = shutil.which("ruff")
        if system_ruff:
            return system_ruff
        return None

    @staticmethod
    def _normalize_lint_entry(line: str) -> str | None:
        """Extract line-number-independent key from a concise ruff output line.

        Concise format: ``path:line:col: RULE message``
        Normalized:     ``path: RULE message``

        This ensures that pre-existing errors still match after edits shift
        line numbers.  Returns *None* for non-error lines (summaries, blanks).
        """
        m = re.match(r"^(.+?):\d+:\d+:\s+(.+)$", line)
        if m:
            return f"{m.group(1)}: {m.group(2)}"
        return None

    async def _capture_lint_baseline(self, files: list[str]) -> None:
        """Capture pre-existing lint errors for target files BEFORE editing.

        Stored in _lint_baseline so verify() can subtract them. This prevents
        the agent from being blamed for errors it didn't introduce.
        Uses --output-format=concise for stable, single-line-per-error output.
        """
        py_files = [f for f in files if f.endswith(".py")]
        if not py_files:
            return
        ruff = self._find_ruff()
        if not ruff:
            return
        targets = " ".join(py_files)
        result = await self._run_quiet(
            f"{ruff} check --output-format=concise {targets}"
        )
        if result and "not found" not in result and "No such file" not in result:
            for line in result.splitlines():
                normalized = self._normalize_lint_entry(line.strip())
                if normalized:
                    self._lint_baseline.add(normalized)
            if self._lint_baseline:
                logger.info(
                    "Lint baseline: %d pre-existing issues captured",
                    len(self._lint_baseline),
                )

    async def _verify_python_lint(
        self, changed_files: list[str] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Tier 2: ruff check — targeted to changed files when possible.

        Subtracts pre-existing errors captured in _lint_baseline so the agent
        is only held responsible for errors it introduced.
        Uses --output-format=concise for stable, single-line-per-error output
        and line-number-independent baseline matching.
        """
        errors: list[str] = []
        warnings: list[str] = []

        ruff = self._find_ruff()
        if not ruff:
            logger.info("ruff not available — skipping lint verification")
            return errors, warnings

        # Build ruff command — target changed files if available
        if changed_files:
            py_files = [f for f in changed_files if f.endswith(".py")]
            if not py_files:
                return errors, warnings
            targets = " ".join(py_files)
            cmd = f"{ruff} check --output-format=concise {targets}"
        else:
            cmd = f"{ruff} check --output-format=concise ."

        ruff_result = await self._run_quiet(cmd)
        if ruff_result is not None:
            if "not found" in ruff_result or "No such file" in ruff_result:
                logger.info("ruff not available — skipping lint verification")
            else:
                for line in ruff_result.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Normalize to strip line:col for baseline comparison
                    normalized = self._normalize_lint_entry(stripped)
                    if not normalized:
                        continue
                    # Skip pre-existing lint errors (captured before edits)
                    if normalized in self._lint_baseline:
                        continue
                    if ": W" in stripped or ": D" in stripped:
                        warnings.append(stripped)
                    else:
                        errors.append(stripped)

        return errors, warnings

    async def _verify_python_tests(self) -> tuple[list[str], list[str]]:
        """Tier 3: Run pytest if tests/ directory exists.

        Uses the project's own Python (venv or system) — NOT codator's
        sys.executable, which would lack the project's dependencies and
        produce false-positive import failures.
        """
        errors: list[str] = []
        warnings: list[str] = []

        tests_dir = self._project_root / "tests"
        backend_tests = self._project_root / "backend" / "tests"
        if not tests_dir.is_dir() and not backend_tests.is_dir():
            return errors, warnings

        # Find the project's own Python (prefer venv, then system)
        project_python: str | None = None
        for venv_name in (".venv", "venv", "env"):
            candidate = self._project_root / venv_name / "bin" / "python"
            if candidate.exists():
                project_python = str(candidate)
                break
        if not project_python:
            # No project venv — skip tests rather than use codator's Python
            # which would lack project dependencies
            logger.info("No project venv found — skipping test verification")
            return errors, warnings

        pytest_result = await self._run_quiet(
            f"{project_python} -m pytest --tb=short -q"
        )
        if pytest_result is not None and "not found" not in pytest_result:
            for line in pytest_result.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if "FAILED" in stripped or "ERROR" in stripped:
                    errors.append(stripped)
                elif "warning" in stripped.lower():
                    warnings.append(stripped)

        return errors, warnings

    async def _check_python_syntax(self, files: list[str]) -> list[str]:
        """Quick py_compile check on specific files — catches syntax errors fast."""
        syntax_errors: list[str] = []
        for rel_path in files:
            if not rel_path.endswith(".py"):
                continue
            full = self._safe_path(rel_path)
            if not full.exists():
                continue
            result = await self._run_quiet(
                f"{sys.executable} -m py_compile {full}"
            )
            if result and ("SyntaxError" in result or "Error" in result):
                syntax_errors.append(f"Syntax error in {rel_path}: {result.strip()}")
                logger.warning("Syntax error detected in %s", rel_path)
        return syntax_errors

    async def _verify_javascript(self, changed_files: list[str] | None = None) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        # --- eslint (skip if not installed) ---
        eslint_result = await self._run_quiet("npx eslint . --format compact --no-error-on-unmatched-pattern 2>&1")
        if eslint_result is not None:
            if "not found" not in eslint_result and "Cannot find module" not in eslint_result:
                for line in eslint_result.splitlines():
                    stripped = line.strip()
                    if not stripped or stripped.startswith("(") or "problem" in stripped.lower():
                        continue
                    if "Error" in stripped:
                        errors.append(stripped)
                    elif "Warning" in stripped:
                        warnings.append(stripped)

        # --- npm test / npx jest (if package.json has test script) ---
        pkg_json = self._project_root / "package.json"
        if pkg_json.exists():
            import json
            try:
                pkg = json.loads(pkg_json.read_text(encoding="utf-8"))
                if "test" in pkg.get("scripts", {}):
                    test_result = await self._run_quiet("npm test -- --ci 2>&1")
                    if test_result is not None:
                        for line in test_result.splitlines():
                            stripped = line.strip()
                            if "FAIL" in stripped or "ERR!" in stripped:
                                errors.append(stripped)
            except (json.JSONDecodeError, OSError):
                pass

        return errors, warnings

    async def _verify_go(self, changed_files: list[str] | None = None) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        # --- go vet ---
        vet_result = await self._run_quiet("go vet ./... 2>&1")
        if vet_result is not None and "not found" not in vet_result:
            for line in vet_result.splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    errors.append(stripped)

        # --- go test ---
        test_result = await self._run_quiet("go test ./... -short 2>&1")
        if test_result is not None and "not found" not in test_result:
            for line in test_result.splitlines():
                stripped = line.strip()
                if "FAIL" in stripped:
                    errors.append(stripped)
                elif "warning" in stripped.lower():
                    warnings.append(stripped)

        return errors, warnings

    async def _verify_rust(self, changed_files: list[str] | None = None) -> tuple[list[str], list[str]]:
        errors: list[str] = []
        warnings: list[str] = []

        # --- cargo check ---
        check_result = await self._run_quiet("cargo check 2>&1")
        if check_result is not None and "not found" not in check_result:
            for line in check_result.splitlines():
                stripped = line.strip()
                if stripped.startswith("error"):
                    errors.append(stripped)
                elif stripped.startswith("warning"):
                    warnings.append(stripped)

        # --- cargo test (only if cargo check passed) ---
        if not errors:
            test_result = await self._run_quiet("cargo test 2>&1")
            if test_result is not None:
                for line in test_result.splitlines():
                    stripped = line.strip()
                    if "FAILED" in stripped or stripped.startswith("error"):
                        errors.append(stripped)

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
        on_token: Callable[[str], Any] | None = None,
        previous_attempts: list[str] | None = None,
        changed_files: list[str] | None = None,
    ) -> AgentPlan:
        """Ask the model to produce a fix plan for *errors*.

        Includes history of previous failed attempts, scratchpad context,
        and list of already-modified files to avoid repeating the same fixes.
        """
        user_msg = (
            f"Original task:\n{original_task}\n\n"
            f"Errors:\n" + "\n".join(errors)
        )
        if project_context:
            user_msg += f"\n\nProject context:\n{project_context}"

        # Inject scratchpad — gives LLM awareness of what was already done
        scratchpad_ctx = self._scratchpad.to_context()
        if scratchpad_ctx:
            user_msg += f"\n\n{scratchpad_ctx}"

        # Tell LLM which files were already modified
        if changed_files:
            user_msg += (
                "\n\n📝 FILES ALREADY MODIFIED (content has changed — re-read before editing):\n"
                + "\n".join(f"  - {f}" for f in changed_files)
            )

        if previous_attempts:
            user_msg += (
                "\n\n⚠️ PREVIOUS FAILED ATTEMPTS (do NOT repeat these):\n"
                + "\n---\n".join(previous_attempts)
                + "\n\nYou MUST try a DIFFERENT approach this time."
            )

        logger.info("Self-healing: %d errors to fix (prev attempts: %d)",
                    len(errors), len(previous_attempts or []))
        raw = await self._ollama_chat(_HEAL_SYSTEM, user_msg, on_token=on_token,
                                     use_architect=True)
        return self._parse_plan(raw, f"fix: {original_task}")

    # -- interactive flow (Claude-like) --------------------------------------

    async def analyze_and_propose(
        self,
        task: str,
        project_context: str = "",
        on_step: Callable[[str, str], Any] | None = None,
        on_token: Callable[[str], Any] | None = None,
    ) -> tuple[list[Proposal], list[ActionResult]]:
        """Analyze the project and return structured proposals for the user.

        This is the first half of the interactive flow:
        1. Plan a read-only analysis
        2. Execute it (read files, analyze)
        3. Ask the model to produce concrete proposals
        4. Return proposals + action log
        """

        async def _notify(description: str, status: str) -> None:
            if on_step is not None:
                result = on_step(description, status)
                if asyncio.iscoroutine(result):
                    await result

        # Force analysis-only plan
        analysis_task = (
            f"Analyze the following request (read and analyze only, NO edits):\n{task}"
        )
        await _notify("Planning analysis…", "started")
        # Don't stream plan tokens — they're internal JSON, not user-facing text
        analysis_plan = await self.plan(
            analysis_task, project_context,
        )
        await _notify(
            f"Analysis plan ready — {len(analysis_plan.steps)} steps", "done"
        )

        # Execute the read-only steps
        all_actions: list[ActionResult] = []
        for step in analysis_plan.steps:
            # Only allow read-only actions
            if step.action in ("edit_file", "create_file", "delete_file", "run_command"):
                logger.warning("Skipping mutating step in analysis: %s", step.action)
                continue
            await _notify(step.description, "running")
            result = await self.act(step)
            all_actions.append(result)
            await _notify(
                step.description, "done" if result.success else "failed"
            )

        # Gather analysis output
        analysis_output = "\n".join(
            a.output for a in all_actions if a.success and a.output
        )

        # Ask model to produce structured proposals (no streaming — output is
        # machine-readable JSON that gets parsed and displayed as rich panels)
        await _notify("Generating proposals…", "running")
        propose_prompt = (
            f"User's request:\n{task}\n\n"
            f"Code analysis results:\n{analysis_output[:12000]}\n\n"
            f"Project context:\n{project_context[:4000]}"
        )
        raw = await self._ollama_chat(
            _PROPOSE_SYSTEM, propose_prompt,
            use_architect=True,
        )
        proposals = self._parse_proposals(raw)
        await _notify(f"{len(proposals)} proposals ready", "done")

        return proposals, all_actions

    @staticmethod
    def _parse_proposals(raw: str) -> list[Proposal]:
        """Parse JSON response into a list of Proposals.

        Tolerant of LLM field naming variations at every level.
        """
        obj = safe_parse_json(raw)
        # Find the list of proposals — try multiple possible keys
        items: list[dict] = []
        for key in ("proposals", "highlights", "improvements", "suggestions",
                     "changes", "items", "recommendations"):
            if key in obj and isinstance(obj[key], list):
                items = obj[key]
                break
        if not items and isinstance(obj, list):
            items = obj
        if not items:
            logger.warning("Proposals list is empty or null after parsing")

        proposals: list[Proposal] = []
        for i, p in enumerate(items):
            if not isinstance(p, dict):
                continue
            # Title: try title → name → first 80 chars of description
            title = (
                p.get("title", "")
                or p.get("name", "")
                or p.get("description", "")[:80]
            ).strip()
            if not title:
                continue
            # Description: try description → detail → summary → code_snippet
            description = (
                p.get("description", "")
                or p.get("detail", "")
                or p.get("summary", "")
                or p.get("code_snippet", "")
                or title
            ).strip()
            # Priority: try priority → impact → severity (handle int or str)
            raw_priority = (
                p.get("priority", "")
                or p.get("impact", "")
                or p.get("severity", "")
                or "medium"
            )
            # Map numeric impact (1=low, 2=medium, 3=high) to string
            if isinstance(raw_priority, int):
                priority = {1: "low", 2: "medium", 3: "high"}.get(
                    raw_priority, "medium"
                )
            else:
                priority = str(raw_priority).strip().lower()
            proposals.append(Proposal(
                index=i,
                title=title,
                description=description,
                file=p.get("file", p.get("file_path", p.get("path", ""))),
                priority=priority,
            ))
        return proposals

    async def implement_proposals(
        self,
        proposals: list[Proposal],
        task: str,
        project_context: str = "",
        on_step: Callable[[str, str], Any] | None = None,
        on_token: Callable[[str], Any] | None = None,
        use_git_transaction: bool = True,
    ) -> AgentResult:
        """Implement selected proposals — the second half of the interactive flow.

        Takes the user-selected proposals and creates an implementation plan,
        then runs the full Act → Verify → Heal cycle.
        """

        async def _notify(description: str, status: str) -> None:
            if on_step is not None:
                result = on_step(description, status)
                if asyncio.iscoroutine(result):
                    await result

        # Build implementation prompt from selected proposals
        proposals_text = "\n".join(
            f"{i+1}. [{p.priority.upper()}] {p.title}\n"
            f"   File: {p.file}\n"
            f"   {p.description}"
            for i, p in enumerate(proposals)
        )
        impl_task = (
            f"Implement the following changes:\n\n{proposals_text}\n\n"
            f"Original request: {task}"
        )

        impl_prompt = impl_task
        if project_context:
            impl_prompt += f"\n\nProject context:\n{project_context[:4000]}"

        # Generate implementation plan
        await _notify("Planning implementation…", "started")
        # Don't stream — this generates internal JSON, not user-facing text
        raw = await self._ollama_chat(
            _IMPLEMENT_SYSTEM, impl_prompt,
            use_architect=True,
        )
        current_plan = self._parse_plan(raw, impl_task)
        current_plan = self._ensure_reads_before_edits(current_plan)
        await _notify(
            f"Implementation plan — {len(current_plan.steps)} steps", "done"
        )

        # Clear file cache and scratchpad for fresh implementation
        self._step_file_cache.clear()
        self._scratchpad.clear()

        # ---- Lint baseline (capture pre-existing errors before edits) ----
        edit_targets = list({
            s.target for s in current_plan.steps
            if s.action in ("edit_file", "create_file") and s.target
        })
        if edit_targets:
            self._lint_baseline.clear()
            await self._capture_lint_baseline(edit_targets)

        # Git transaction
        original_branch: str | None = None
        agent_branch: str | None = None
        if use_git_transaction:
            original_branch, agent_branch = self._git_create_branch()

        try:
            # Execute steps
            all_actions: list[ActionResult] = []
            heal_iterations = 0
            prev_error_count: int | None = None  # no-progress detection
            # Accumulate ALL modified files across iterations so that
            # verify() always targets the right set (even if a heal
            # iteration's edit fails, the file is still dirty from before).
            all_changed_files: set[str] = set()

            for iteration in range(1 + self._max_heal):
                changed_files: list[str] = []
                for step in current_plan.steps:
                    await _notify(step.description, "running")
                    result = await self.act(step)
                    all_actions.append(result)
                    self._scratchpad.record(
                        step, result.success,
                        output=result.output, error=result.error,
                    )
                    status = "done" if result.success else "failed"
                    await _notify(step.description, status)
                    if result.success and step.action in ("edit_file", "create_file"):
                        changed_files.append(step.target)
                all_changed_files.update(changed_files)

                # Use accumulated set so verify never falls back to "."
                verify_targets = list(all_changed_files) if all_changed_files else changed_files
                await _notify("Verifying…", "running")
                verification = await self.verify(changed_files=verify_targets)
                await _notify(
                    "Verification " + ("passed" if verification.success else "failed"),
                    "done" if verification.success else "failed",
                )

                if verification.success:
                    agent_result = AgentResult(
                        plan=current_plan,
                        actions=all_actions,
                        verification=verification,
                        heal_iterations=heal_iterations,
                        final_success=True,
                    )
                    break

                if iteration < self._max_heal:
                    # No-progress / explosion detection
                    current_error_count = len(verification.errors)
                    regression = (
                        prev_error_count is not None
                        and current_error_count >= prev_error_count
                    )
                    explosion = (
                        prev_error_count is not None
                        and prev_error_count > 0
                        and current_error_count >= prev_error_count * 3
                    )
                    if regression or explosion:
                        reason = "explosion" if explosion else "no progress"
                        logger.warning(
                            "No progress in implement_proposals: errors %d → %d (%s)",
                            prev_error_count or 0, current_error_count, reason,
                        )
                        await _notify(
                            f"No progress ({reason}, errors: {current_error_count}). Stopping.",
                            "failed",
                        )
                        agent_result = AgentResult(
                            plan=current_plan,
                            actions=all_actions,
                            verification=verification,
                            heal_iterations=heal_iterations,
                            final_success=False,
                        )
                        break
                    prev_error_count = current_error_count

                    heal_iterations += 1
                    await _notify(
                        f"Self-healing (attempt {heal_iterations})…", "running"
                    )
                    prev_attempts = []
                    if heal_iterations > 1:
                        for a in all_actions:
                            if not a.success:
                                prev_attempts.append(
                                    f"Attempt: {a.step.description}\n"
                                    f"Error: {a.error or 'unknown'}"
                                )
                    try:
                        current_plan = await self.self_heal(
                            verification.errors, impl_task, project_context,
                            previous_attempts=prev_attempts or None,
                            changed_files=changed_files,
                        )
                        current_plan = self._ensure_reads_before_edits(current_plan)
                        await _notify(
                            f"Heal plan — {len(current_plan.steps)} steps", "done"
                        )
                    except (json.JSONDecodeError, ValueError, KeyError) as exc:
                        logger.warning("Heal plan failed: %s", exc)
                        await _notify(f"Self-heal failed: {exc}", "failed")
                        agent_result = AgentResult(
                            plan=current_plan,
                            actions=all_actions,
                            verification=verification,
                            heal_iterations=heal_iterations,
                            final_success=False,
                        )
                        break
            else:
                agent_result = AgentResult(
                    plan=current_plan,
                    actions=all_actions,
                    verification=verification,
                    heal_iterations=heal_iterations,
                    final_success=False,
                )

        except Exception:
            if agent_branch and original_branch:
                self._git_rollback(original_branch, agent_branch)
            raise

        # Git merge/rollback
        if agent_branch and original_branch:
            if agent_result.final_success:
                self._git_merge(original_branch, agent_branch)
                await _notify("Git: merged changes", "done")
            else:
                self._git_rollback(original_branch, agent_branch)
                await _notify("Git: rolled back changes", "done")

        return agent_result

    async def run(
        self,
        task: str,
        project_context: str = "",
        on_step: Callable[[str, str], object] | None = None,
        on_token: Callable[[str], Any] | None = None,
        use_git_transaction: bool = True,
    ) -> AgentResult:
        """Execute the full Plan → Act → Verify (→ Heal) cycle.

        When *on_token* is provided, the model's raw output is streamed
        token-by-token (allows live "thinking" display).
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
                task, project_context, _notify, on_token=on_token,
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

    @staticmethod
    def _ensure_reads_before_edits(plan: AgentPlan) -> AgentPlan:
        """Auto-insert read_file steps before edit_file if file wasn't read yet.

        Models often forget the "read first" rule. This ensures the agent
        always has file content cached before attempting an edit.
        """
        read_targets: set[str] = set()
        fixed_steps: list[AgentStep] = []
        next_index = 0

        for step in plan.steps:
            if step.action in ("read_file", "analyze"):
                read_targets.add(step.target)
            elif step.action == "edit_file" and step.target not in read_targets:
                # Insert a read_file step before the edit
                read_step = AgentStep(
                    index=next_index,
                    action="read_file",
                    target=step.target,
                    description=f"Read {step.target} before editing",
                )
                fixed_steps.append(read_step)
                read_targets.add(step.target)
                next_index += 1
                logger.info(
                    "Auto-inserted read_file for %s before edit_file",
                    step.target,
                )

            step.index = next_index
            fixed_steps.append(step)
            next_index += 1

        return AgentPlan(
            task=plan.task,
            steps=fixed_steps,
            reasoning=plan.reasoning,
        )

    async def _run_inner(
        self,
        task: str,
        project_context: str,
        _notify: Callable,
        on_token: Callable[[str], Any] | None = None,
    ) -> AgentResult:
        """Core agent loop (extracted for git transaction wrapping)."""
        # Clear caches for fresh run
        self._step_file_cache.clear()
        self._scratchpad.clear()

        # ---- Plan ----------------------------------------------------------
        await _notify("Planning…", "started")
        # Don't stream plan — internal JSON, not user-facing text
        current_plan = await self.plan(task, project_context)
        current_plan = self._ensure_reads_before_edits(current_plan)
        await _notify(f"Plan ready — {len(current_plan.steps)} steps", "done")

        all_actions: list[ActionResult] = []
        heal_iterations = 0
        prev_error_count: int | None = None  # for no-progress detection

        # Detect analysis-only plans (no edits/creates/deletes/commands)
        _mutating_actions = {"edit_file", "create_file", "delete_file", "run_command"}
        is_analysis_only = not any(
            s.action in _mutating_actions for s in current_plan.steps
        )

        # ---- Lint baseline (capture pre-existing errors before any edits) ---
        if not is_analysis_only:
            edit_targets = list({
                s.target for s in current_plan.steps
                if s.action in ("edit_file", "create_file") and s.target
            })
            if edit_targets:
                self._lint_baseline.clear()
                await self._capture_lint_baseline(edit_targets)

        for iteration in range(1 + self._max_heal):
            # ---- Act -------------------------------------------------------
            changed_files: list[str] = []
            for step in current_plan.steps:
                await _notify(step.description, "running")
                result = await self.act(step)
                all_actions.append(result)
                # Record into scratchpad for LLM context
                self._scratchpad.record(
                    step, result.success,
                    output=result.output, error=result.error,
                )
                status = "done" if result.success else "failed"
                await _notify(step.description, status)
                if not result.success:
                    logger.warning(
                        "Step %d failed: %s", step.index, result.error
                    )
                # Track files modified by mutating actions
                if result.success and step.action in ("edit_file", "create_file"):
                    changed_files.append(step.target)

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
                        "Helpful coding assistant. Provide a clear, "
                        "actionable review of the code. Be specific about what "
                        "is good and what should be improved. "
                        "Respond in the same language as the user's question.",
                        summary_prompt,
                        force_json=False,
                        on_token=on_token,
                        use_architect=True,
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
                    verification = VerifyResult(success=False)

                return AgentResult(
                    plan=current_plan,
                    actions=all_actions,
                    verification=verification,
                    heal_iterations=0,
                    final_success=verification.success,
                )

            await _notify("Verifying…", "running")
            verification = await self.verify(changed_files=changed_files)
            await _notify(
                "Verification " + ("passed" if verification.success else "failed"),
                "done" if verification.success else "failed",
            )

            if verification.success:
                logger.info(
                    self._scratchpad.status_report(changed_files, None, "done")
                )
                return AgentResult(
                    plan=current_plan,
                    actions=all_actions,
                    verification=verification,
                    heal_iterations=heal_iterations,
                    final_success=True,
                )

            # ---- No-progress detection ------------------------------------
            current_error_count = len(verification.errors)
            # Abort if errors not decreasing OR if errors exploded (3x+ growth)
            regression = (
                prev_error_count is not None
                and current_error_count >= prev_error_count
            )
            explosion = (
                prev_error_count is not None
                and prev_error_count > 0
                and current_error_count >= prev_error_count * 3
            )
            if regression or explosion:
                reason = "explosion" if explosion else "no progress"
                status = self._scratchpad.status_report(
                    changed_files, verification.errors,
                    f"ABORTED — {reason}",
                )
                logger.warning(
                    "No progress in implement_proposals: errors %d → %d (%s). %s",
                    prev_error_count or 0, current_error_count, reason, status,
                )
                await _notify(
                    f"No progress ({reason}, errors: {current_error_count}). Stopping.",
                    "failed",
                )
                break
            prev_error_count = current_error_count

            # ---- Heal (if budget remains) ----------------------------------
            if iteration < self._max_heal:
                heal_iterations += 1
                logger.info(
                    self._scratchpad.status_report(
                        changed_files, verification.errors,
                        f"heal attempt {heal_iterations}",
                    )
                )
                await _notify(
                    f"Self-healing (attempt {heal_iterations})…", "running"
                )
                # Build history of previous attempts for context
                prev_attempts = []
                if heal_iterations > 1:
                    # Summarize what was tried in previous iterations
                    for a in all_actions:
                        if not a.success:
                            prev_attempts.append(
                                f"Attempt: {a.step.description}\n"
                                f"Error: {a.error or 'unknown'}"
                            )
                try:
                    current_plan = await self.self_heal(
                        verification.errors, task, project_context,
                        previous_attempts=prev_attempts or None,
                        changed_files=changed_files,
                    )
                    current_plan = self._ensure_reads_before_edits(current_plan)
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
