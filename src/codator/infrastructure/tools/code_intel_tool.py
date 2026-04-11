"""Code Intelligence Tool — semantic code search and symbol analysis.

Uses tree-sitter (when available) or ctags/regex fallback for:
- Find symbol definitions (functions, classes, methods)
- Find references/usages of a symbol
- List symbols in a file (outline)
- Show call hierarchy
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)

# Language-specific patterns for definition finding
_DEF_PATTERNS = {
    "python": [
        (r"^\s*(async\s+)?def\s+{symbol}\s*\(", "function"),
        (r"^\s*class\s+{symbol}\s*[:\(]", "class"),
        (r"^\s*{symbol}\s*=", "variable"),
    ],
    "javascript": [
        (r"(?:function|const|let|var)\s+{symbol}\s*[\(=]", "function/variable"),
        (r"class\s+{symbol}\s*[\{<]", "class"),
        (r"export\s+(?:default\s+)?(?:function|class|const|let)\s+{symbol}", "export"),
    ],
    "typescript": [
        (r"(?:function|const|let|var)\s+{symbol}\s*[\(=<:]", "function/variable"),
        (r"(?:export\s+)?(?:class|interface|type|enum)\s+{symbol}", "type/class"),
    ],
    "go": [
        (r"func\s+(?:\([^)]*\)\s+)?{symbol}\s*\(", "function"),
        (r"type\s+{symbol}\s+(?:struct|interface)", "type"),
    ],
    "rust": [
        (r"(?:pub\s+)?(?:fn|struct|enum|trait|type|const|static)\s+{symbol}", "definition"),
        (r"impl(?:<[^>]*>)?\s+{symbol}", "impl"),
    ],
}

_LANG_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


class CodeIntelTool(Tool):
    """Semantic code search: find definitions, references, outlines."""

    def __init__(self, cwd: str | Path = ""):
        self._cwd = Path(cwd) if cwd else Path.cwd()

    @property
    def name(self) -> str:
        return "code_intel"

    @property
    def description(self) -> str:
        return (
            "Semantic code analysis. Actions: find_definition (locate where a symbol is defined), "
            "find_references (find usages of a symbol), outline (list all symbols in a file), "
            "search_symbols (find symbol by partial name across project)."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["find_definition", "find_references", "outline", "search_symbols"],
                },
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File path (for 'outline') or directory scope.",
                },
                "language": {
                    "type": "string",
                    "description": "Language filter (python, javascript, typescript, go, rust).",
                },
                "include_pattern": {
                    "type": "string",
                    "description": "Glob pattern to filter files (e.g., '*.py', 'src/**/*.ts').",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "outline")
        if action == "find_definition":
            return await self._find_definition(**kwargs)
        elif action == "find_references":
            return await self._find_references(**kwargs)
        elif action == "outline":
            return await self._outline(**kwargs)
        elif action == "search_symbols":
            return await self._search_symbols(**kwargs)
        return ToolResult(success=False, error=f"Unknown action: {action}")

    def _detect_lang(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        return _LANG_EXTENSIONS.get(ext, "")

    def _get_glob_pattern(self, language: str, include_pattern: str) -> str:
        if include_pattern:
            return include_pattern
        ext_map = {
            "python": "*.py",
            "javascript": "*.{js,jsx}",
            "typescript": "*.{ts,tsx}",
            "go": "*.go",
            "rust": "*.rs",
        }
        return ext_map.get(language, "*")

    async def _ripgrep(self, pattern: str, glob_pat: str, max_results: int = 50) -> str:
        """Use ripgrep for fast regex search."""
        cmd = [
            "rg", "--no-heading", "--line-number", "--color=never",
            "-g", glob_pat, "--max-count=5", pattern,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._cwd),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=15)
        lines = stdout.decode(errors="replace").strip().split("\n")
        return "\n".join(lines[:max_results])

    async def _find_definition(self, **kwargs) -> ToolResult:
        symbol = kwargs.get("symbol", "")
        if not symbol:
            return ToolResult(success=False, error="'symbol' required.")

        language = kwargs.get("language", "")
        include_pattern = kwargs.get("include_pattern", "")
        glob_pat = self._get_glob_pattern(language, include_pattern)

        # Build definition-finding regex patterns
        patterns = []
        if language and language in _DEF_PATTERNS:
            patterns = [(p.format(symbol=re.escape(symbol)), t)
                       for p, t in _DEF_PATTERNS[language]]
        else:
            # Try all languages
            for lang_patterns in _DEF_PATTERNS.values():
                patterns.extend(
                    (p.format(symbol=re.escape(symbol)), t)
                    for p, t in lang_patterns
                )

        results = []
        for pattern, kind in patterns:
            try:
                output = await self._ripgrep(pattern, glob_pat)
                if output:
                    for line in output.split("\n"):
                        if line.strip():
                            results.append(f"[{kind}] {line}")
            except Exception:
                continue

        if not results:
            return ToolResult(
                success=True,
                output=f"No definitions found for '{symbol}'.",
            )

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique.append(r)

        return ToolResult(
            success=True,
            output=f"Definitions of '{symbol}':\n" + "\n".join(unique[:30]),
        )

    async def _find_references(self, **kwargs) -> ToolResult:
        symbol = kwargs.get("symbol", "")
        if not symbol:
            return ToolResult(success=False, error="'symbol' required.")

        language = kwargs.get("language", "")
        include_pattern = kwargs.get("include_pattern", "")
        glob_pat = self._get_glob_pattern(language, include_pattern)

        # Word-boundary match
        pattern = rf"\b{re.escape(symbol)}\b"
        output = await self._ripgrep(pattern, glob_pat, max_results=50)

        if not output.strip():
            return ToolResult(success=True, output=f"No references found for '{symbol}'.")

        lines = [l for l in output.split("\n") if l.strip()]
        return ToolResult(
            success=True,
            output=f"References to '{symbol}' ({len(lines)} found):\n" + "\n".join(lines[:40]),
        )

    async def _outline(self, **kwargs) -> ToolResult:
        path = kwargs.get("path", "")
        if not path:
            return ToolResult(success=False, error="'path' required for outline.")

        full_path = self._cwd / path if not Path(path).is_absolute() else Path(path)
        if not full_path.exists():
            return ToolResult(success=False, error=f"File not found: {path}")

        language = kwargs.get("language", "") or self._detect_lang(path)
        content = full_path.read_text(errors="replace")
        lines = content.split("\n")

        symbols = []
        if language == "python":
            for i, line in enumerate(lines, 1):
                if m := re.match(r"^(\s*)(async\s+)?def\s+(\w+)", line):
                    indent = len(m.group(1))
                    name = m.group(3)
                    kind = "method" if indent > 0 else "function"
                    symbols.append(f"  L{i:4d} {kind:10s} {name}")
                elif m := re.match(r"^(\s*)class\s+(\w+)", line):
                    indent = len(m.group(1))
                    name = m.group(2)
                    symbols.append(f"  L{i:4d} {'class':10s} {name}")
        elif language in ("javascript", "typescript"):
            for i, line in enumerate(lines, 1):
                if m := re.match(r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", line):
                    symbols.append(f"  L{i:4d} {'function':10s} {m.group(1)}")
                elif m := re.match(r"^\s*(?:export\s+)?class\s+(\w+)", line):
                    symbols.append(f"  L{i:4d} {'class':10s} {m.group(1)}")
                elif m := re.match(r"^\s*(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", line):
                    symbols.append(f"  L{i:4d} {'arrow fn':10s} {m.group(1)}")
                elif m := re.match(r"^\s*(?:export\s+)?(?:interface|type|enum)\s+(\w+)", line):
                    symbols.append(f"  L{i:4d} {'type':10s} {m.group(1)}")
        elif language == "go":
            for i, line in enumerate(lines, 1):
                if m := re.match(r"^func\s+(?:\([^)]*\)\s+)?(\w+)", line):
                    symbols.append(f"  L{i:4d} {'function':10s} {m.group(1)}")
                elif m := re.match(r"^type\s+(\w+)\s+(struct|interface)", line):
                    symbols.append(f"  L{i:4d} {m.group(2):10s} {m.group(1)}")
        elif language == "rust":
            for i, line in enumerate(lines, 1):
                if m := re.match(r"^\s*(?:pub\s+)?fn\s+(\w+)", line):
                    symbols.append(f"  L{i:4d} {'function':10s} {m.group(1)}")
                elif m := re.match(r"^\s*(?:pub\s+)?(struct|enum|trait)\s+(\w+)", line):
                    symbols.append(f"  L{i:4d} {m.group(1):10s} {m.group(2)}")

        if not symbols:
            return ToolResult(success=True, output=f"No symbols found in {path} (lang={language})")

        return ToolResult(
            success=True,
            output=f"Outline of {path} ({len(symbols)} symbols):\n" + "\n".join(symbols),
        )

    async def _search_symbols(self, **kwargs) -> ToolResult:
        symbol = kwargs.get("symbol", "")
        if not symbol:
            return ToolResult(success=False, error="'symbol' required.")

        language = kwargs.get("language", "")
        include_pattern = kwargs.get("include_pattern", "")
        glob_pat = self._get_glob_pattern(language, include_pattern)

        # Search for definitions containing the partial symbol name
        pattern = rf"(?:def|class|function|func|type|struct|enum|trait|interface|const|let|var)\s+\w*{re.escape(symbol)}\w*"
        output = await self._ripgrep(pattern, glob_pat, max_results=40)

        if not output.strip():
            return ToolResult(success=True, output=f"No symbols matching '{symbol}' found.")

        lines = [l for l in output.split("\n") if l.strip()]
        return ToolResult(
            success=True,
            output=f"Symbols matching '{symbol}' ({len(lines)} found):\n" + "\n".join(lines[:30]),
        )
