"""Refactoring tool — rename symbol, extract function, inline variable."""

import asyncio
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


class RefactorTool:
    name = "refactor"
    description = (
        "Refactor code: rename symbols across project, extract function/class, "
        "inline variable, move symbol to another file. Supports Python (via rope), "
        "and regex-based refactoring for any language."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["rename", "extract_function", "extract_variable", "inline", "move"],
                "description": "Refactoring action to perform",
            },
            "path": {
                "type": "string",
                "description": "File or directory path (project root for rename across project)",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name to refactor (current name for rename, code selection for extract)",
            },
            "new_name": {
                "type": "string",
                "description": "New name for the symbol (required for rename, extract_function, extract_variable)",
            },
            "target_file": {
                "type": "string",
                "description": "Target file path (for move action)",
            },
            "line_start": {
                "type": "integer",
                "description": "Start line for extract operations",
            },
            "line_end": {
                "type": "integer",
                "description": "End line for extract operations",
            },
            "language": {
                "type": "string",
                "enum": ["python", "javascript", "typescript", "go", "rust", "auto"],
                "description": "Language (auto-detected if not specified)",
                "default": "auto",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Preview changes without applying them",
                "default": False,
            },
        },
        "required": ["action", "path", "symbol"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs["path"]
        symbol = kwargs["symbol"]
        new_name = kwargs.get("new_name", "")
        dry_run = kwargs.get("dry_run", False)
        language = kwargs.get("language", "auto")

        if language == "auto":
            language = self._detect_language(path)

        try:
            if action == "rename":
                return await self._rename(path, symbol, new_name, language, dry_run)
            elif action == "extract_function":
                line_start = kwargs.get("line_start", 0)
                line_end = kwargs.get("line_end", 0)
                return await self._extract_function(path, symbol, new_name, line_start, line_end, language, dry_run)
            elif action == "extract_variable":
                return await self._extract_variable(path, symbol, new_name, language, dry_run)
            elif action == "inline":
                return await self._inline(path, symbol, language, dry_run)
            elif action == "move":
                target = kwargs.get("target_file", "")
                return await self._move(path, symbol, target, language, dry_run)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _detect_language(self, path: str) -> str:
        ext_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".jsx": "javascript", ".tsx": "typescript",
            ".go": "go", ".rs": "rust",
        }
        if os.path.isfile(path):
            _, ext = os.path.splitext(path)
            return ext_map.get(ext, "python")
        # Check common files in directory
        for f in os.listdir(path)[:20]:
            _, ext = os.path.splitext(f)
            if ext in ext_map:
                return ext_map[ext]
        return "python"

    async def _rename(self, path: str, old_name: str, new_name: str, language: str, dry_run: bool) -> ToolResult:
        if not new_name:
            return ToolResult(success=False, error="new_name is required for rename")

        if language == "python":
            return await self._rename_rope(path, old_name, new_name, dry_run)
        else:
            return await self._rename_regex(path, old_name, new_name, language, dry_run)

    async def _rename_rope(self, path: str, old_name: str, new_name: str, dry_run: bool) -> ToolResult:
        """Use rope library for Python refactoring."""
        try:
            import rope.base.project
            import rope.refactor.rename
        except ImportError:
            return await self._rename_regex(path, old_name, new_name, "python", dry_run)

        project_root = path if os.path.isdir(path) else os.path.dirname(path)
        project = rope.base.project.Project(project_root)

        try:
            # Find the resource containing the symbol
            matches = []
            for resource in project.get_files():
                content = resource.read()
                for m in re.finditer(r'\b' + re.escape(old_name) + r'\b', content):
                    matches.append((resource, m.start()))
                    break  # First occurrence is enough for rope

            if not matches:
                return ToolResult(success=False, error=f"Symbol '{old_name}' not found in project")

            resource, offset = matches[0]
            renamer = rope.refactor.rename.Rename(project, resource, offset)
            changes = renamer.get_changes(new_name)

            if dry_run:
                return ToolResult(
                    success=True,
                    output=f"Dry run — rename '{old_name}' → '{new_name}':\n{changes.get_description()}",
                    artifacts={"changes_preview": changes.get_description()},
                )

            project.do(changes)
            return ToolResult(
                success=True,
                output=f"Renamed '{old_name}' → '{new_name}' across project\n{changes.get_description()}",
                artifacts={"files_changed": len(changes.changes)},
            )
        finally:
            project.close()

    async def _rename_regex(self, path: str, old_name: str, new_name: str, language: str, dry_run: bool) -> ToolResult:
        """Regex-based rename across files."""
        ext_map = {
            "python": "*.py", "javascript": "*.{js,jsx}",
            "typescript": "*.{ts,tsx}", "go": "*.go", "rust": "*.rs",
        }
        glob_pattern = ext_map.get(language, "*")
        search_dir = path if os.path.isdir(path) else os.path.dirname(path)

        # Use ripgrep to find occurrences
        cmd = ["rg", "-l", "--glob", glob_pattern, r"\b" + old_name + r"\b", search_dir]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await proc.communicate()
        files = [f for f in stdout.decode().strip().split("\n") if f]

        if not files:
            return ToolResult(success=False, error=f"Symbol '{old_name}' not found")

        if dry_run:
            # Count occurrences
            cmd2 = ["rg", "--count", "--glob", glob_pattern, r"\b" + old_name + r"\b", search_dir]
            proc2 = await asyncio.create_subprocess_exec(
                *cmd2, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout2, _ = await proc2.communicate()
            return ToolResult(
                success=True,
                output=f"Dry run — rename '{old_name}' → '{new_name}':\n{stdout2.decode()}",
                artifacts={"files_affected": len(files)},
            )

        # Apply rename using sed
        pattern = r"\b" + re.escape(old_name) + r"\b"
        changed = 0
        for filepath in files:
            with open(filepath, "r") as f:
                content = f.read()
            new_content = re.sub(pattern, new_name, content)
            if new_content != content:
                with open(filepath, "w") as f:
                    f.write(new_content)
                changed += 1

        return ToolResult(
            success=True,
            output=f"Renamed '{old_name}' → '{new_name}' in {changed} files",
            artifacts={"files_changed": changed, "files": files},
        )

    async def _extract_function(self, path: str, code_or_symbol: str, new_name: str,
                                 line_start: int, line_end: int, language: str, dry_run: bool) -> ToolResult:
        """Extract lines into a new function."""
        if not new_name:
            return ToolResult(success=False, error="new_name required for extract_function")
        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"File not found: {path}")
        if line_start <= 0 or line_end <= 0:
            return ToolResult(success=False, error="line_start and line_end required for extract_function")

        with open(path, "r") as f:
            lines = f.readlines()

        if line_end > len(lines):
            line_end = len(lines)

        extracted = lines[line_start - 1:line_end]
        indent = re.match(r"(\s*)", extracted[0]).group(1) if extracted else ""

        # Detect variables used (simple heuristic)
        all_vars = set(re.findall(r'\b([a-z_]\w*)\b', "".join(extracted)))
        before_text = "".join(lines[:line_start - 1])
        defined_before = set(re.findall(r'\b([a-z_]\w*)\s*=', before_text))
        params = sorted(all_vars & defined_before)

        if language == "python":
            func_def = f"{indent}def {new_name}({', '.join(params)}):\n"
            func_body = "".join(f"    {line}" if not line.startswith(indent) else f"    {line[len(indent):]}" for line in extracted)
            replacement = f"{indent}{new_name}({', '.join(params)})\n"
            new_func = f"\n{func_def}{func_body}\n"
        else:
            param_str = ", ".join(params)
            func_def = f"{indent}function {new_name}({param_str}) {{\n"
            func_body = "".join(f"  {line}" for line in extracted)
            replacement = f"{indent}{new_name}({param_str});\n"
            new_func = f"\n{func_def}{func_body}{indent}}}\n"

        new_lines = lines[:line_start - 1] + [replacement] + lines[line_end:]
        # Insert function before the call
        insert_pos = line_start - 1
        new_lines.insert(insert_pos, new_func)

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Would extract lines {line_start}-{line_end} into {new_name}({', '.join(params)})",
                artifacts={"function_signature": f"{new_name}({', '.join(params)})", "lines": len(extracted)},
            )

        with open(path, "w") as f:
            f.writelines(new_lines)

        return ToolResult(
            success=True,
            output=f"Extracted lines {line_start}-{line_end} into function '{new_name}({', '.join(params)})'",
            artifacts={"params_detected": params},
        )

    async def _extract_variable(self, path: str, expression: str, new_name: str, language: str, dry_run: bool) -> ToolResult:
        """Extract repeated expression into a variable."""
        if not new_name:
            return ToolResult(success=False, error="new_name required")
        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"File not found: {path}")

        with open(path, "r") as f:
            content = f.read()

        occurrences = content.count(expression)
        if occurrences == 0:
            return ToolResult(success=False, error=f"Expression '{expression}' not found in file")

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Would extract '{expression}' into variable '{new_name}' ({occurrences} occurrences)",
            )

        # Find first occurrence and insert variable assignment before that line
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if expression in line:
                indent = re.match(r"(\s*)", line).group(1)
                assignment = f"{indent}{new_name} = {expression}"
                lines.insert(i, assignment)
                break

        new_content = "\n".join(lines)
        new_content = new_content.replace(expression, new_name)

        with open(path, "w") as f:
            f.write(new_content)

        return ToolResult(
            success=True,
            output=f"Extracted '{expression}' → '{new_name}' ({occurrences} replacements)",
        )

    async def _inline(self, path: str, variable: str, language: str, dry_run: bool) -> ToolResult:
        """Inline a variable — replace all usages with its value."""
        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"File not found: {path}")

        with open(path, "r") as f:
            content = f.read()

        # Find assignment
        pattern = rf"^\s*(?:(?:const|let|var)\s+)?{re.escape(variable)}\s*=\s*(.+)$"
        match = re.search(pattern, content, re.MULTILINE)
        if not match:
            return ToolResult(success=False, error=f"Cannot find assignment for '{variable}'")

        value = match.group(1).rstrip().rstrip(";")
        assignment_line = match.group(0)
        usages = len(re.findall(r'\b' + re.escape(variable) + r'\b', content)) - 1

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Would inline '{variable}' = '{value}' ({usages} usages)",
            )

        # Remove assignment line and replace all usages
        new_content = content.replace(assignment_line + "\n", "")
        new_content = re.sub(r'\b' + re.escape(variable) + r'\b', value, new_content)

        with open(path, "w") as f:
            f.write(new_content)

        return ToolResult(success=True, output=f"Inlined '{variable}' → '{value}' ({usages} replacements)")

    async def _move(self, path: str, symbol: str, target_file: str, language: str, dry_run: bool) -> ToolResult:
        """Move a function/class to another file."""
        if not target_file:
            return ToolResult(success=False, error="target_file required for move")
        if not os.path.isfile(path):
            return ToolResult(success=False, error=f"Source file not found: {path}")

        with open(path, "r") as f:
            lines = f.readlines()

        # Find the symbol definition
        if language == "python":
            pattern = rf"^(class|def)\s+{re.escape(symbol)}"
        elif language in ("javascript", "typescript"):
            pattern = rf"^(export\s+)?(function|class|const|let|var)\s+{re.escape(symbol)}"
        elif language == "go":
            pattern = rf"^func\s+{re.escape(symbol)}"
        else:
            pattern = rf"(func|fn|class|def|function)\s+{re.escape(symbol)}"

        start_idx = None
        for i, line in enumerate(lines):
            if re.match(pattern, line):
                start_idx = i
                break

        if start_idx is None:
            return ToolResult(success=False, error=f"Symbol '{symbol}' not found in {path}")

        # Find end of symbol (next top-level definition or end of file)
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            if re.match(r'^(class|def|func|fn|function|export)\s', lines[i]) and not lines[i].startswith(" "):
                end_idx = i
                break

        extracted_code = "".join(lines[start_idx:end_idx])

        if dry_run:
            return ToolResult(
                success=True,
                output=f"Would move '{symbol}' ({end_idx - start_idx} lines) from {path} → {target_file}",
            )

        # Remove from source
        remaining = lines[:start_idx] + lines[end_idx:]
        with open(path, "w") as f:
            f.writelines(remaining)

        # Append to target
        mode = "a" if os.path.exists(target_file) else "w"
        with open(target_file, mode) as f:
            f.write("\n" + extracted_code)

        return ToolResult(
            success=True,
            output=f"Moved '{symbol}' ({end_idx - start_idx} lines) from {path} → {target_file}",
        )
