"""Diagram generation tool - Mermaid, Graphviz, PlantUML from code structure."""

import asyncio
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


class DiagramTool:
    name = "diagram"
    description = (
        "Generate architecture diagrams from code. Produces Mermaid, Graphviz (DOT), "
        "or PlantUML diagrams showing: class hierarchies, module dependencies, "
        "call graphs, database schemas, and sequence diagrams. Can also render to SVG/PNG."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["class_diagram", "module_deps", "call_graph", "sequence", "er_diagram", "custom", "render"],
                "description": "Type of diagram to generate",
            },
            "path": {
                "type": "string",
                "description": "Source directory or file to analyze",
            },
            "format": {
                "type": "string",
                "enum": ["mermaid", "graphviz", "plantuml"],
                "description": "Output diagram format",
                "default": "mermaid",
            },
            "output_file": {
                "type": "string",
                "description": "File path to save rendered diagram (SVG/PNG)",
            },
            "content": {
                "type": "string",
                "description": "Custom diagram content (for custom and render actions)",
            },
            "file_filter": {
                "type": "string",
                "description": "Filter files to include (glob pattern)",
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum depth for dependency trees",
                "default": 3,
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "class_diagram":
                return await self._class_diagram(kwargs)
            elif action == "module_deps":
                return await self._module_deps(kwargs)
            elif action == "call_graph":
                return await self._call_graph(kwargs)
            elif action == "sequence":
                return await self._sequence_diagram(kwargs)
            elif action == "er_diagram":
                return await self._er_diagram(kwargs)
            elif action == "custom":
                return self._custom_diagram(kwargs)
            elif action == "render":
                return await self._render(kwargs)
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _class_diagram(self, kwargs: dict) -> ToolResult:
        """Generate class diagram from Python/TypeScript source."""
        path = kwargs.get("path", ".")
        fmt = kwargs.get("format", "mermaid")
        file_filter = kwargs.get("file_filter", "*.py")

        classes = []
        relationships = []

        ignore = {"node_modules", "venv", ".venv", ".git", "__pycache__"}
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignore and not d.startswith(".")]
            for fname in files:
                if not self._matches_filter(fname, file_filter):
                    continue
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r", errors="ignore") as f:
                        content = f.read()
                except (IOError, OSError):
                    continue
                if fname.endswith(".py"):
                    self._parse_python_classes(content, classes, relationships)
                elif fname.endswith((".ts", ".js")):
                    self._parse_ts_classes(content, classes, relationships)

        if not classes:
            return ToolResult(success=False, error="No classes found")

        if fmt == "mermaid":
            diagram = self._mermaid_class(classes, relationships)
        elif fmt == "graphviz":
            diagram = self._graphviz_class(classes, relationships)
        else:
            diagram = self._plantuml_class(classes, relationships)

        return ToolResult(success=True, output=diagram,
                         artifacts={"classes": len(classes), "relationships": len(relationships)})

    def _matches_filter(self, filename, pattern):
        if pattern == "*":
            return True
        ext = pattern.replace("*", "")
        return filename.endswith(ext)

    def _parse_python_classes(self, content, classes, relationships):
        class_pat = re.compile(r'^class\s+(\w+)(?:\((.*?)\))?:', re.MULTILINE)
        method_pat = re.compile(r'^\s+def\s+(\w+)\s*\(', re.MULTILINE)
        for match in class_pat.finditer(content):
            name = match.group(1)
            bases = match.group(2) or ""
            start = match.end()
            nxt = class_pat.search(content, start)
            end = nxt.start() if nxt else len(content)
            body = content[start:end]
            methods = method_pat.findall(body)
            classes.append({"name": name, "methods": methods[:15], "bases": bases.split(",")})
            for base in bases.split(","):
                base = base.strip().split("(")[0].split("[")[0]
                if base and base not in ("object", "ABC", "BaseModel"):
                    relationships.append({"from": name, "to": base, "type": "inherits"})

    def _parse_ts_classes(self, content, classes, relationships):
        pat = re.compile(r'(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w,\s]+))?')
        for m in pat.finditer(content):
            name, extends, implements = m.group(1), m.group(2), m.group(3)
            classes.append({"name": name, "methods": [], "bases": []})
            if extends:
                relationships.append({"from": name, "to": extends, "type": "inherits"})
            if implements:
                for iface in implements.split(","):
                    relationships.append({"from": name, "to": iface.strip(), "type": "implements"})

    def _mermaid_class(self, classes, rels):
        lines = ["classDiagram"]
        for c in classes[:50]:
            lines.append("    class " + c["name"] + " {")
            for m in c.get("methods", [])[:10]:
                lines.append("        +" + m + "()")
            lines.append("    }")
        for r in rels[:100]:
            arrow = " <|-- " if r["type"] == "inherits" else " <|.. "
            lines.append("    " + r["to"] + arrow + r["from"])
        return "\n".join(lines)

    def _graphviz_class(self, classes, rels):
        lines = ["digraph classes {", "    rankdir=BT;", "    node [shape=record];"]
        for c in classes[:50]:
            methods = "|".join("+" + m + "()" for m in c.get("methods", [])[:8])
            label = c["name"] + "|" + methods if methods else c["name"]
            lines.append("    " + c["name"] + " [label=\"{" + label + "}\"];")
        for r in rels[:100]:
            style = "solid" if r["type"] == "inherits" else "dashed"
            lines.append("    " + r["from"] + " -> " + r["to"] + " [style=" + style + "];")
        lines.append("}")
        return "\n".join(lines)

    def _plantuml_class(self, classes, rels):
        lines = ["@startuml"]
        for c in classes[:50]:
            lines.append("class " + c["name"] + " {")
            for m in c.get("methods", [])[:10]:
                lines.append("    +" + m + "()")
            lines.append("}")
        for r in rels[:100]:
            arrow = " <|-- " if r["type"] == "inherits" else " <|.. "
            lines.append(r["to"] + arrow + r["from"])
        lines.append("@enduml")
        return "\n".join(lines)

    async def _module_deps(self, kwargs: dict) -> ToolResult:
        """Generate module dependency diagram."""
        path = kwargs.get("path", ".")
        fmt = kwargs.get("format", "mermaid")
        imports = {}
        ignore = {"node_modules", "venv", ".venv", ".git", "__pycache__"}
        stdlib = {"os", "sys", "re", "json", "typing", "pathlib", "asyncio", "dataclasses",
                  "collections", "functools", "itertools", "abc", "io", "time", "datetime",
                  "logging", "hashlib", "subprocess", "tempfile", "shutil", "platform", "math"}

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignore]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                filepath = os.path.join(root, fname)
                module = os.path.relpath(filepath, path).replace("/", ".").replace(".py", "")
                if module.endswith(".__init__"):
                    module = module[:-9]
                try:
                    with open(filepath, "r", errors="ignore") as f:
                        content = f.read()
                except IOError:
                    continue
                deps = set()
                for m in re.finditer(r'^(?:from|import)\s+([\w.]+)', content, re.MULTILINE):
                    dep = m.group(1).split(".")[0]
                    if dep not in stdlib:
                        deps.add(dep)
                short_mod = module.split(".")[0]
                if deps:
                    imports.setdefault(short_mod, set()).update(deps)

        if not imports:
            return ToolResult(success=False, error="No module imports found")

        if fmt == "mermaid":
            lines = ["graph LR"]
            seen = set()
            for mod, deps in list(imports.items())[:30]:
                for dep in list(deps)[:10]:
                    edge = mod + "-->" + dep
                    if edge not in seen:
                        lines.append("    " + mod + " --> " + dep)
                        seen.add(edge)
        else:
            lines = ["digraph modules {", "    rankdir=LR;"]
            for mod, deps in list(imports.items())[:30]:
                for dep in list(deps)[:10]:
                    lines.append("    \"" + mod + "\" -> \"" + dep + "\";")
            lines.append("}")

        return ToolResult(success=True, output="\n".join(lines), artifacts={"modules": len(imports)})

    async def _call_graph(self, kwargs: dict) -> ToolResult:
        """Generate function call graph for a single file."""
        path = kwargs.get("path", "")
        fmt = kwargs.get("format", "mermaid")
        if not path or not os.path.isfile(path):
            return ToolResult(success=False, error="path must be a valid file")

        with open(path, "r", errors="ignore") as f:
            content = f.read()

        functions = re.findall(r'^def\s+(\w+)\s*\(', content, re.MULTILINE)
        calls = {}
        for func in functions:
            pat = r'^def\s+' + func + r'\s*\(.*?\).*?:'
            match = re.search(pat, content, re.MULTILINE)
            if not match:
                continue
            start = match.end()
            nxt = re.search(r'^def\s+\w+\s*\(', content[start:], re.MULTILINE)
            end = start + nxt.start() if nxt else len(content)
            body = content[start:end]
            called = set()
            for other in functions:
                if other != func and re.search(r'\b' + other + r'\s*\(', body):
                    called.add(other)
            if called:
                calls[func] = called

        if not calls:
            return ToolResult(success=True, output="No internal function calls detected")

        if fmt == "mermaid":
            lines = ["graph TD"]
            for caller, callees in calls.items():
                for callee in callees:
                    lines.append("    " + caller + " --> " + callee)
        else:
            lines = ["digraph calls {"]
            for caller, callees in calls.items():
                for callee in callees:
                    lines.append("    \"" + caller + "\" -> \"" + callee + "\";")
            lines.append("}")

        return ToolResult(success=True, output="\n".join(lines),
                         artifacts={"functions": len(functions), "edges": sum(len(v) for v in calls.values())})

    async def _sequence_diagram(self, kwargs: dict) -> ToolResult:
        """Generate sequence diagram from description."""
        content = kwargs.get("content", "")
        if not content:
            return ToolResult(success=False, error="content is required for sequence diagrams")

        lines = ["sequenceDiagram"]
        for line in content.strip().split("\n"):
            line = line.strip()
            if "->" in line:
                parts = re.split(r'\s*->\s*', line, 1)
                if len(parts) == 2:
                    sender, rest = parts
                    if ":" in rest:
                        receiver, msg = rest.split(":", 1)
                        lines.append("    " + sender.strip() + "->>+" + receiver.strip() + ": " + msg.strip())
                    else:
                        lines.append("    " + sender.strip() + "->>+" + rest.strip() + ": call")
            else:
                lines.append("    Note over Client: " + line)

        return ToolResult(success=True, output="\n".join(lines))

    async def _er_diagram(self, kwargs: dict) -> ToolResult:
        """Generate ER diagram from models or SQL."""
        path = kwargs.get("path", ".")
        fmt = kwargs.get("format", "mermaid")
        entities = {}
        ignore = {"node_modules", "venv", ".venv", ".git", "__pycache__"}

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ignore]
            for fname in files:
                if not fname.endswith((".py", ".sql")):
                    continue
                filepath = os.path.join(root, fname)
                try:
                    with open(filepath, "r", errors="ignore") as f:
                        content = f.read()
                except IOError:
                    continue
                # SQLAlchemy models
                for m in re.finditer(r'class\s+(\w+)\(.*?(?:Base|Model|db\.Model).*?\):', content):
                    table = m.group(1)
                    start = m.end()
                    nxt = re.search(r'^class\s+', content[start:], re.MULTILINE)
                    end = start + nxt.start() if nxt else len(content)
                    body = content[start:end]
                    cols = re.findall(r'(\w+)\s*=\s*(?:db\.)?Column\((\w+)', body)
                    entities[table] = cols
                # SQL CREATE TABLE
                for m in re.finditer(r'CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\w+', content, re.IGNORECASE):
                    table = m.group(0).split()[-1].strip("\"'`")
                    start = m.end()
                    end_m = re.search(r'\);', content[start:])
                    end = start + end_m.end() if end_m else start + 500
                    body = content[start:end]
                    cols = re.findall(r'(\w+)\s+(INTEGER|TEXT|VARCHAR|BOOLEAN|TIMESTAMP|FLOAT|REAL|BLOB)', body, re.IGNORECASE)
                    if cols:
                        entities[table] = cols

        if not entities:
            return ToolResult(success=False, error="No database models/tables found")

        if fmt == "mermaid":
            lines = ["erDiagram"]
            for table, cols in list(entities.items())[:30]:
                lines.append("    " + table + " {")
                for col_name, col_type in cols[:15]:
                    lines.append("        " + col_type.lower() + " " + col_name)
                lines.append("    }")
        else:
            lines = ["@startuml"]
            for table, cols in list(entities.items())[:30]:
                lines.append("entity " + table + " {")
                for col_name, col_type in cols[:15]:
                    lines.append("    " + col_name + " : " + col_type)
                lines.append("}")
            lines.append("@enduml")

        return ToolResult(success=True, output="\n".join(lines), artifacts={"tables": len(entities)})

    def _custom_diagram(self, kwargs: dict) -> ToolResult:
        content = kwargs.get("content", "")
        if not content:
            return ToolResult(success=False, error="content is required")
        return ToolResult(success=True, output=content)

    async def _render(self, kwargs: dict) -> ToolResult:
        """Render diagram to SVG/PNG."""
        content = kwargs.get("content", "")
        output_file = kwargs.get("output_file", "diagram.svg")
        fmt = kwargs.get("format", "mermaid")
        if not content:
            return ToolResult(success=False, error="content is required for rendering")

        import tempfile
        suffix = ".mmd" if fmt == "mermaid" else ".dot"
        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
            f.write(content)
            input_file = f.name
        try:
            if fmt == "mermaid":
                cmd = ["mmdc", "-i", input_file, "-o", output_file]
            elif fmt == "graphviz":
                ext = os.path.splitext(output_file)[1][1:] or "svg"
                cmd = ["dot", "-T" + ext, "-o", output_file, input_file]
            else:
                cmd = ["plantuml", "-t" + os.path.splitext(output_file)[1][1:], input_file]
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return ToolResult(success=False, error="Rendering failed: " + stderr.decode())
            return ToolResult(success=True, output="Diagram rendered to: " + output_file,
                             artifacts={"output_file": output_file})
        finally:
            os.unlink(input_file)
