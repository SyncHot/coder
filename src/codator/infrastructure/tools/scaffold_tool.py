"""Scaffold/template tool - generate boilerplate from templates."""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


TEMPLATES = {
    "python_package": {
        "files": {
            "src/{name}/__init__.py": '"""{name} - {description}"""\n\n__version__ = "0.1.0"\n',
            "src/{name}/main.py": '"""Main module."""\n\n\ndef main():\n    print("Hello from {name}")\n\n\nif __name__ == "__main__":\n    main()\n',
            "tests/__init__.py": "",
            "tests/test_{name}.py": 'import pytest\nfrom {name}.main import main\n\n\ndef test_main(capsys):\n    main()\n    captured = capsys.readouterr()\n    assert "Hello" in captured.out\n',
            "pyproject.toml": '[build-system]\nrequires = ["setuptools>=68.0"]\nbuild-backend = "setuptools.backends._legacy:_Backend"\n\n[project]\nname = "{name}"\nversion = "0.1.0"\ndescription = "{description}"\nrequires-python = ">=3.10"\n\n[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
            "README.md": "# {name}\n\n{description}\n\n## Installation\n\n```bash\npip install -e .\n```\n\n## Usage\n\n```python\nfrom {name}.main import main\nmain()\n```\n",
            ".gitignore": "__pycache__/\n*.py[cod]\n*$py.class\n*.egg-info/\ndist/\nbuild/\n.venv/\n.env\n",
        },
    },
    "flask_app": {
        "files": {
            "app/__init__.py": 'from flask import Flask\n\n\ndef create_app():\n    app = Flask(__name__)\n\n    from .routes import main_bp\n    app.register_blueprint(main_bp)\n\n    return app\n',
            "app/routes.py": 'from flask import Blueprint, jsonify\n\nmain_bp = Blueprint("main", __name__)\n\n\n@main_bp.route("/health")\ndef health():\n    return jsonify({"status": "ok"})\n',
            "requirements.txt": "flask>=3.0\ngunicorn>=21.0\npytest>=8.0\n",
            "run.py": 'from app import create_app\n\napp = create_app()\n\nif __name__ == "__main__":\n    app.run(debug=True)\n',
            "Dockerfile": 'FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD ["gunicorn", "-b", "0.0.0.0:8000", "run:app"]\n',
        },
    },
    "fastapi_app": {
        "files": {
            "app/__init__.py": "",
            "app/main.py": 'from fastapi import FastAPI\nfrom app.routers import health\n\napp = FastAPI(title="{name}")\napp.include_router(health.router)\n',
            "app/routers/__init__.py": "",
            "app/routers/health.py": 'from fastapi import APIRouter\n\nrouter = APIRouter()\n\n\n@router.get("/health")\nasync def health():\n    return {"status": "ok"}\n',
            "requirements.txt": "fastapi>=0.110\nuvicorn[standard]>=0.27\npytest>=8.0\nhttpx>=0.27\n",
            "Dockerfile": 'FROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install --no-cache-dir -r requirements.txt\nCOPY . .\nCMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]\n',
        },
    },
    "react_app": {
        "files": {
            "src/App.tsx": 'import React from "react";\n\nfunction App() {\n  return <div className="app"><h1>{name}</h1></div>;\n}\n\nexport default App;\n',
            "src/index.tsx": 'import React from "react";\nimport ReactDOM from "react-dom/client";\nimport App from "./App";\n\nReactDOM.createRoot(document.getElementById("root")!).render(\n  <React.StrictMode><App /></React.StrictMode>\n);\n',
        },
    },
    "go_service": {
        "files": {
            "main.go": 'package main\n\nimport (\n\t"fmt"\n\t"log"\n\t"net/http"\n)\n\nfunc main() {\n\thttp.HandleFunc("/health", healthHandler)\n\tfmt.Println("Starting server on :8080")\n\tlog.Fatal(http.ListenAndServe(":8080", nil))\n}\n\nfunc healthHandler(w http.ResponseWriter, r *http.Request) {\n\tw.Header().Set("Content-Type", "application/json")\n\tw.Write([]byte(`{"status":"ok"}`))\n}\n',
            "go.mod": "module {name}\n\ngo 1.22\n",
            "Dockerfile": 'FROM golang:1.22-alpine AS builder\nWORKDIR /app\nCOPY . .\nRUN go build -o server .\n\nFROM alpine:latest\nCOPY --from=builder /app/server /server\nCMD ["/server"]\n',
        },
    },
    "cli_tool": {
        "files": {
            "src/{name}/__init__.py": '"""{name} CLI tool."""\n\n__version__ = "0.1.0"\n',
            "src/{name}/cli.py": 'import argparse\nimport sys\n\n\ndef main():\n    parser = argparse.ArgumentParser(description="{description}")\n    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")\n    parser.add_argument("input", help="Input to process")\n    args = parser.parse_args()\n    print("Processing: " + args.input)\n\n\nif __name__ == "__main__":\n    main()\n',
            "pyproject.toml": '[build-system]\nrequires = ["setuptools>=68.0"]\nbuild-backend = "setuptools.backends._legacy:_Backend"\n\n[project]\nname = "{name}"\nversion = "0.1.0"\ndescription = "{description}"\nrequires-python = ">=3.10"\n\n[project.scripts]\n{name} = "{name}.cli:main"\n',
        },
    },
}


class ScaffoldTool:
    name = "scaffold"
    description = (
        "Generate project boilerplate from templates. Supports: Python package, "
        "Flask app, FastAPI app, React app, Go service, CLI tool, and custom templates. "
        "Creates directory structure, config files, and starter code."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "custom"],
                "description": "Action: create from template, list available templates, or custom template",
            },
            "template": {
                "type": "string",
                "enum": ["python_package", "flask_app", "fastapi_app", "react_app", "go_service", "cli_tool"],
                "description": "Template to use",
            },
            "path": {
                "type": "string",
                "description": "Directory to create the project in",
            },
            "name": {
                "type": "string",
                "description": "Project name",
            },
            "description": {
                "type": "string",
                "description": "Project description",
                "default": "A new project",
            },
            "variables": {
                "type": "object",
                "description": "Custom template variables (key-value pairs)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        try:
            if action == "create":
                return await self._create(kwargs)
            elif action == "list":
                return self._list_templates()
            elif action == "custom":
                return await self._custom(kwargs)
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    async def _create(self, kwargs: dict) -> ToolResult:
        """Create project from template."""
        template_name = kwargs.get("template", "")
        path = kwargs.get("path", "")
        name = kwargs.get("name", "myproject")
        description = kwargs.get("description", "A new project")

        if not template_name:
            return ToolResult(success=False, error="template is required")
        if not path:
            return ToolResult(success=False, error="path is required")

        if template_name not in TEMPLATES:
            return ToolResult(success=False, error="Unknown template: " + template_name)

        template = TEMPLATES[template_name]
        created_files = []

        for file_path_template, content_template in template["files"].items():
            file_path = file_path_template.replace("{name}", name)
            content = content_template.replace("{name}", name).replace("{description}", description)

            full_path = os.path.join(path, file_path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            with open(full_path, "w") as f:
                f.write(content)
            created_files.append(file_path)

        output = "Created project '" + name + "' from template '" + template_name + "':\n"
        output += "  Directory: " + path + "\n\n"
        output += "Files created:\n"
        for f in sorted(created_files):
            output += "  " + f + "\n"

        return ToolResult(
            success=True,
            output=output,
            artifacts={"files_created": len(created_files), "template": template_name},
        )

    def _list_templates(self) -> ToolResult:
        """List available templates."""
        lines = ["Available templates:\n"]
        descriptions = {
            "python_package": "Python package with src layout, tests, pyproject.toml",
            "flask_app": "Flask web application with blueprints, Dockerfile",
            "fastapi_app": "FastAPI async web service with routers, Dockerfile",
            "react_app": "React + TypeScript + Vite frontend application",
            "go_service": "Go HTTP service with health endpoint, Dockerfile",
            "cli_tool": "Python CLI tool with argparse, installable via pip",
        }
        for name, desc in descriptions.items():
            files = TEMPLATES[name]["files"]
            lines.append("  " + name + " (" + str(len(files)) + " files)")
            lines.append("    " + desc)
            lines.append("")

        return ToolResult(success=True, output="\n".join(lines))

    async def _custom(self, kwargs: dict) -> ToolResult:
        """Create from custom template using cookiecutter."""
        path = kwargs.get("path", "")
        variables = kwargs.get("variables", {})
        if not path:
            return ToolResult(success=False, error="path required")

        proc = await asyncio.create_subprocess_exec(
            "cookiecutter", "--version",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        await proc.communicate()
        if proc.returncode != 0:
            return ToolResult(success=False, error="Custom templates require cookiecutter. Install with: pip install cookiecutter")

        template_url = kwargs.get("template", "")
        if not template_url:
            return ToolResult(success=False, error="template URL/path required for custom action")

        cmd = ["cookiecutter", template_url, "-o", path, "--no-input"]
        for k, v in variables.items():
            cmd.append(k + "=" + str(v))

        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            return ToolResult(success=False, error="cookiecutter failed: " + stderr.decode())

        return ToolResult(success=True, output="Created from cookiecutter template:\n" + stdout.decode())
