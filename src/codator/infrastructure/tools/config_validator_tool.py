"""Config validator tool — validate YAML/JSON/TOML against schemas, detect misconfigs."""

import json
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


class ConfigValidatorTool:
    name = "config_validator"
    description = (
        "Validate configuration files: check YAML/JSON/TOML/INI syntax, "
        "validate against JSON Schema, detect common misconfigurations, "
        "check environment variable references, and verify required fields."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["validate", "schema_check", "lint", "env_check", "compare"],
                "description": "Validation action: syntax check, schema validation, lint for issues, check env vars, or compare configs",
            },
            "path": {
                "type": "string",
                "description": "Config file path to validate",
            },
            "schema_path": {
                "type": "string",
                "description": "JSON Schema file path (for schema_check action)",
            },
            "required_fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of required field paths (dot notation, e.g., 'database.host')",
            },
            "compare_with": {
                "type": "string",
                "description": "Path to another config file to compare with",
            },
        },
        "required": ["action", "path"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs["path"]

        if not os.path.isfile(path):
            return ToolResult(success=False, error="File not found: " + path)

        try:
            if action == "validate":
                return self._validate_syntax(path, kwargs)
            elif action == "schema_check":
                return self._schema_check(path, kwargs)
            elif action == "lint":
                return self._lint_config(path)
            elif action == "env_check":
                return self._check_env_vars(path)
            elif action == "compare":
                return self._compare_configs(path, kwargs)
            else:
                return ToolResult(success=False, error="Unknown action: " + action)
        except Exception as e:
            return ToolResult(success=False, error=str(e))

    def _detect_format(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        format_map = {
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
            ".toml": "toml", ".ini": "ini", ".cfg": "ini",
            ".env": "env", ".properties": "properties",
        }
        return format_map.get(ext, "unknown")

    def _parse_config(self, path: str) -> tuple:
        """Parse config file, return (data, format, error)."""
        fmt = self._detect_format(path)
        with open(path, "r") as f:
            content = f.read()

        if fmt == "json":
            try:
                data = json.loads(content)
                return data, fmt, None
            except json.JSONDecodeError as e:
                return None, fmt, str(e)

        elif fmt == "yaml":
            try:
                import yaml
                data = yaml.safe_load(content)
                return data, fmt, None
            except ImportError:
                return None, fmt, "PyYAML not installed"
            except yaml.YAMLError as e:
                return None, fmt, str(e)

        elif fmt == "toml":
            try:
                import tomllib
                data = tomllib.loads(content)
                return data, fmt, None
            except ImportError:
                try:
                    import tomli as tomllib
                    data = tomllib.loads(content)
                    return data, fmt, None
                except ImportError:
                    return None, fmt, "tomllib/tomli not available"
            except Exception as e:
                return None, fmt, str(e)

        elif fmt == "ini":
            import configparser
            parser = configparser.ConfigParser()
            try:
                parser.read_string(content)
                data = {s: dict(parser[s]) for s in parser.sections()}
                return data, fmt, None
            except configparser.Error as e:
                return None, fmt, str(e)

        elif fmt == "env":
            data = {}
            for line in content.split("\n"):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    data[key.strip()] = value.strip().strip("\"'")
            return data, fmt, None

        return None, fmt, "Unsupported format: " + fmt

    def _validate_syntax(self, path: str, kwargs: dict) -> ToolResult:
        """Validate config file syntax."""
        data, fmt, error = self._parse_config(path)
        required = kwargs.get("required_fields", [])

        if error:
            return ToolResult(success=False, output="Syntax error in " + path + " (" + fmt + "):\n  " + error)

        issues = []

        # Check required fields
        if required and isinstance(data, dict):
            for field_path in required:
                value = self._get_nested(data, field_path)
                if value is None:
                    issues.append("Missing required field: " + field_path)

        if issues:
            return ToolResult(
                success=False,
                output="Validation issues:\n  " + "\n  ".join(issues),
                artifacts={"format": fmt, "issues": len(issues)},
            )

        keys_count = self._count_keys(data) if isinstance(data, dict) else 0
        return ToolResult(
            success=True,
            output="Valid " + fmt.upper() + " file: " + path + " (" + str(keys_count) + " keys)",
            artifacts={"format": fmt, "keys": keys_count},
        )

    def _get_nested(self, data: dict, path: str):
        """Get nested value by dot-notation path."""
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _count_keys(self, data, depth=0) -> int:
        if depth > 10:
            return 0
        count = 0
        if isinstance(data, dict):
            count += len(data)
            for v in data.values():
                count += self._count_keys(v, depth + 1)
        elif isinstance(data, list):
            for item in data:
                count += self._count_keys(item, depth + 1)
        return count

    def _schema_check(self, path: str, kwargs: dict) -> ToolResult:
        """Validate config against JSON Schema."""
        schema_path = kwargs.get("schema_path", "")
        if not schema_path or not os.path.isfile(schema_path):
            return ToolResult(success=False, error="schema_path is required and must exist")

        data, fmt, error = self._parse_config(path)
        if error:
            return ToolResult(success=False, error="Parse error: " + error)

        with open(schema_path) as f:
            schema = json.load(f)

        try:
            import jsonschema
            validator = jsonschema.Draft7Validator(schema)
            errors = list(validator.iter_errors(data))

            if not errors:
                return ToolResult(success=True, output="Schema validation passed")

            lines = ["Schema validation failed (" + str(len(errors)) + " errors):\n"]
            for err in errors[:20]:
                path_str = ".".join(str(p) for p in err.absolute_path)
                lines.append("  " + path_str + ": " + err.message)

            return ToolResult(success=False, output="\n".join(lines), artifacts={"errors": len(errors)})

        except ImportError:
            return ToolResult(success=False, error="jsonschema not installed. Install with: pip install jsonschema")

    def _lint_config(self, path: str) -> ToolResult:
        """Lint config for common issues."""
        with open(path, "r") as f:
            content = f.read()

        data, fmt, error = self._parse_config(path)
        issues = []

        if error:
            issues.append(("ERROR", "Syntax error: " + error))

        # Common checks
        if fmt in ("yaml", "json", "toml"):
            # Check for duplicate keys (YAML/JSON)
            if fmt == "yaml":
                seen_keys = set()
                for line in content.split("\n"):
                    match = re.match(r'^(\w[\w.-]*):', line)
                    if match:
                        key = match.group(1)
                        if key in seen_keys:
                            issues.append(("WARNING", "Possible duplicate key: " + key))
                        seen_keys.add(key)

            # Check for TODO/FIXME in configs
            for i, line in enumerate(content.split("\n"), 1):
                if re.search(r'(?i)\b(TODO|FIXME|HACK|XXX)\b', line):
                    issues.append(("INFO", "Line " + str(i) + ": Contains TODO/FIXME marker"))

            # Check for hardcoded credentials
            secret_patterns = [
                (r'(?i)(password|secret|token|api.?key)\s*[:=]\s*["\']?(?![\s{$])\S+', "Possible hardcoded credential"),
                (r'(?i)localhost|127\.0\.0\.1', "Localhost reference (may not work in production)"),
            ]
            for pattern, msg in secret_patterns:
                for i, line in enumerate(content.split("\n"), 1):
                    if line.strip().startswith("#") or line.strip().startswith("//"):
                        continue
                    if re.search(pattern, line):
                        issues.append(("WARNING", "Line " + str(i) + ": " + msg))

        # Check for empty values
        if isinstance(data, dict):
            self._check_empty_values(data, "", issues)

        if not issues:
            return ToolResult(success=True, output="No issues found in " + path)

        lines = ["Config lint results for " + path + ":\n"]
        for severity, msg in issues[:30]:
            lines.append("  [" + severity + "] " + msg)

        has_errors = any(s == "ERROR" for s, _ in issues)
        return ToolResult(
            success=not has_errors,
            output="\n".join(lines),
            artifacts={"issues": len(issues), "errors": sum(1 for s, _ in issues if s == "ERROR")},
        )

    def _check_empty_values(self, data, prefix, issues, depth=0):
        if depth > 5:
            return
        if isinstance(data, dict):
            for k, v in data.items():
                path = prefix + "." + k if prefix else k
                if v is None or v == "":
                    issues.append(("INFO", "Empty value: " + path))
                elif isinstance(v, (dict, list)):
                    self._check_empty_values(v, path, issues, depth + 1)

    def _check_env_vars(self, path: str) -> ToolResult:
        """Check that referenced environment variables are set."""
        with open(path, "r") as f:
            content = f.read()

        # Find env var references
        env_patterns = [
            re.compile(r'\$\{(\w+)\}'),  # ${VAR}
            re.compile(r'\$(\w+)'),  # $VAR
            re.compile(r'%(\w+)%'),  # %VAR% (Windows)
        ]

        referenced = set()
        for pattern in env_patterns:
            referenced.update(pattern.findall(content))

        if not referenced:
            return ToolResult(success=True, output="No environment variable references found in " + path)

        missing = []
        present = []
        for var in sorted(referenced):
            if var in os.environ:
                present.append(var)
            else:
                missing.append(var)

        lines = ["Environment variable check for " + path + ":\n"]
        lines.append("  Referenced: " + str(len(referenced)))
        lines.append("  Present: " + str(len(present)))
        lines.append("  Missing: " + str(len(missing)))

        if missing:
            lines.append("\nMissing variables:")
            for var in missing:
                lines.append("  " + var)

        return ToolResult(
            success=len(missing) == 0,
            output="\n".join(lines),
            artifacts={"referenced": list(referenced), "missing": missing},
        )

    def _compare_configs(self, path: str, kwargs: dict) -> ToolResult:
        """Compare two config files."""
        compare_with = kwargs.get("compare_with", "")
        if not compare_with or not os.path.isfile(compare_with):
            return ToolResult(success=False, error="compare_with path is required")

        data1, fmt1, err1 = self._parse_config(path)
        data2, fmt2, err2 = self._parse_config(compare_with)

        if err1:
            return ToolResult(success=False, error="Error parsing " + path + ": " + err1)
        if err2:
            return ToolResult(success=False, error="Error parsing " + compare_with + ": " + err2)

        if not isinstance(data1, dict) or not isinstance(data2, dict):
            return ToolResult(success=False, error="Both configs must be dictionaries for comparison")

        keys1 = set(self._flatten_keys(data1))
        keys2 = set(self._flatten_keys(data2))

        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        common = keys1 & keys2

        # Check value differences
        different_values = []
        for key in sorted(common):
            v1 = self._get_nested(data1, key)
            v2 = self._get_nested(data2, key)
            if v1 != v2:
                different_values.append((key, v1, v2))

        lines = ["Config comparison:", "  " + path + " vs " + compare_with + "\n"]
        lines.append("  Common keys: " + str(len(common)))
        lines.append("  Only in first: " + str(len(only_in_1)))
        lines.append("  Only in second: " + str(len(only_in_2)))
        lines.append("  Different values: " + str(len(different_values)))

        if only_in_1:
            lines.append("\nOnly in " + os.path.basename(path) + ":")
            for k in sorted(only_in_1)[:15]:
                lines.append("  + " + k)

        if only_in_2:
            lines.append("\nOnly in " + os.path.basename(compare_with) + ":")
            for k in sorted(only_in_2)[:15]:
                lines.append("  + " + k)

        if different_values:
            lines.append("\nDifferent values:")
            for key, v1, v2 in different_values[:15]:
                lines.append("  " + key + ": " + repr(v1)[:50] + " vs " + repr(v2)[:50])

        return ToolResult(
            success=True,
            output="\n".join(lines)[:5000],
            artifacts={"only_first": len(only_in_1), "only_second": len(only_in_2), "different": len(different_values)},
        )

    def _flatten_keys(self, data, prefix="", depth=0):
        if depth > 8:
            return []
        keys = []
        if isinstance(data, dict):
            for k, v in data.items():
                path = prefix + "." + k if prefix else k
                keys.append(path)
                if isinstance(v, dict):
                    keys.extend(self._flatten_keys(v, path, depth + 1))
        return keys
