"""Document Reader Tool — read PDFs, XLSX, CSV, Markdown with structure.

Extracts text, tables, and metadata from various document formats.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
from pathlib import Path
from typing import Any

from codator.domain.interfaces import Tool
from codator.domain.models import ToolResult

logger = logging.getLogger(__name__)


class DocReaderTool(Tool):
    """Read and extract content from PDF, XLSX, CSV, and Markdown files."""

    def __init__(self, cwd: str | Path = ""):
        self._cwd = Path(cwd) if cwd else Path.cwd()

    @property
    def name(self) -> str:
        return "doc_reader"

    @property
    def description(self) -> str:
        return (
            "Read documents and extract structured content. "
            "Supports PDF (text + tables), XLSX/CSV (tabular data), "
            "Markdown (sections + structure). Returns parsed text with metadata."
        )

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to document file."},
                "format": {
                    "type": "string",
                    "enum": ["auto", "pdf", "xlsx", "csv", "markdown", "json", "yaml"],
                    "description": "Document format (default: auto-detect).",
                },
                "page_range": {
                    "type": "string",
                    "description": "Page range for PDF (e.g., '1-5', '2,4,6').",
                },
                "sheet": {"type": "string", "description": "Sheet name for XLSX."},
                "max_rows": {"type": "integer", "description": "Max rows (default: 100)."},
                "extract": {
                    "type": "string",
                    "enum": ["text", "tables", "metadata", "structure", "all"],
                    "description": "What to extract (default: 'all').",
                },
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        path_str = kwargs.get("path", "")
        if not path_str:
            return ToolResult(success=False, error="'path' required.")

        path = Path(path_str)
        if not path.is_absolute():
            path = self._cwd / path
        if not path.exists():
            return ToolResult(success=False, error=f"File not found: {path_str}")

        fmt = kwargs.get("format", "auto")
        if fmt == "auto":
            fmt = self._detect_format(path)

        dispatch = {
            "pdf": self._read_pdf,
            "xlsx": self._read_xlsx,
            "csv": self._read_csv,
            "markdown": self._read_markdown,
            "json": self._read_json,
            "yaml": self._read_yaml,
        }

        handler = dispatch.get(fmt)
        if not handler:
            return ToolResult(success=False, error=f"Unsupported format: {fmt}")

        return handler(path, **kwargs)

    def _detect_format(self, path: Path) -> str:
        ext = path.suffix.lower()
        mapping = {
            ".pdf": "pdf", ".xlsx": "xlsx", ".xls": "xlsx",
            ".csv": "csv", ".tsv": "csv",
            ".md": "markdown", ".markdown": "markdown",
            ".json": "json", ".yaml": "yaml", ".yml": "yaml",
        }
        return mapping.get(ext, "markdown")

    def _read_pdf(self, path: Path, **kwargs) -> ToolResult:
        try:
            import pymupdf
        except ImportError:
            try:
                import fitz as pymupdf
            except ImportError:
                return ToolResult(
                    success=False,
                    error="PDF requires PyMuPDF. Install: pip install pymupdf",
                )

        page_range = kwargs.get("page_range", "")

        try:
            doc = pymupdf.open(str(path))
            pages_to_read = self._parse_page_range(page_range, len(doc))

            text_parts = []
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "pages": len(doc),
            }

            for pnum in pages_to_read:
                page = doc[pnum]
                text_parts.append(f"--- Page {pnum + 1} ---\n{page.get_text()}")

            doc.close()

            num_pages = metadata['pages']
            output = f"PDF: {path.name} ({num_pages} pages)\n"
            if metadata["title"]:
                output += f"Title: {metadata['title']}\n"
            output += "\n".join(text_parts)

            if len(output) > 10000:
                output = output[:10000] + "\n... [truncated - use page_range]"

            return ToolResult(success=True, output=output, artifacts=metadata)
        except Exception as e:
            return ToolResult(success=False, error=f"PDF read failed: {e}")

    def _read_xlsx(self, path: Path, **kwargs) -> ToolResult:
        try:
            import openpyxl
        except ImportError:
            return ToolResult(
                success=False,
                error="XLSX requires openpyxl. Install: pip install openpyxl",
            )

        sheet_name = kwargs.get("sheet", "")
        max_rows = kwargs.get("max_rows", 100)

        try:
            wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
            sheets = wb.sheetnames
            ws = wb[sheet_name] if sheet_name and sheet_name in sheets else wb.active

            rows = []
            headers = []
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i == 0:
                    headers = [str(c) if c else f"col_{j}" for j, c in enumerate(row)]
                else:
                    rows.append(dict(zip(headers, row)))
                if i >= max_rows:
                    break

            wb.close()

            col_str = ", ".join(headers)
            sheet_str = ", ".join(sheets)
            output_parts = [
                f"XLSX: {path.name} (sheets: {sheet_str})",
                f"Active: {ws.title} - {len(rows)} rows x {len(headers)} cols",
                f"Columns: {col_str}",
                "",
            ]
            for i, row in enumerate(rows[:50]):
                output_parts.append(f"  {i+1}. {row}")
            if len(rows) > 50:
                remaining = len(rows) - 50
                output_parts.append(f"  ... ({remaining} more rows)")

            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                artifacts={"headers": headers, "rows": rows[:max_rows], "sheets": sheets},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"XLSX read failed: {e}")

    def _read_csv(self, path: Path, **kwargs) -> ToolResult:
        max_rows = kwargs.get("max_rows", 100)
        try:
            content = path.read_text(errors="replace")
            delimiter = "," if path.suffix == ".csv" else "\t"
            reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
            headers = reader.fieldnames or []
            rows = []
            for i, row in enumerate(reader):
                rows.append(dict(row))
                if i >= max_rows:
                    break

            col_str = ", ".join(headers)
            output_parts = [
                f"CSV: {path.name} - {len(rows)} rows x {len(headers)} cols",
                f"Columns: {col_str}",
                "",
            ]
            for i, row in enumerate(rows[:50]):
                output_parts.append(f"  {i+1}. {row}")

            return ToolResult(
                success=True,
                output="\n".join(output_parts),
                artifacts={"headers": list(headers), "rows": rows},
            )
        except Exception as e:
            return ToolResult(success=False, error=f"CSV read failed: {e}")

    def _read_markdown(self, path: Path, **kwargs) -> ToolResult:
        content = path.read_text(errors="replace")
        extract = kwargs.get("extract", "all")

        headings = re.findall(r"^(#{1,6})\s+(.+)$", content, re.M)
        structure = []
        for h in headings:
            indent = "  " * (len(h[0]) - 1)
            structure.append(f"{indent}{h[1]}")
        code_blocks = len(re.findall(r"^```", content, re.M)) // 2
        links = len(re.findall(r"\[.+?\]\(.+?\)", content))
        word_count = len(content.split())

        output_parts = [
            f"Markdown: {path.name} ({word_count} words, {code_blocks} code blocks, {links} links)",
        ]

        if structure and extract in ("structure", "all"):
            output_parts.append("\nStructure:")
            output_parts.extend(f"  {s}" for s in structure[:30])

        if extract in ("text", "all"):
            text = content[:8000]
            if len(content) > 8000:
                text += "\n... [truncated]"
            output_parts.append(f"\n{text}")

        return ToolResult(
            success=True,
            output="\n".join(output_parts),
            artifacts={"headings": [h[1] for h in headings], "word_count": word_count},
        )

    def _read_json(self, path: Path, **kwargs) -> ToolResult:
        try:
            data = json.loads(path.read_text(errors="replace"))
            if isinstance(data, list):
                output = f"JSON Array: {len(data)} items\n"
                output += json.dumps(data[:10], indent=2, ensure_ascii=False)
                if len(data) > 10:
                    remaining = len(data) - 10
                    output += f"\n... ({remaining} more items)"
            else:
                num_keys = len(data) if isinstance(data, dict) else 0
                output = f"JSON Object: {num_keys} keys\n"
                output += json.dumps(data, indent=2, ensure_ascii=False)[:5000]
            return ToolResult(success=True, output=output, artifacts={"data": data})
        except Exception as e:
            return ToolResult(success=False, error=f"JSON parse failed: {e}")

    def _read_yaml(self, path: Path, **kwargs) -> ToolResult:
        try:
            import yaml
        except ImportError:
            return ToolResult(
                success=False,
                error="YAML requires PyYAML. Install: pip install pyyaml",
            )
        try:
            data = yaml.safe_load(path.read_text(errors="replace"))
            output = f"YAML: {path.name}\n"
            output += json.dumps(data, indent=2, ensure_ascii=False, default=str)[:5000]
            return ToolResult(success=True, output=output, artifacts={"data": data})
        except Exception as e:
            return ToolResult(success=False, error=f"YAML parse failed: {e}")

    @staticmethod
    def _parse_page_range(spec: str, total: int) -> list[int]:
        if not spec:
            return list(range(min(total, 20)))
        pages = set()
        for part in spec.split(","):
            if "-" in part:
                start, end = part.split("-", 1)
                pages.update(range(int(start) - 1, min(int(end), total)))
            else:
                p = int(part) - 1
                if 0 <= p < total:
                    pages.add(p)
        return sorted(pages)
