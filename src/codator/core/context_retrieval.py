"""Contextual code indexing and retrieval for enriched LLM context.

Provides semantic chunking of source files with parent context (class hierarchy,
module info) and TF-IDF-based retrieval so the LLM receives the most relevant
code fragments alongside each query.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codator.config import get_settings
from codator.core.project_indexer import EXTENSION_TO_LANG

if TYPE_CHECKING:
    from codator.domain.models import Symbol

logger = logging.getLogger(__name__)

# Extensions we support for chunking (code-only subset of index_extensions)
_CODE_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".rb"}
)

_MAX_FILE_SIZE = 512 * 1024  # 512 KB

# Line-based fallback chunk size when tree-sitter symbols are unavailable
_FALLBACK_CHUNK_LINES = 50
_FALLBACK_OVERLAP_LINES = 10


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """A semantically meaningful fragment of source code."""

    file_path: str
    start_line: int
    end_line: int
    content: str
    parent_context: str
    language: str
    chunk_type: str  # "function", "class", "method", "module_header", "block"
    imports: list[str] = field(default_factory=list)
    docstring: str = ""


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-zA-Z_]\w*")


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


# ---------------------------------------------------------------------------
# Contextual index
# ---------------------------------------------------------------------------

class ContextualIndex:
    """Indexes a project into semantic :class:`CodeChunk` objects and supports
    TF-IDF keyword search for retrieval-augmented generation."""

    def __init__(self, project_root: str) -> None:
        self._root = Path(project_root).resolve()
        self._chunks: list[CodeChunk] = []

        # TF-IDF structures (built lazily)
        self._idf: dict[str, float] = {}
        self._tf_vectors: list[dict[str, float]] = []
        self._index_dirty = True

        # Optional tree-sitter integration
        self._ts_available = False
        self._parsers: dict[str, object] = {}
        self._init_tree_sitter()

    # -- tree-sitter bootstrap (matches project_indexer.py pattern) ----------

    def _init_tree_sitter(self) -> None:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=FutureWarning, module="tree_sitter"
                )
                import tree_sitter_languages  # noqa: F401

            self._ts_available = True
        except ImportError:
            logger.info(
                "tree-sitter-languages not available; "
                "falling back to line-based chunking"
            )

    def _get_parser(self, language: str):
        if not self._ts_available:
            return None
        if language not in self._parsers:
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", category=FutureWarning, module="tree_sitter"
                    )
                    import tree_sitter_languages

                    self._parsers[language] = tree_sitter_languages.get_parser(language)
            except Exception:
                self._parsers[language] = None
        return self._parsers[language]

    # -- public API ----------------------------------------------------------

    async def index_project(self) -> int:
        """Index all supported source files under *project_root*.

        Returns the total number of :class:`CodeChunk` objects produced.
        """
        settings = get_settings()
        exclude_dirs = set(settings.project.exclude_dirs)

        loop = asyncio.get_running_loop()
        files = await loop.run_in_executor(
            None, lambda: self._collect_files(exclude_dirs)
        )

        self._chunks.clear()
        self._index_dirty = True

        batch_size = 50
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            results = await asyncio.gather(
                *(
                    loop.run_in_executor(None, self._process_file, fp)
                    for fp in batch
                ),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, list):
                    self._chunks.extend(result)
                elif isinstance(result, BaseException):
                    logger.debug("Failed to process file: %s", result)

        logger.info(
            "Indexed %d chunks from %d files in %s",
            len(self._chunks),
            len(files),
            self._root,
        )
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[CodeChunk]:
        """Return the *top_k* chunks most relevant to *query* using TF-IDF."""
        if not self._chunks:
            return []

        if self._index_dirty:
            self._build_tfidf_index()

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        # Build query TF vector
        query_tf: dict[str, float] = defaultdict(float)
        for tok in query_tokens:
            query_tf[tok] += 1.0
        max_tf = max(query_tf.values())
        query_vec = {
            tok: (0.5 + 0.5 * tf / max_tf) * self._idf.get(tok, 0.0)
            for tok, tf in query_tf.items()
        }

        # Score each chunk
        scored: list[tuple[float, int]] = []
        for idx, doc_vec in enumerate(self._tf_vectors):
            score = sum(
                query_vec.get(tok, 0.0) * weight
                for tok, weight in doc_vec.items()
            )
            if score > 0:
                scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._chunks[idx] for _, idx in scored[:top_k]]

    def format_chunks_for_prompt(self, chunks: list[CodeChunk]) -> str:
        """Format retrieved chunks as a context block for LLM prompts."""
        if not chunks:
            return ""

        parts = ["--- Relevant Code ---"]
        for chunk in chunks:
            header_lines = [
                f"# file: {chunk.file_path} (lines {chunk.start_line}-{chunk.end_line})"
            ]
            if chunk.parent_context:
                header_lines.append(f"# context: {chunk.parent_context}")
            parts.append("\n".join(header_lines))
            parts.append(chunk.content)
        parts.append("--- End Relevant Code ---")
        return "\n\n".join(parts)

    # -- file collection -----------------------------------------------------

    def _collect_files(self, exclude_dirs: set[str]) -> list[Path]:
        files: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(self._root):
            dirnames[:] = [
                d for d in dirnames if d not in exclude_dirs and not d.startswith(".")
            ]
            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix not in _CODE_EXTENSIONS:
                    continue
                try:
                    if fpath.stat().st_size <= _MAX_FILE_SIZE:
                        files.append(fpath)
                except OSError:
                    continue
        return files

    # -- per-file processing -------------------------------------------------

    def _process_file(self, file_path: Path) -> list[CodeChunk]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        language = EXTENSION_TO_LANG.get(file_path.suffix, file_path.suffix.lstrip("."))
        rel_path = str(file_path.relative_to(self._root))

        chunks = self._chunk_file(rel_path, content, language)
        chunks = self._add_parent_context(chunks, rel_path)
        return chunks

    # -- chunking ------------------------------------------------------------

    def _chunk_file(
        self, file_path: str, content: str, language: str
    ) -> list[CodeChunk]:
        """Split a file into semantic chunks.

        Uses tree-sitter symbol boundaries when available, otherwise falls back
        to overlapping line-based windows of ~50 lines.
        """
        lines = content.split("\n")
        if not lines:
            return []

        # Extract file-level imports (used to annotate every chunk)
        file_imports = self._extract_imports(lines, language)

        # Try tree-sitter based chunking
        symbols = self._extract_symbols_ts(content, language)
        if symbols:
            return self._chunk_from_symbols(
                file_path, content, lines, language, symbols, file_imports
            )

        # Fallback: line-based chunking
        return self._chunk_line_based(file_path, content, lines, language, file_imports)

    def _chunk_from_symbols(
        self,
        file_path: str,
        content: str,
        lines: list[str],
        language: str,
        symbols: list[Symbol],
        file_imports: list[str],
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []

        # Module header: everything before the first symbol
        first_sym_line = min(s.line for s in symbols)
        if first_sym_line > 1:
            header_content = "\n".join(lines[: first_sym_line - 1]).rstrip()
            if header_content.strip():
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=1,
                        end_line=first_sym_line - 1,
                        content=header_content,
                        parent_context="",
                        language=language,
                        chunk_type="module_header",
                        imports=file_imports,
                        docstring=self._extract_docstring(header_content, language),
                    )
                )

        # One chunk per symbol
        for sym in symbols:
            start = sym.line
            end = sym.end_line or sym.line
            chunk_content = "\n".join(lines[start - 1 : end]).rstrip()
            if not chunk_content.strip():
                continue

            chunk_type = sym.kind.value if hasattr(sym.kind, "value") else str(sym.kind)
            chunks.append(
                CodeChunk(
                    file_path=file_path,
                    start_line=start,
                    end_line=end,
                    content=chunk_content,
                    parent_context="",
                    language=language,
                    chunk_type=chunk_type,
                    imports=file_imports,
                    docstring=self._extract_docstring(chunk_content, language),
                )
            )

        return chunks

    def _chunk_line_based(
        self,
        file_path: str,
        content: str,
        lines: list[str],
        language: str,
        file_imports: list[str],
    ) -> list[CodeChunk]:
        chunks: list[CodeChunk] = []
        total = len(lines)
        start = 0

        while start < total:
            end = min(start + _FALLBACK_CHUNK_LINES, total)
            chunk_content = "\n".join(lines[start:end]).rstrip()
            if chunk_content.strip():
                chunks.append(
                    CodeChunk(
                        file_path=file_path,
                        start_line=start + 1,
                        end_line=end,
                        content=chunk_content,
                        parent_context="",
                        language=language,
                        chunk_type="block",
                        imports=file_imports if start == 0 else [],
                        docstring="",
                    )
                )
            start = end - _FALLBACK_OVERLAP_LINES if end < total else total

        return chunks

    # -- parent context enrichment -------------------------------------------

    def _add_parent_context(
        self, chunks: list[CodeChunk], file_path: str
    ) -> list[CodeChunk]:
        """Enrich chunks with hierarchical parent context.

        Builds context strings like ``class ChatEngine > method chat_stream``
        or ``module config > function load_config`` by looking at surrounding
        class/module structure.
        """
        if not chunks:
            return chunks

        module_name = Path(file_path).stem

        # Identify class chunks and their line ranges for nesting detection
        class_ranges: list[tuple[int, int, str]] = []
        for chunk in chunks:
            if chunk.chunk_type == "class":
                name = self._name_from_content(chunk.content, chunk.language)
                if name:
                    class_ranges.append((chunk.start_line, chunk.end_line, name))

        # Extract module-level docstring from the header chunk (if any)
        module_doc = ""
        for chunk in chunks:
            if chunk.chunk_type == "module_header" and chunk.docstring:
                module_doc = chunk.docstring
                break

        for chunk in chunks:
            parts: list[str] = []

            if chunk.chunk_type == "module_header":
                parts.append(f"module {module_name}")
                if module_doc:
                    chunk.docstring = module_doc
            else:
                # Check if this chunk is nested inside a class
                enclosing_class = self._find_enclosing_class(
                    chunk.start_line, chunk.end_line, class_ranges
                )
                if enclosing_class:
                    parts.append(f"class {enclosing_class}")
                else:
                    parts.append(f"module {module_name}")

                chunk_name = self._name_from_content(chunk.content, chunk.language)
                label = chunk.chunk_type
                if chunk_name:
                    parts.append(f"{label} {chunk_name}")

            chunk.parent_context = " > ".join(parts)

        return chunks

    # -- tree-sitter symbol extraction (local to this module) ----------------

    def _extract_symbols_ts(self, content: str, language: str) -> list[Symbol]:
        """Extract symbols from *content* using tree-sitter.

        Returns an empty list when tree-sitter is unavailable or the language
        is not supported.
        """
        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            tree = parser.parse(content.encode("utf-8"))
        except Exception:
            return []

        from codator.core.project_indexer import SYMBOL_QUERIES

        query_map = SYMBOL_QUERIES.get(language, {})
        if not query_map:
            return []

        symbols: list[Symbol] = []
        self._walk_tree_for_symbols(tree.root_node, query_map, symbols, depth=0)
        return symbols

    def _walk_tree_for_symbols(
        self,
        node,
        query_map: dict,
        symbols: list[Symbol],
        depth: int,
    ) -> None:
        if depth > 3:
            return

        from codator.domain.models import Symbol

        if node.type in query_map:
            name = self._ts_extract_name(node)
            if name:
                kind = query_map[node.type]
                symbols.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        file_path="",
                        line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    )
                )

        for child in node.children:
            self._walk_tree_for_symbols(child, query_map, symbols, depth + 1)

    @staticmethod
    def _ts_extract_name(node) -> str:
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8", errors="replace")
        if node.named_children:
            first = node.named_children[0]
            if first.type in ("identifier", "type_identifier"):
                return first.text.decode("utf-8", errors="replace")
        return ""

    # -- extraction helpers --------------------------------------------------

    @staticmethod
    def _extract_imports(lines: list[str], language: str) -> list[str]:
        """Extract import statements from the beginning of a file."""
        imports: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                continue
            if language == "python" and (
                stripped.startswith("import ") or stripped.startswith("from ")
            ):
                imports.append(stripped)
            elif language in ("javascript", "typescript") and stripped.startswith(
                "import "
            ):
                imports.append(stripped)
            elif language == "go" and stripped.startswith("import"):
                imports.append(stripped)
            elif language == "rust" and stripped.startswith("use "):
                imports.append(stripped)
            elif language == "java" and stripped.startswith("import "):
                imports.append(stripped)
            elif language == "ruby" and (
                stripped.startswith("require ") or stripped.startswith("require_relative ")
            ):
                imports.append(stripped)
            elif language in ("c", "cpp") and stripped.startswith("#include"):
                imports.append(stripped)
            # Stop scanning after the import block (first non-import, non-blank,
            # non-comment, non-decorator, non-docstring line)
            elif (
                not stripped.startswith("@")
                and not stripped.startswith('"""')
                and not stripped.startswith("'''")
                and not stripped.startswith("package ")
                and not stripped.startswith("from __future__")
                and not stripped.startswith("#!")
                and not stripped.startswith("//!")
                and imports
            ):
                break
        return imports

    @staticmethod
    def _extract_docstring(content: str, language: str) -> str:
        """Extract leading docstring from a chunk of code."""
        if language == "python":
            # Triple-quoted string at the start of a definition
            for quote in ('"""', "'''"):
                idx = content.find(quote)
                if idx == -1:
                    continue
                end = content.find(quote, idx + 3)
                if end != -1:
                    return content[idx + 3 : end].strip()
        elif language in ("javascript", "typescript", "java", "go", "rust", "c", "cpp"):
            # /** ... */ or /// style docstrings
            match = re.search(r"/\*\*(.*?)\*/", content, re.DOTALL)
            if match:
                raw = match.group(1)
                lines = [
                    ln.strip().lstrip("*").strip() for ln in raw.split("\n")
                ]
                return "\n".join(ln for ln in lines if ln).strip()
            # Consecutive /// comments
            doc_lines: list[str] = []
            for line in content.split("\n"):
                stripped = line.strip()
                if stripped.startswith("///"):
                    doc_lines.append(stripped[3:].strip())
                elif doc_lines:
                    break
            if doc_lines:
                return "\n".join(doc_lines)
        return ""

    @staticmethod
    def _name_from_content(content: str, language: str) -> str:
        """Extract the definition name from the first line of a chunk."""
        first_line = content.strip().split("\n")[0]
        # Match common definition patterns
        m = re.search(
            r"\b(?:def|fn|func|function|class|struct|impl|trait|enum|interface|type)\s+"
            r"([A-Za-z_]\w*)",
            first_line,
        )
        return m.group(1) if m else ""

    @staticmethod
    def _find_enclosing_class(
        start: int, end: int, class_ranges: list[tuple[int, int, str]]
    ) -> str | None:
        """Return the name of the class whose range encloses the given lines."""
        for cls_start, cls_end, cls_name in class_ranges:
            if cls_start < start and end <= cls_end:
                return cls_name
        return None

    # -- TF-IDF index --------------------------------------------------------

    def _build_tfidf_index(self) -> None:
        """Build an inverted index with TF-IDF weights over all chunks."""
        n_docs = len(self._chunks)
        if n_docs == 0:
            return

        # Document frequency for each term
        df: dict[str, int] = defaultdict(int)
        doc_tokens: list[list[str]] = []

        for chunk in self._chunks:
            tokens = _tokenize(chunk.content)
            # Include parent context and docstring for richer matching
            tokens.extend(_tokenize(chunk.parent_context))
            if chunk.docstring:
                tokens.extend(_tokenize(chunk.docstring))
            doc_tokens.append(tokens)
            seen: set[str] = set()
            for tok in tokens:
                if tok not in seen:
                    df[tok] += 1
                    seen.add(tok)

        # IDF: log(N / df) with smoothing
        self._idf = {
            tok: math.log((n_docs + 1) / (freq + 1)) + 1.0
            for tok, freq in df.items()
        }

        # TF-IDF vector per document (augmented TF: 0.5 + 0.5 * tf/max_tf)
        self._tf_vectors = []
        for tokens in doc_tokens:
            tf: dict[str, float] = defaultdict(float)
            for tok in tokens:
                tf[tok] += 1.0
            if not tf:
                self._tf_vectors.append({})
                continue
            max_tf = max(tf.values())
            vec = {
                tok: (0.5 + 0.5 * count / max_tf) * self._idf.get(tok, 0.0)
                for tok, count in tf.items()
            }
            self._tf_vectors.append(vec)

        self._index_dirty = False
        logger.debug("Built TF-IDF index over %d chunks", n_docs)
