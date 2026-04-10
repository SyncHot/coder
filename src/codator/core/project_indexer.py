"""Asynchronous project indexer using tree-sitter for symbol extraction."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from codator.config import get_settings
from codator.domain.interfaces import ProjectIndexer
from codator.domain.models import FileInfo, ProjectMap, Symbol, SymbolKind

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Language extension mapping
EXTENSION_TO_LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".jsx": "javascript", ".tsx": "typescript",
    ".rs": "rust", ".go": "go",
    ".c": "c", ".cpp": "cpp", ".h": "c",
    ".java": "java", ".rb": "ruby", ".php": "php",
    ".sh": "bash",
}

# tree-sitter node types → our SymbolKind mapping (per language)
SYMBOL_QUERIES: dict[str, dict[str, SymbolKind]] = {
    "python": {
        "function_definition": SymbolKind.FUNCTION,
        "class_definition": SymbolKind.CLASS,
        "import_statement": SymbolKind.IMPORT,
        "import_from_statement": SymbolKind.IMPORT,
    },
    "javascript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
        "arrow_function": SymbolKind.FUNCTION,
        "import_statement": SymbolKind.IMPORT,
    },
    "typescript": {
        "function_declaration": SymbolKind.FUNCTION,
        "class_declaration": SymbolKind.CLASS,
        "method_definition": SymbolKind.METHOD,
        "arrow_function": SymbolKind.FUNCTION,
        "import_statement": SymbolKind.IMPORT,
        "interface_declaration": SymbolKind.CLASS,
        "type_alias_declaration": SymbolKind.CLASS,
    },
    "rust": {
        "function_item": SymbolKind.FUNCTION,
        "struct_item": SymbolKind.CLASS,
        "impl_item": SymbolKind.CLASS,
        "enum_item": SymbolKind.CLASS,
        "trait_item": SymbolKind.CLASS,
        "use_declaration": SymbolKind.IMPORT,
    },
    "go": {
        "function_declaration": SymbolKind.FUNCTION,
        "method_declaration": SymbolKind.METHOD,
        "type_declaration": SymbolKind.CLASS,
        "import_declaration": SymbolKind.IMPORT,
    },
}


class TreeSitterProjectIndexer(ProjectIndexer):
    """Scans a project directory asynchronously and extracts symbols using tree-sitter."""

    def __init__(self):
        self._parsers: dict[str, object] = {}
        self._ts_available = False
        self._init_tree_sitter()

    def _init_tree_sitter(self):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
                import tree_sitter_languages  # noqa: F401
            self._ts_available = True
        except ImportError:
            logger.warning(
                "tree-sitter-languages not available; indexing will use filename-only mode"
            )

    def _get_parser(self, language: str):
        if not self._ts_available:
            return None
        if language not in self._parsers:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, module="tree_sitter")
                    import tree_sitter_languages
                    self._parsers[language] = tree_sitter_languages.get_parser(language)
            except Exception:
                self._parsers[language] = None
        return self._parsers[language]

    async def index(self, root: str) -> ProjectMap:
        settings = get_settings()
        root_path = Path(root).resolve()
        project_map = ProjectMap(root=str(root_path))

        # Collect files asynchronously
        files_to_index = await asyncio.get_running_loop().run_in_executor(
            None, lambda: self._collect_files(root_path, settings)
        )

        # Parse files in batches to avoid blocking
        batch_size = 50
        for i in range(0, len(files_to_index), batch_size):
            batch = files_to_index[i:i + batch_size]
            results = await asyncio.gather(
                *(self._index_file(f, root_path) for f in batch),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, FileInfo):
                    rel = result.path
                    project_map.files[rel] = result
                    project_map.total_symbols += len(result.symbols)

        project_map.total_files = len(project_map.files)
        logger.info("Indexed %d files, %d symbols in %s",
                     project_map.total_files, project_map.total_symbols, root)
        return project_map

    async def update(self, root: str, changed_files: list[str]) -> ProjectMap:
        # For now, full re-index. Incremental update can be optimized later.
        return await self.index(root)

    def _collect_files(self, root: Path, settings) -> list[Path]:
        """Walk directory tree, respecting exclusions and size limits."""
        files = []
        exclude = set(settings.project.exclude_dirs)
        extensions = set(settings.project.index_extensions)
        max_size = settings.project.max_file_size

        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded directories in-place
            dirnames[:] = [d for d in dirnames if d not in exclude and not d.startswith(".")]

            for fname in filenames:
                fpath = Path(dirpath) / fname
                if fpath.suffix in extensions:
                    try:
                        if fpath.stat().st_size <= max_size:
                            files.append(fpath)
                    except OSError:
                        continue
        return files

    async def _index_file(self, file_path: Path, root: Path) -> FileInfo:
        """Parse a single file and extract symbols."""
        loop = asyncio.get_running_loop()
        rel_path = str(file_path.relative_to(root))
        language = EXTENSION_TO_LANG.get(file_path.suffix, "")
        stat = file_path.stat()

        finfo = FileInfo(
            path=rel_path,
            language=language or file_path.suffix.lstrip("."),
            size_bytes=stat.st_size,
            last_modified=stat.st_mtime,
        )

        if language and self._ts_available:
            symbols = await loop.run_in_executor(
                None, lambda: self._extract_symbols(file_path, language)
            )
            finfo.symbols = symbols

        return finfo

    def _extract_symbols(self, file_path: Path, language: str) -> list[Symbol]:
        """Use tree-sitter to extract top-level symbols from a file."""
        parser = self._get_parser(language)
        if parser is None:
            return []

        try:
            source = file_path.read_bytes()
            tree = parser.parse(source)
        except Exception:
            return []

        query_map = SYMBOL_QUERIES.get(language, {})
        if not query_map:
            return []

        symbols = []
        self._walk_tree(tree.root_node, query_map, str(file_path), symbols, depth=0)
        return symbols

    def _walk_tree(
        self, node, query_map: dict, file_path: str, symbols: list[Symbol], depth: int,
    ) -> None:
        """Recursively walk AST nodes, collecting symbols for matching types."""
        if depth > 3:  # limit depth to top-level + 1 nesting
            return

        if node.type in query_map:
            name = self._extract_name(node)
            if name:
                kind = query_map[node.type]
                sig = self._extract_signature(node, name)
                symbols.append(Symbol(
                    name=name,
                    kind=kind,
                    file_path=file_path,
                    line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    signature=sig,
                ))

        for child in node.children:
            self._walk_tree(child, query_map, file_path, symbols, depth + 1)

    def _extract_name(self, node) -> str:
        """Extract the identifier name from a definition node."""
        for child in node.children:
            if child.type in ("identifier", "type_identifier", "property_identifier"):
                return child.text.decode("utf-8", errors="replace")
        # Handle import nodes: tree-sitter uses dotted_name / module_name
        if node.type in ("import_statement", "import_from_statement"):
            for child in node.children:
                if child.type in ("dotted_name", "module_name"):
                    return child.text.decode("utf-8", errors="replace")
            # Fallback: extract the full import text as the name
            text = node.text.decode("utf-8", errors="replace").split("\n")[0].strip()
            return text[:80] if text else ""
        # For some node types the name is the first named child
        if node.named_children:
            first = node.named_children[0]
            if first.type in ("identifier", "type_identifier"):
                return first.text.decode("utf-8", errors="replace")
        return ""

    def _extract_signature(self, node, name: str) -> str:
        """Extract a short signature (first line of the definition)."""
        text = node.text.decode("utf-8", errors="replace")
        first_line = text.split("\n")[0].strip()
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."
        return first_line
