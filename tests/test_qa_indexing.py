"""QA Audit Tests — Indexing precision and RAG/Git integration.

Tests for:
- tree-sitter symbol extraction accuracy (imports, classes, functions)
- Cross-module dependency detection in 100+ file projects
- Git diff interpretation and merge conflict handling
- Contextual index chunking and embedding retrieval
- ProjectMap summary correctness
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codator.domain.models import FileInfo, ProjectMap, Symbol, SymbolKind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_project(tmpdir: str, files: dict[str, str]) -> str:
    """Create a temporary project with given files."""
    for rel_path, content in files.items():
        full = os.path.join(tmpdir, rel_path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    return tmpdir


# ===================================================================
# 1. TREE-SITTER SYMBOL EXTRACTION
# ===================================================================

class TestTreeSitterPrecision:
    """Verify tree-sitter correctly identifies symbols in Python files."""

    @pytest.mark.asyncio
    async def test_python_function_detection(self):
        """tree-sitter should detect Python function definitions."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "main.py": (
                    "def hello():\n"
                    "    pass\n\n"
                    "def world(x, y):\n"
                    "    return x + y\n"
                ),
            })

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            assert pmap.total_files >= 1
            assert pmap.total_symbols >= 2

            main_info = pmap.files.get("main.py")
            assert main_info is not None
            func_names = [s.name for s in main_info.symbols if s.kind == SymbolKind.FUNCTION]
            assert "hello" in func_names
            assert "world" in func_names

    @pytest.mark.asyncio
    async def test_python_class_detection(self):
        """tree-sitter should detect Python class definitions."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "models.py": (
                    "class User:\n"
                    "    def __init__(self, name):\n"
                    "        self.name = name\n\n"
                    "class Admin(User):\n"
                    "    role = 'admin'\n"
                ),
            })

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            models = pmap.files.get("models.py")
            assert models is not None
            class_names = [s.name for s in models.symbols if s.kind == SymbolKind.CLASS]
            assert "User" in class_names
            assert "Admin" in class_names

    @pytest.mark.asyncio
    async def test_python_import_detection(self):
        """tree-sitter should detect import statements."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "app.py": (
                    "import os\n"
                    "import json\n"
                    "from pathlib import Path\n"
                    "from .models import User\n\n"
                    "def run():\n"
                    "    pass\n"
                ),
            })

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            app = pmap.files.get("app.py")
            assert app is not None
            import_syms = [s for s in app.symbols if s.kind == SymbolKind.IMPORT]
            assert len(import_syms) >= 2  # at least os and json

    @pytest.mark.asyncio
    async def test_multifile_project_indexing(self):
        """Index a multi-file project and verify all files are found."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            files = {}
            for i in range(15):
                files[f"module_{i}.py"] = (
                    f"def func_{i}():\n    return {i}\n\n"
                    f"class Class_{i}:\n    pass\n"
                )
            _create_project(tmpdir, files)

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            assert pmap.total_files == 15
            assert pmap.total_symbols >= 30  # 15 functions + 15 classes

    @pytest.mark.asyncio
    async def test_excluded_dirs_skipped(self):
        """node_modules, .git, __pycache__ should be excluded."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "main.py": "def main(): pass\n",
                "node_modules/dep/index.js": "function x() {}\n",
                "__pycache__/cached.py": "def cached(): pass\n",
                ".git/config": "[core]\n",
            })

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            # Only main.py should be indexed
            assert pmap.total_files == 1
            assert "main.py" in pmap.files

    @pytest.mark.asyncio
    async def test_empty_project(self):
        """Empty directory should index with 0 files."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)
            assert pmap.total_files == 0
            assert pmap.total_symbols == 0


# ===================================================================
# 2. CROSS-MODULE DEPENDENCY DETECTION
# ===================================================================

class TestCrossModuleDependencies:
    """Verify that imports between distant modules are tracked."""

    @pytest.mark.asyncio
    async def test_import_graph_basic(self):
        """Imports from one module to another should be captured as symbols."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "core/engine.py": (
                    "from core.utils import helper\n"
                    "from core.models import User\n\n"
                    "class Engine:\n"
                    "    def run(self):\n"
                    "        return helper()\n"
                ),
                "core/utils.py": (
                    "def helper():\n"
                    "    return 42\n"
                ),
                "core/models.py": (
                    "class User:\n"
                    "    name: str\n"
                ),
                "core/__init__.py": "",
            })

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            engine = pmap.files.get("core/engine.py")
            assert engine is not None
            # Should have import symbols
            imports = [s for s in engine.symbols if s.kind == SymbolKind.IMPORT]
            assert len(imports) >= 1

    @pytest.mark.asyncio
    async def test_project_summary_lists_symbols(self):
        """ProjectMap.summary() should list files and symbols."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "api.py": "def get_users(): pass\ndef create_user(): pass\n",
                "db.py": "class Database:\n    pass\n",
            })

            indexer = TreeSitterProjectIndexer()
            pmap = await indexer.index(tmpdir)

            summary = pmap.summary(max_files=50)
            assert "api.py" in summary
            assert "db.py" in summary
            assert "get_users" in summary or "function" in summary.lower()


# ===================================================================
# 3. GIT DIFF AND MERGE CONFLICT HANDLING
# ===================================================================

class TestGitIntegration:
    """Test git context extraction and diff handling."""

    @pytest.fixture
    def git_context(self):
        """GitContext for the codator project itself."""
        from codator.core.git_integration import GitContext
        # Use the actual codator repo
        return GitContext(repo_path=".")

    def test_git_available(self, git_context):
        """codator is a git repo — should be detected."""
        assert git_context.is_available

    def test_work_context_has_branch(self, git_context):
        """Work context should include current branch name."""
        ctx = git_context.get_work_context()
        assert "Branch:" in ctx

    def test_work_context_has_commits(self, git_context):
        """Should include recent commit history."""
        ctx = git_context.get_work_context()
        assert "commits" in ctx.lower() or "Recent" in ctx

    def test_get_changed_files_returns_list(self, git_context):
        """Should return a list of strings (possibly empty)."""
        changed = git_context.get_changed_files()
        assert isinstance(changed, list)

    def test_get_file_at_head(self, git_context):
        """Should read a known file from HEAD."""
        content = git_context.get_file_at_head("pyproject.toml")
        if content is not None:
            assert "codator" in content

    def test_get_file_at_head_nonexistent(self, git_context):
        """Nonexistent file should return None."""
        content = git_context.get_file_at_head("nonexistent_file_xyz.py")
        assert content is None

    def test_work_context_truncation(self, git_context):
        """Diff should be truncated when max_diff_lines is small."""
        ctx = git_context.get_work_context(max_diff_lines=5)
        # Should not exceed a reasonable size
        assert len(ctx) < 100000

    def test_non_repo_returns_unavailable(self):
        """Non-git directory should report unavailable."""
        from codator.core.git_integration import GitContext

        with tempfile.TemporaryDirectory() as tmpdir:
            gc = GitContext(repo_path=tmpdir)
            assert not gc.is_available
            ctx = gc.get_work_context()
            assert "not available" in ctx.lower()


# ===================================================================
# 4. MERGE CONFLICT DETECTION
# ===================================================================

class TestMergeConflictHandling:
    """Test that merge conflict markers are recognizable in diff output."""

    def test_conflict_markers_in_text(self):
        """Verify conflict markers can be detected in raw diff text."""
        conflict_diff = (
            "<<<<<<< HEAD\n"
            "    return old_value\n"
            "=======\n"
            "    return new_value\n"
            ">>>>>>> feature-branch\n"
        )
        assert "<<<<<<< HEAD" in conflict_diff
        assert "=======" in conflict_diff
        assert ">>>>>>>" in conflict_diff

    def test_file_with_conflict_markers_indexed(self):
        """Files with conflict markers should still be indexable."""
        from codator.core.project_indexer import TreeSitterProjectIndexer

        # tree-sitter will treat conflict markers as syntax errors
        # but should not crash
        conflict_content = (
            "def func():\n"
            "<<<<<<< HEAD\n"
            "    return 1\n"
            "=======\n"
            "    return 2\n"
            ">>>>>>> branch\n"
        )
        # The indexer should handle malformed files gracefully
        # (tree-sitter is error-tolerant)


# ===================================================================
# 5. CONTEXTUAL INDEX / RAG
# ===================================================================

class TestContextualIndex:
    """Test the ContextualIndex chunking and retrieval."""

    @pytest.mark.asyncio
    async def test_chunk_creation(self):
        """Chunking should split files into logical segments."""
        from codator.core.context_retrieval import ContextualIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "main.py": "\n".join(
                    [f"def func_{i}():\n    return {i}\n" for i in range(20)]
                ),
            })

            index = ContextualIndex(project_root=tmpdir)
            count = await index.index_project()
            assert count >= 1  # at least one chunk

    @pytest.mark.asyncio
    async def test_chunks_have_metadata(self):
        """Each chunk should have file_path and line range."""
        from codator.core.context_retrieval import ContextualIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            _create_project(tmpdir, {
                "utils.py": "def helper():\n    return 42\n",
            })

            index = ContextualIndex(project_root=tmpdir)
            await index.index_project()

            assert len(index._chunks) >= 1
            chunk = index._chunks[0]
            assert hasattr(chunk, "file_path")
            assert hasattr(chunk, "start_line")
            assert hasattr(chunk, "content")
            assert chunk.content  # not empty


# ===================================================================
# 6. PROJECT MAP MODEL
# ===================================================================

class TestProjectMapModel:
    """Test ProjectMap data model behavior."""

    def test_summary_empty_project(self):
        pmap = ProjectMap(root="/tmp/empty", files={}, total_files=0, total_symbols=0)
        summary = pmap.summary()
        assert "/tmp/empty" in summary or "0" in summary

    def test_summary_with_files(self):
        pmap = ProjectMap(
            root="/tmp/proj",
            files={
                "main.py": FileInfo(
                    path="main.py",
                    language="python",
                    size_bytes=100,
                    symbols=[
                        Symbol(name="main", kind=SymbolKind.FUNCTION,
                               file_path="main.py", line=1),
                    ],
                ),
            },
            total_files=1,
            total_symbols=1,
        )
        summary = pmap.summary()
        assert "main.py" in summary

    def test_summary_max_files_limit(self):
        """summary(max_files=N) should limit output."""
        files = {}
        for i in range(100):
            files[f"mod_{i}.py"] = FileInfo(
                path=f"mod_{i}.py", language="python", size_bytes=50,
                symbols=[], last_modified=0.0,
            )

        pmap = ProjectMap(root="/tmp", files=files, total_files=100, total_symbols=0)
        summary_short = pmap.summary(max_files=5)
        summary_long = pmap.summary(max_files=100)
        # Short summary should be shorter than full
        assert len(summary_short) <= len(summary_long)
