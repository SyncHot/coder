"""Tests for ContextualIndex — TF-IDF search and chunk management."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from codator.core.context_retrieval import ContextualIndex, CodeChunk


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Create a minimal project for indexing."""
    (tmp_path / "main.py").write_text(
        'def hello():\n    """Say hello."""\n    print("Hello world")\n\n'
        'def compute(x, y):\n    """Multiply two numbers."""\n    return x * y\n'
    )
    (tmp_path / "utils.py").write_text(
        'import os\nimport sys\n\n'
        'def read_file(path):\n    """Read a file."""\n'
        '    with open(path) as f:\n        return f.read()\n\n'
        'def write_file(path, content):\n    """Write to a file."""\n'
        '    with open(path, "w") as f:\n        f.write(content)\n'
    )
    return tmp_path


class TestContextualIndex:
    @pytest.mark.asyncio
    async def test_index_project(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        count = await idx.index_project()
        assert count > 0

    @pytest.mark.asyncio
    async def test_search_returns_results(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        await idx.index_project()
        results = idx.search("hello world", top_k=3)
        assert len(results) > 0
        assert isinstance(results[0], CodeChunk)

    @pytest.mark.asyncio
    async def test_search_relevance(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        await idx.index_project()
        results = idx.search("read file", top_k=3)
        # Should find utils.py content
        found_utils = any("utils" in r.file_path for r in results)
        assert found_utils

    @pytest.mark.asyncio
    async def test_empty_query(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        await idx.index_project()
        results = idx.search("", top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_format_chunks(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        await idx.index_project()
        results = idx.search("hello", top_k=2)
        formatted = idx.format_chunks_for_prompt(results)
        assert "Relevant Code" in formatted

    @pytest.mark.asyncio
    async def test_empty_project(self, tmp_path: Path):
        idx = ContextualIndex(str(tmp_path))
        count = await idx.index_project()
        assert count == 0
        assert idx.search("anything") == []

    def test_has_embeddings_false_by_default(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        assert idx.has_embeddings is False

    @pytest.mark.asyncio
    async def test_rrf_fuse(self, project_dir: Path):
        idx = ContextualIndex(str(project_dir))
        await idx.index_project()
        chunks = idx.search("hello", top_k=5)
        if len(chunks) >= 2:
            fused = ContextualIndex._rrf_fuse(
                chunks[:2], list(reversed(chunks[:2])), top_k=2,
            )
            assert len(fused) == 2
