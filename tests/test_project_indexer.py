"""Tests for project indexer."""

import os
import tempfile
from pathlib import Path

import pytest

from codator.core.project_indexer import TreeSitterProjectIndexer


@pytest.mark.asyncio
async def test_index_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        indexer = TreeSitterProjectIndexer()
        pm = await indexer.index(tmpdir)
        assert pm.total_files == 0


@pytest.mark.asyncio
async def test_index_python_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a Python file with a function and a class
        py_file = Path(tmpdir) / "example.py"
        py_file.write_text(
            "def hello():\n    pass\n\nclass MyClass:\n    def method(self):\n        pass\n"
        )
        indexer = TreeSitterProjectIndexer()
        pm = await indexer.index(tmpdir)
        assert pm.total_files == 1
        assert "example.py" in pm.files
        finfo = pm.files["example.py"]
        assert finfo.language == "python"
        # If tree-sitter is available, we should have symbols
        if finfo.symbols:
            names = {s.name for s in finfo.symbols}
            assert "hello" in names
            assert "MyClass" in names


@pytest.mark.asyncio
async def test_index_respects_exclude_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files in a normal dir and an excluded dir
        normal = Path(tmpdir) / "src"
        normal.mkdir()
        (normal / "app.py").write_text("def run(): pass\n")

        excluded = Path(tmpdir) / "__pycache__"
        excluded.mkdir()
        (excluded / "cached.py").write_text("x = 1\n")

        indexer = TreeSitterProjectIndexer()
        pm = await indexer.index(tmpdir)
        paths = list(pm.files.keys())
        assert any("app.py" in p for p in paths)
        assert not any("cached.py" in p for p in paths)


@pytest.mark.asyncio
async def test_project_map_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "main.py").write_text("def main(): pass\n")
        indexer = TreeSitterProjectIndexer()
        pm = await indexer.index(tmpdir)
        summary = pm.summary()
        assert "main.py" in summary
