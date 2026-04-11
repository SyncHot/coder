"""Semantic embedding search — RAG over codebase using local embeddings."""

import asyncio
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolResult:
    success: bool = True
    output: str = ""
    error: str = ""
    exit_code: int = 0
    artifacts: dict = field(default_factory=dict)


IGNORE_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", ".mypy_cache", "dist", "build", ".tox", "egg-info"}
CODE_EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt"}


class EmbeddingSearchTool:
    name = "embedding_search"
    description = (
        "Semantic search over codebase using local embeddings. Index project files, "
        "then query by meaning (not just keywords). Uses sentence-transformers for embeddings "
        "and stores vectors in SQLite for fast retrieval. Ideal for finding related code, "
        "understanding concepts, and discovering similar implementations."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["index", "search", "similar", "status"],
                "description": "Action: index project, search by meaning, find similar code, or check index status",
            },
            "path": {
                "type": "string",
                "description": "Project directory to index or search in",
            },
            "query": {
                "type": "string",
                "description": "Natural language query for semantic search",
            },
            "file_path": {
                "type": "string",
                "description": "File path for 'similar' action — find code similar to this file",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 10,
            },
            "file_filter": {
                "type": "string",
                "description": "Filter results to files matching this pattern",
            },
            "chunk_size": {
                "type": "integer",
                "description": "Number of lines per chunk when indexing",
                "default": 30,
            },
        },
        "required": ["action", "path"],
    }

    def __init__(self):
        self._model = None
        self._db_dir = os.path.expanduser("~/.codator/embeddings")
        os.makedirs(self._db_dir, exist_ok=True)

    def _get_db_path(self, project_path: str) -> str:
        key = hashlib.md5(os.path.abspath(project_path).encode()).hexdigest()
        return os.path.join(self._db_dir, f"index_{key}.db")

    def _init_db(self, db_path: str):
        conn = sqlite3.connect(db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                file_hash TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_file ON chunks(file_path)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        conn.commit()
        return conn

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs["action"]
        path = kwargs["path"]

        try:
            if action == "index":
                return await self._index_project(path, kwargs)
            elif action == "search":
                query = kwargs.get("query", "")
                if not query:
                    return ToolResult(success=False, error="query is required for search action")
                return await self._search(path, query, kwargs)
            elif action == "similar":
                file_path = kwargs.get("file_path", "")
                if not file_path:
                    return ToolResult(success=False, error="file_path is required for similar action")
                return await self._find_similar(path, file_path, kwargs)
            elif action == "status":
                return await self._get_status(path)
            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except ImportError as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=f"{type(e).__name__}: {str(e)}")

    async def _index_project(self, path: str, kwargs: dict) -> ToolResult:
        """Index all code files in the project."""
        chunk_size = kwargs.get("chunk_size", 30)

        if not os.path.isdir(path):
            return ToolResult(success=False, error=f"Directory not found: {path}")

        model = self._load_model()
        db_path = self._get_db_path(path)
        conn = self._init_db(db_path)

        # Clear existing index
        conn.execute("DELETE FROM chunks")
        conn.commit()

        chunks = []
        file_count = 0

        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for fname in files:
                _, ext = os.path.splitext(fname)
                if ext not in CODE_EXTENSIONS:
                    continue

                filepath = os.path.join(root, fname)
                rel_path = os.path.relpath(filepath, path)
                file_count += 1

                try:
                    with open(filepath, "r", errors="ignore") as f:
                        lines = f.readlines()
                except (IOError, OSError):
                    continue

                file_hash = hashlib.md5("".join(lines).encode()).hexdigest()

                # Chunk the file
                for i in range(0, len(lines), chunk_size):
                    chunk_lines = lines[i:i + chunk_size]
                    content = "".join(chunk_lines).strip()
                    if len(content) < 20:  # Skip tiny chunks
                        continue
                    # Prepend file context
                    context = f"File: {rel_path}\n{content}"
                    chunks.append((rel_path, i + 1, i + len(chunk_lines), content, context, file_hash))

        if not chunks:
            return ToolResult(success=False, error="No code files found to index")

        # Generate embeddings in batches
        batch_size = 64
        total_indexed = 0
        contexts = [c[4] for c in chunks]

        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]
            embeddings = model.encode(batch, show_progress_bar=False)

            for j, emb in enumerate(embeddings):
                idx = i + j
                rel_path, line_start, line_end, content, _, file_hash = chunks[idx]
                conn.execute(
                    "INSERT INTO chunks (file_path, line_start, line_end, content, embedding, file_hash) VALUES (?,?,?,?,?,?)",
                    (rel_path, line_start, line_end, content, emb.tobytes(), file_hash),
                )
                total_indexed += 1

        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("indexed_at", str(asyncio.get_event_loop().time())),
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("file_count", str(file_count)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("chunk_count", str(total_indexed)),
        )
        conn.commit()
        conn.close()

        return ToolResult(
            success=True,
            output=f"Indexed {file_count} files → {total_indexed} chunks (chunk_size={chunk_size})",
            artifacts={"files": file_count, "chunks": total_indexed, "db_path": db_path},
        )

    async def _search(self, path: str, query: str, kwargs: dict) -> ToolResult:
        """Semantic search over indexed codebase."""
        import numpy as np

        top_k = kwargs.get("top_k", 10)
        file_filter = kwargs.get("file_filter", "")

        model = self._load_model()
        db_path = self._get_db_path(path)

        if not os.path.isfile(db_path):
            return ToolResult(success=False, error="Project not indexed. Run with action='index' first.")

        conn = sqlite3.connect(db_path)
        query_embedding = model.encode([query])[0]

        # Load all embeddings
        if file_filter:
            rows = conn.execute(
                "SELECT id, file_path, line_start, line_end, content, embedding FROM chunks WHERE file_path LIKE ?",
                (f"%{file_filter}%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, file_path, line_start, line_end, content, embedding FROM chunks"
            ).fetchall()

        if not rows:
            return ToolResult(success=False, error="No indexed chunks found")

        # Compute cosine similarity
        similarities = []
        for row in rows:
            chunk_id, file_path, line_start, line_end, content, emb_bytes = row
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
            similarities.append((sim, file_path, line_start, line_end, content))

        similarities.sort(reverse=True)
        top_results = similarities[:top_k]

        lines = [f"Query: \"{query}\"\nTop {len(top_results)} results:\n"]
        for i, (score, fpath, ls, le, content) in enumerate(top_results, 1):
            preview = content[:200].replace("\n", " ↵ ")
            lines.append(f"{i}. [{score:.3f}] {fpath}:{ls}-{le}")
            lines.append(f"   {preview}\n")

        conn.close()
        return ToolResult(
            success=True,
            output="\n".join(lines)[:5000],
            artifacts={"results": len(top_results), "top_score": float(top_results[0][0]) if top_results else 0},
        )

    async def _find_similar(self, path: str, file_path: str, kwargs: dict) -> ToolResult:
        """Find code chunks similar to a given file."""
        import numpy as np

        top_k = kwargs.get("top_k", 10)
        model = self._load_model()
        db_path = self._get_db_path(path)

        if not os.path.isfile(db_path):
            return ToolResult(success=False, error="Project not indexed. Run with action='index' first.")

        # Read target file
        full_path = os.path.join(path, file_path) if not os.path.isabs(file_path) else file_path
        if not os.path.isfile(full_path):
            return ToolResult(success=False, error=f"File not found: {full_path}")

        with open(full_path, "r", errors="ignore") as f:
            content = f.read()

        file_embedding = model.encode([content[:2000]])[0]

        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT file_path, line_start, line_end, content, embedding FROM chunks WHERE file_path != ?",
            (file_path,),
        ).fetchall()

        similarities = []
        for fpath, ls, le, chunk_content, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = np.dot(file_embedding, emb) / (np.linalg.norm(file_embedding) * np.linalg.norm(emb) + 1e-8)
            similarities.append((sim, fpath, ls, le, chunk_content))

        similarities.sort(reverse=True)
        top_results = similarities[:top_k]

        lines = [f"Similar to: {file_path}\nTop {len(top_results)} results:\n"]
        for i, (score, fpath, ls, le, chunk_content) in enumerate(top_results, 1):
            preview = chunk_content[:150].replace("\n", " ↵ ")
            lines.append(f"{i}. [{score:.3f}] {fpath}:{ls}-{le}")
            lines.append(f"   {preview}\n")

        conn.close()
        return ToolResult(success=True, output="\n".join(lines)[:5000])

    async def _get_status(self, path: str) -> ToolResult:
        """Check index status for a project."""
        db_path = self._get_db_path(path)

        if not os.path.isfile(db_path):
            return ToolResult(success=True, output="Not indexed. Run with action='index' to create index.")

        conn = sqlite3.connect(db_path)
        meta = dict(conn.execute("SELECT key, value FROM metadata").fetchall())
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        file_count = len(set(r[0] for r in conn.execute("SELECT DISTINCT file_path FROM chunks").fetchall()))
        conn.close()

        return ToolResult(
            success=True,
            output=f"Index status:\n  Files: {file_count}\n  Chunks: {chunk_count}\n  DB: {db_path}",
            artifacts={"files": file_count, "chunks": chunk_count, "db_path": db_path},
        )
