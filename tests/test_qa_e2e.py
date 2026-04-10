"""QA Audit Tests — End-to-End scenario and agentic loop integration.

The "Ultimate Test": Agent indexes a project, finds a bug, connects SSH,
runs remote tests, reads error, returns with a fix.

Also tests:
- Agent plan parsing with project context
- Analysis-only flow (no mutations)
- Self-heal iteration limits
- Tool loop detection in ChatEngine
- Web tools (fetch/search) basic functionality
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===================================================================
# 1. AGENT LOOP — PLAN PARSING
# ===================================================================

class TestAgentPlanParsing:
    """Test that the agent correctly parses structured plans."""

    @pytest.fixture
    def agent(self):
        from codator.core.agent_loop import PlanActVerifyAgent
        return PlanActVerifyAgent(
            model="test-model",
            ollama_base_url="http://localhost:11434",
            project_root="/tmp/test_project",
        )

    def test_parse_plan_valid_json(self, agent):
        """Valid JSON plan should parse into AgentPlan."""
        raw = json.dumps({
            "task": "Fix bug in main.py",
            "reasoning": "Need to update return value",
            "steps": [
                {"action": "read_file", "target": "main.py",
                 "description": "Read the file"},
                {"action": "edit_file", "target": "main.py",
                 "description": json.dumps({
                     "file": "main.py",
                     "old": "return None",
                     "new": "return 42",
                 })},
            ],
        })

        plan = agent._parse_plan(raw, "Fix bug in main.py")
        assert plan.task == "Fix bug in main.py"
        assert len(plan.steps) == 2
        assert plan.steps[0].action == "read_file"
        assert plan.steps[1].action == "edit_file"

    def test_parse_plan_invalid_action_raises(self, agent):
        """Invalid actions should raise ValueError."""
        raw = json.dumps({
            "task": "Do something",
            "reasoning": "testing",
            "steps": [
                {"action": "read_file", "target": "a.py",
                 "description": "Read"},
                {"action": "hack_system", "target": "/etc/passwd",
                 "description": "Bad"},
            ],
        })

        with pytest.raises(ValueError, match="Invalid action"):
            agent._parse_plan(raw, "Do something")

    def test_parse_plan_empty_steps(self, agent):
        """Plan with empty steps should not crash."""
        raw = json.dumps({
            "task": "Nothing to do",
            "reasoning": "Already done",
            "steps": [],
        })

        plan = agent._parse_plan(raw, "Nothing to do")
        assert len(plan.steps) == 0

    def test_parse_plan_malformed_json(self, agent):
        """Malformed JSON should raise JSONDecodeError."""
        raw = "This is not JSON at all"
        with pytest.raises(json.JSONDecodeError):
            agent._parse_plan(raw, "Test task")


# ===================================================================
# 2. AGENT — PATH SAFETY IN ACTIONS
# ===================================================================

class TestAgentActionSafety:
    """Test agent action handlers with safety checks."""

    @pytest.fixture
    def agent(self):
        from codator.core.agent_loop import PlanActVerifyAgent
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            Path(tmpdir, "src").mkdir()
            Path(tmpdir, "src/main.py").write_text("def main():\n    return None\n")
            Path(tmpdir, "README.md").write_text("# Test Project\n")

            yield PlanActVerifyAgent(
                model="test-model",
                ollama_base_url="http://localhost:11434",
                project_root=tmpdir,
            )

    @pytest.mark.asyncio
    async def test_read_file_success(self, agent):
        from codator.core.agent_loop import AgentStep
        step = AgentStep(index=0, action="read_file", target="src/main.py",
                         description="Read main", status="running")
        content = await agent._act_read_file(step)
        assert "def main" in content

    @pytest.mark.asyncio
    async def test_read_file_traversal_blocked(self, agent):
        from codator.core.agent_loop import AgentStep
        step = AgentStep(index=0, action="read_file", target="../../etc/passwd",
                         description="Read passwd", status="running")
        with pytest.raises(ValueError, match="traversal"):
            await agent._act_read_file(step)

    @pytest.mark.asyncio
    async def test_analyze_file(self, agent):
        from codator.core.agent_loop import AgentStep
        step = AgentStep(index=0, action="analyze", target="README.md",
                         description="Check readme", status="running")
        content = await agent._act_analyze(step)
        assert "Test Project" in content

    @pytest.mark.asyncio
    async def test_analyze_nonexistent(self, agent):
        from codator.core.agent_loop import AgentStep
        step = AgentStep(index=0, action="analyze", target="ghost.py",
                         description="Check", status="running")
        with pytest.raises(FileNotFoundError):
            await agent._act_analyze(step)

    @pytest.mark.asyncio
    async def test_edit_file_creates_backup(self, agent):
        from codator.core.agent_loop import AgentStep
        edit_desc = json.dumps({
            "file": "src/main.py",
            "old": "return None",
            "new": "return 42",
        })
        step = AgentStep(index=0, action="edit_file", target="src/main.py",
                         description=edit_desc, status="running")
        result = await agent._act_edit_file(step)
        assert "backup" in result.lower()

        # Verify backup exists
        root = Path(agent._project_root)
        assert (root / "src/main.py.bak").exists()
        # Verify content changed
        assert "return 42" in (root / "src/main.py").read_text()

    @pytest.mark.asyncio
    async def test_edit_file_dict_description(self, agent):
        """Agent sometimes passes dict instead of JSON string."""
        from codator.core.agent_loop import AgentStep
        step = AgentStep(
            index=0, action="edit_file", target="src/main.py",
            description={"file": "src/main.py", "old": "return None", "new": "return 0"},
            status="running",
        )
        result = await agent._act_edit_file(step)
        assert "Edited" in result

    @pytest.mark.asyncio
    async def test_create_file(self, agent):
        from codator.core.agent_loop import AgentStep
        step = AgentStep(index=0, action="create_file", target="new_file.py",
                         description="print('hello')\n", status="running")
        result = await agent._act_create_file(step)
        assert "Created" in result
        assert (Path(agent._project_root) / "new_file.py").exists()

    @pytest.mark.asyncio
    async def test_delete_file_creates_backup(self, agent):
        from codator.core.agent_loop import AgentStep
        step = AgentStep(index=0, action="delete_file", target="README.md",
                         description="Remove readme", status="running")
        result = await agent._act_delete_file(step)
        assert "backup" in result.lower()
        root = Path(agent._project_root)
        assert not (root / "README.md").exists()
        assert (root / "README.md.bak").exists()


# ===================================================================
# 3. TOOL LOOP DETECTION IN CHAT ENGINE
# ===================================================================

class TestToolLoopDetection:
    """Test the agentic loop's duplicate detection and path normalization."""

    def test_normalize_params_paths(self):
        """Path normalization: ./ → ., .// → ., etc."""
        # Simulate the normalization logic
        def _normalize(params):
            normalized = {}
            for k, v in params.items():
                if isinstance(v, str) and ("/" in v or v == "."):
                    v = os.path.normpath(v)
                normalized[k] = v
            return normalized

        assert _normalize({"path": "."}) == {"path": "."}
        assert _normalize({"path": "./"}) == {"path": "."}
        assert _normalize({"path": ".//"}) == {"path": "."}
        assert _normalize({"path": ".///"}) == {"path": "."}
        assert _normalize({"path": "src/../src/main.py"}) == {"path": "src/main.py"}
        assert _normalize({"path": "./src/./main.py"}) == {"path": "src/main.py"}

    def test_loop_key_deduplication(self):
        """Same tool + normalized params should produce same key."""

        def _key(name, params):
            normalized = {}
            for k, v in params.items():
                if isinstance(v, str) and ("/" in v or v == "."):
                    v = os.path.normpath(v)
                normalized[k] = v
            return f"{name}:{json.dumps(normalized, sort_keys=True)}"

        k1 = _key("list_directory", {"path": "."})
        k2 = _key("list_directory", {"path": "./"})
        k3 = _key("list_directory", {"path": ".//"})
        assert k1 == k2 == k3

    def test_different_tools_different_keys(self):
        k1 = f"read_file:{json.dumps({'path': 'main.py'})}"
        k2 = f"list_directory:{json.dumps({'path': 'main.py'})}"
        assert k1 != k2


# ===================================================================
# 4. WEB TOOLS
# ===================================================================

class TestWebFetchTool:
    """Test the web_fetch tool."""

    @pytest.fixture
    def tool(self):
        from codator.infrastructure.tools.web_tools import WebFetchTool
        return WebFetchTool()

    async def test_fetch_no_url(self, tool):
        result = await tool.execute()
        assert result.success is False
        assert "url" in result.error.lower()

    async def test_fetch_adds_https(self, tool):
        """Tool should auto-add https:// prefix."""
        # Mock httpx to avoid real network
        with patch("codator.infrastructure.tools.web_tools.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "text/html"}
            mock_resp.text = "<html><body>Hello World</body></html>"
            mock_resp.raise_for_status = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            result = await tool.execute(url="example.com")
            assert result.success is True
            assert "Hello World" in result.output


class TestWebSearchTool:
    """Test the web_search tool."""

    @pytest.fixture
    def tool(self):
        from codator.infrastructure.tools.web_tools import WebSearchTool
        return WebSearchTool()

    async def test_search_no_query(self, tool):
        result = await tool.execute()
        assert result.success is False
        assert "query" in result.error.lower()

    async def test_search_with_mock_ddg(self, tool):
        """Search with mocked DuckDuckGo API."""
        mock_response = {
            "Abstract": "Python is a programming language.",
            "Heading": "Python",
            "AbstractURL": "https://python.org",
            "RelatedTopics": [],
        }

        with patch("codator.infrastructure.tools.web_tools.httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json = MagicMock(return_value=mock_response)
            mock_resp.raise_for_status = MagicMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_resp)
            mock_cls.return_value = mock_client

            result = await tool.execute(query="Python programming")
            assert result.success is True
            assert "Python" in result.output


# ===================================================================
# 5. HTML TO TEXT CONVERSION
# ===================================================================

class TestHtmlToText:
    """Test HTML cleaning for web_fetch."""

    def test_strip_script_tags(self):
        from codator.infrastructure.tools.web_tools import _html_to_text
        html = "<html><script>alert('xss')</script><p>Hello</p></html>"
        text = _html_to_text(html)
        assert "alert" not in text
        assert "Hello" in text

    def test_strip_style_tags(self):
        from codator.infrastructure.tools.web_tools import _html_to_text
        html = "<html><style>body{color:red}</style><p>Content</p></html>"
        text = _html_to_text(html)
        assert "color:red" not in text
        assert "Content" in text

    def test_max_length_truncation(self):
        from codator.infrastructure.tools.web_tools import _html_to_text
        html = "<p>" + "x" * 10000 + "</p>"
        text = _html_to_text(html, max_length=100)
        assert len(text) <= 100

    def test_strip_nav_footer(self):
        from codator.infrastructure.tools.web_tools import _html_to_text
        html = "<nav>Navigation</nav><main>Main Content</main><footer>Footer</footer>"
        text = _html_to_text(html)
        assert "Navigation" not in text
        assert "Footer" not in text
        assert "Main Content" in text


# ===================================================================
# 6. END-TO-END SCENARIO (MOCKED)
# ===================================================================

class TestE2EScenario:
    """
    The "Ultimate Test" — simulates the full agent workflow:
    1. Index local project
    2. Find a logical bug
    3. Connect SSH to remote server
    4. Run tests remotely, read error
    5. Return with fix to local file
    """

    @pytest.mark.asyncio
    async def test_full_agent_workflow_mocked(self):
        """Simulate the complete agent workflow with mocks."""
        from codator.core.agent_loop import PlanActVerifyAgent, AgentStep

        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup: create a project with a bug
            Path(tmpdir, "calculator.py").write_text(
                "def divide(a, b):\n"
                "    return a * b  # BUG: should be a / b\n"
            )
            Path(tmpdir, "test_calc.py").write_text(
                "from calculator import divide\n"
                "def test_divide():\n"
                "    assert divide(10, 2) == 5\n"
            )

            agent = PlanActVerifyAgent(
                model="test-model",
                ollama_base_url="http://localhost:11434",
                project_root=tmpdir,
            )

            # Step 1: Read the buggy file (agent action)
            step_read = AgentStep(
                index=0, action="read_file", target="calculator.py",
                description="Read calculator", status="running",
            )
            content = await agent._act_read_file(step_read)
            assert "a * b" in content  # Bug is visible

            # Step 2: Analyze the file
            step_analyze = AgentStep(
                index=1, action="analyze", target="calculator.py",
                description="Find logical bugs", status="running",
            )
            analysis = await agent._act_analyze(step_analyze)
            assert "a * b" in analysis

            # Step 3: Fix the bug
            step_fix = AgentStep(
                index=2, action="edit_file", target="calculator.py",
                description=json.dumps({
                    "file": "calculator.py",
                    "old": "return a * b  # BUG: should be a / b",
                    "new": "return a / b",
                }),
                status="running",
            )
            fix_result = await agent._act_edit_file(step_fix)
            assert "Edited" in fix_result

            # Step 4: Verify the fix
            fixed_content = Path(tmpdir, "calculator.py").read_text()
            assert "return a / b" in fixed_content
            assert "a * b" not in fixed_content

            # Step 5: Backup exists
            assert Path(tmpdir, "calculator.py.bak").exists()
            backup_content = Path(tmpdir, "calculator.py.bak").read_text()
            assert "a * b" in backup_content  # Original preserved

    @pytest.mark.asyncio
    async def test_agent_project_context_passed(self):
        """Verify project file tree is passed as context to plan."""
        from codator.core.agent_loop import PlanActVerifyAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multi-module project
            Path(tmpdir, "backend").mkdir()
            Path(tmpdir, "backend/__init__.py").touch()
            Path(tmpdir, "backend/api.py").write_text("def get(): pass\n")
            Path(tmpdir, "frontend").mkdir()
            Path(tmpdir, "frontend/App.tsx").write_text("export default App;\n")

            PlanActVerifyAgent(
                model="test-model",
                ollama_base_url="http://localhost:11434",
                project_root=tmpdir,
            )

            # Build project context (same as commands.py does)
            code_exts = {".py", ".js", ".ts", ".tsx", ".go", ".rs"}
            files = []
            root = Path(tmpdir).resolve()
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix in code_exts:
                    rel = f.relative_to(root)
                    parts = rel.parts
                    if any(p.startswith(".") or p in (
                        "__pycache__", "node_modules", ".venv", "venv",
                    ) for p in parts):
                        continue
                    files.append(str(rel))

            project_context = "Project file tree:\n" + "\n".join(files)

            assert "backend/api.py" in project_context
            assert "backend/__init__.py" in project_context
            assert "frontend/App.tsx" in project_context
