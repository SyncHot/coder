"""Tests for safe_parse_json — LLM JSON resilience."""

from __future__ import annotations

import json

import pytest

from codator.core.agent_loop import safe_parse_json


class TestSafeParseJson:
    """Verify safe_parse_json handles various LLM JSON malformations."""

    def test_valid_json(self):
        """Standard valid JSON passes through."""
        data = safe_parse_json('{"file": "foo.py", "old": "x", "new": "y"}')
        assert data["file"] == "foo.py"
        assert data["old"] == "x"
        assert data["new"] == "y"

    def test_markdown_fences(self):
        """JSON wrapped in markdown code fences."""
        raw = '```json\n{"file": "f.py", "old": "a", "new": "b"}\n```'
        data = safe_parse_json(raw)
        assert data["file"] == "f.py"

    def test_invalid_backslash_escapes(self):
        r"""LLM uses \d, \w etc. that aren't valid JSON escapes."""
        raw = r'{"pattern": "\\d+", "file": "test.py"}'
        data = safe_parse_json(raw)
        assert "file" in data

    def test_unescaped_quotes_in_code(self):
        """LLM forgets to escape some double quotes inside code values.

        This is the critical case: replace(",", ".") where the second
        argument's quotes are not escaped.
        """
        raw = (
            '{"file":"video.py",'
            '"old":"vtt = \\"WEBVTT\\n\\n\\" + srt_content.replace(\\",\\", \\".\\")",'
            '"new":"vtt = \\"WEBVTT\\n\\n\\" + fixed"}'
        )
        # This is valid JSON — sanity check
        data = safe_parse_json(raw)
        assert 'replace(",", ".")' in data["old"]

    def test_unescaped_quotes_partial(self):
        """LLM escapes some quotes but not others — the real-world failure.

        Pattern: replace(\\",\\", ".") — first arg escaped, second not.
        """
        # Construct the exact malformed JSON from QA testing
        raw = (
            '{"file":"video.py",'
            '"old":"vtt = \\"WEBVTT\\n\\n\\" + srt_content.replace(\\",\\", ".")",'
            '"new":"fixed_code"}'
        )
        # Standard json.loads would fail on this
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

        # safe_parse_json should handle it
        data = safe_parse_json(raw)
        assert data["file"] == "video.py"
        assert "replace" in data["old"]
        assert data["new"] == "fixed_code"

    def test_unescaped_quotes_both_args(self):
        """Neither argument's quotes are escaped."""
        raw = (
            '{"file":"test.py",'
            '"old":"x = func("hello", "world")",'
            '"new":"x = func("hi", "earth")"}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

        data = safe_parse_json(raw)
        assert data["file"] == "test.py"
        assert "hello" in data["old"]

    def test_unescaped_quotes_complex_code(self):
        """Multiple unescaped quotes in a realistic code snippet."""
        raw = (
            '{"file":"app.py",'
            '"old":"if x == "yes" and y == "no":\\n    return "maybe"",'
            '"new":"if x == "yes":\\n    return "definitely""}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

        data = safe_parse_json(raw)
        assert data["file"] == "app.py"
        assert "yes" in data["old"]

    def test_unescaped_quotes_with_newlines(self):
        """Unescaped quotes combined with newline escapes."""
        raw = (
            '{"file":"v.py",'
            '"old":"x = "WEBVTT\\n\\n" + y.replace(",", ".")",'
            '"new":"x = "WEBVTT\\n\\n" + fixed"}'
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(raw)

        data = safe_parse_json(raw)
        assert data["file"] == "v.py"
        assert "WEBVTT" in data["old"]

    def test_trailing_comma(self):
        """Trailing comma after last field (common LLM error)."""
        raw = '{"file": "test.py", "old": "x", "new": "y",}'
        # dirtyjson handles this
        data = safe_parse_json(raw)
        assert data["file"] == "test.py"

    def test_single_quoted_keys(self):
        """LLM uses single quotes for JSON keys."""
        raw = "{'file': 'test.py', 'old': 'x', 'new': 'y'}"
        # dirtyjson handles this
        data = safe_parse_json(raw)
        assert data["file"] == "test.py"

    def test_raises_on_garbage(self):
        """Truly unparseable input raises JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            safe_parse_json("this is not json at all")

    def test_nested_objects(self):
        """JSON with nested objects parses correctly."""
        raw = '{"file": "test.py", "meta": {"line": 42}, "old": "x", "new": "y"}'
        data = safe_parse_json(raw)
        assert data["meta"]["line"] == 42

    def test_real_world_qwen_output(self):
        """Reproduce the exact failure from qwen2.5-coder:14b QA testing.

        The LLM generates a one-liner replacement with mixed quoting and
        regex patterns containing unescaped quotes.
        """
        # Simplified version of the actual failing JSON
        raw = (
            '{"file":"backend/blueprints/video_station.py",'
            '"old":"vtt = \\"WEBVTT\\n\\n\\" + srt_content.replace(\\",\\", ".")",'
            '"new":"vtt = \\"WEBVTT\\n\\n\\" + fixed_content"}'
        )
        data = safe_parse_json(raw)
        assert data["file"] == "backend/blueprints/video_station.py"
        assert "srt_content" in data["old"]
        assert data["new"].endswith("fixed_content")
