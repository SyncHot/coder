"""Tests for _unescape_collapsed_code — LLM double-escaped newline fix."""

from __future__ import annotations

import pytest

from codator.core.agent_loop import _unescape_collapsed_code


class TestUnescapeCollapsedCode:
    """Verify literal \\n → real newlines outside strings, preserved inside."""

    def test_no_escaped_newlines(self):
        """Text without literal \\n returns unchanged."""
        text = "import re\nprint('hello')"
        assert _unescape_collapsed_code(text) == text

    def test_basic_statement_separator(self):
        r"""Literal \n between statements → real newlines."""
        text = r"import re\nvtt = x"
        assert _unescape_collapsed_code(text) == "import re\nvtt = x"

    def test_preserves_single_quoted_string(self):
        r"""Literal \n inside single-quoted string stays as \n."""
        text = r"x = '\n'.join(lines)"
        assert _unescape_collapsed_code(text) == "x = '\\n'.join(lines)"

    def test_preserves_double_quoted_string(self):
        r"""Literal \n inside double-quoted string stays as \n."""
        text = r'x = "\n" + y'
        assert _unescape_collapsed_code(text) == 'x = "\\n" + y'

    def test_mixed_code_and_strings(self):
        r"""Mix of code separators and string-internal \n."""
        text = r"import re\nresult = x.split('\n')\nprint(result)"
        expected = "import re\nresult = x.split('\\n')\nprint(result)"
        assert _unescape_collapsed_code(text) == expected

    def test_double_quoted_mixed(self):
        r"""Double-quoted string with \n + code separator."""
        text = r'x = "\n"\nprint(x)'
        expected = 'x = "\\n"\nprint(x)'
        assert _unescape_collapsed_code(text) == expected

    def test_triple_quoted_string(self):
        r"""Literal \n inside triple-quoted string stays."""
        text = r"x = '''hello\nworld'''\nprint(x)"
        expected = "x = '''hello\\nworld'''\nprint(x)"
        assert _unescape_collapsed_code(text) == expected

    def test_triple_double_quoted(self):
        r"""Triple double-quoted string preserves \n."""
        text = r'x = """line1\nline2"""\nprint(x)'
        expected = 'x = """line1\\nline2"""\nprint(x)'
        assert _unescape_collapsed_code(text) == expected

    def test_text_already_has_real_newlines(self):
        r"""Text with both real newlines and literal \n in strings."""
        # This represents: import re<newline>result = x.split('\n')
        text = "import re\nresult = x.split('\\n')"
        assert _unescape_collapsed_code(text) == text

    def test_empty_string(self):
        assert _unescape_collapsed_code("") == ""

    def test_escaped_backslash_before_n(self):
        r"""\\n (escaped backslash then n) should not become newline."""
        text = r"path = 'C:\\new'"
        assert _unescape_collapsed_code(text) == text

    def test_tab_outside_string(self):
        r"""Literal \t outside strings → real tab."""
        text = r"if x:\n\tprint(x)"
        expected = "if x:\n\tprint(x)"
        assert _unescape_collapsed_code(text) == expected

    def test_tab_inside_string_preserved(self):
        r"""Literal \t inside strings stays as \t."""
        text = r"x = '\t'.join(lines)"
        assert _unescape_collapsed_code(text) == "x = '\\t'.join(lines)"

    def test_nested_quotes(self):
        r"""Single quotes inside double-quoted string don't confuse parser."""
        text = r'''x = "it's a '\n' test"\nprint(x)'''
        expected = "x = \"it's a '\\n' test\"\nprint(x)"
        assert _unescape_collapsed_code(text) == expected

    def test_realistic_srt_fix(self):
        r"""The actual pattern from QA testing — multiline regex fix."""
        text = (
            r"import re\n"
            r"vtt_lines = []\n"
            r"for line in srt_content.split('\n'):\n"
            r"    if re.match(r'^\d+:\d+:\d+,\d+', line):\n"
            r"        vtt_lines.append(line.replace(',', '.'))\n"
            r"    else:\n"
            r"        vtt_lines.append(line)\n"
            r"vtt = 'WEBVTT\n\n' + '\n'.join(vtt_lines)"
        )
        result = _unescape_collapsed_code(text)
        lines = result.split("\n")
        assert lines[0] == "import re"
        assert lines[1] == "vtt_lines = []"
        assert "split('\\n')" in lines[2]
        assert "'WEBVTT\\n\\n'" in result
        assert "'\\n'.join(vtt_lines)" in result

    def test_no_change_when_no_escapes(self):
        """Plain text without any backslash sequences."""
        text = "hello world"
        assert _unescape_collapsed_code(text) is text  # same object

    def test_regex_in_string(self):
        r"""Regex patterns with \d etc. preserved."""
        text = r"import re\npattern = r'\d+:\d+:\d+,\d+'"
        result = _unescape_collapsed_code(text)
        assert result.startswith("import re\n")
        assert r"\d+:\d+:\d+,\d+" in result
