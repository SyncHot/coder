"""Tests for LLM newline normalization — double/under-escaped newline fixes."""

from __future__ import annotations

import pytest

from codator.core.agent_loop import (
    PlanActVerifyAgent,
    _unescape_collapsed_code,
    _escape_broken_strings,
    normalize_edit_text,
)


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


class TestEscapeBrokenStrings:
    """Verify real newlines inside string literals → \\n."""

    def test_no_newlines(self):
        text = "x = 'hello'"
        assert _escape_broken_strings(text) == text

    def test_newline_in_single_quoted_string(self):
        """Real newline inside '...' → \\n."""
        text = "x = 'hello\nworld'"
        assert _escape_broken_strings(text) == "x = 'hello\\nworld'"

    def test_newline_in_double_quoted_string(self):
        """Real newline inside \"...\" → \\n."""
        text = 'x = "hello\nworld"'
        assert _escape_broken_strings(text) == 'x = "hello\\nworld"'

    def test_newline_outside_string_preserved(self):
        """Real newlines between statements stay as-is."""
        text = "import re\nprint('hello')"
        assert _escape_broken_strings(text) == text

    def test_strips_indentation_after_newline_in_string(self):
        """Spurious indentation after newline inside string is stripped."""
        text = "x = 'WEBVTT\n            ' + y"
        assert _escape_broken_strings(text) == "x = 'WEBVTT\\n' + y"

    def test_multiple_newlines_in_string(self):
        """Multiple real newlines → multiple \\n."""
        text = "x = 'a\n\nb'"
        assert _escape_broken_strings(text) == "x = 'a\\n\\nb'"

    def test_triple_quoted_strings_untouched(self):
        """Real newlines in triple-quoted strings are valid — don't touch."""
        text = "x = '''hello\nworld'''\nprint(x)"
        assert _escape_broken_strings(text) == text

    def test_mixed_strings_and_code(self):
        """Fix strings while preserving code newlines."""
        text = "vtt = 'WEBVTT\n\n' + '\n'.join(lines)\nreturn vtt"
        expected = "vtt = 'WEBVTT\\n\\n' + '\\n'.join(lines)\nreturn vtt"
        assert _escape_broken_strings(text) == expected

    def test_realistic_under_escaped(self):
        """The pattern from QA: LLM under-escaped \\n in string literals."""
        text = (
            "import re\n"
            "vtt = 'WEBVTT\n"
            "\n"
            "' + '\n"
            "'.join(lines)"
        )
        result = _escape_broken_strings(text)
        assert "'WEBVTT\\n\\n'" in result
        assert "'\\n'.join(lines)" in result
        assert result.startswith("import re\n")

    def test_strips_indentation_realistic(self):
        """Indentation after newline in string is stripped (real LLM output)."""
        text = "vtt = 'WEBVTT\n\n            ' + '\n            '.join(x)"
        result = _escape_broken_strings(text)
        assert "'WEBVTT\\n\\n'" in result
        assert "'\\n'.join(x)" in result


class TestNormalizeEditText:
    """End-to-end: normalize_edit_text handles both cases."""

    def test_double_escaped(self):
        r"""Collapsed code with literal \n → proper multiline."""
        text = r"import re\nx = '\n'.join(lines)"
        result = normalize_edit_text(text)
        assert result == "import re\nx = '\\n'.join(lines)"

    def test_under_escaped(self):
        """Broken strings with real newlines → proper escapes."""
        text = "import re\nx = '\n'.join(lines)"
        result = normalize_edit_text(text)
        assert result == "import re\nx = '\\n'.join(lines)"

    def test_already_correct(self):
        """Properly formatted code passes through unchanged."""
        text = "import re\nx = '\\n'.join(lines)"
        result = normalize_edit_text(text)
        assert result == text

    def test_mixed_escaping(self):
        r"""Some \n double-escaped, some under-escaped."""
        # "import re" + literal-\n + "x = '" + real-newline + "'.join(y)"
        text = "import re\\nx = '\n'.join(y)"
        result = normalize_edit_text(text)
        assert result == "import re\nx = '\\n'.join(y)"


class TestAlignIndentation:
    """Verify _align_indentation fixes LLM indentation issues."""

    _align = staticmethod(PlanActVerifyAgent._align_indentation)

    def test_no_change_when_already_correct(self):
        matched = "            vtt = old_code"
        new = "            vtt = new_code\n            print('done')"
        assert self._align(matched, new) == new

    def test_add_missing_indentation(self):
        """LLM omits all indentation — add from matched text."""
        matched = "            vtt = old_code"
        new = "import re\nvtt = new_code"
        expected = "            import re\n            vtt = new_code"
        assert self._align(matched, new) == expected

    def test_first_line_correct_subsequent_missing(self):
        """LLM indents first line but not subsequent lines — fix subsequent."""
        matched = "            vtt = old_code"
        new = "            import re\nvtt = new_code"
        expected = "            import re\n            vtt = new_code"
        assert self._align(matched, new) == expected

    def test_preserves_relative_indentation(self):
        """Relative indentation within new_text is preserved."""
        matched = "        x = old"
        new = "if condition:\n    x = new\nelse:\n    x = fallback"
        expected = (
            "        if condition:\n"
            "            x = new\n"
            "        else:\n"
            "            x = fallback"
        )
        assert self._align(matched, new) == expected

    def test_single_line_unchanged(self):
        """Single-line replacement needs no indentation fix."""
        matched = "            old_code"
        new = "new_code"
        assert self._align(matched, new) == "new_code"

    def test_empty_lines_not_indented(self):
        """Blank lines in new_text should stay blank."""
        matched = "        x = 1"
        new = "x = 1\n\ny = 2"
        expected = "        x = 1\n\n        y = 2"
        assert self._align(matched, new) == expected

    def test_partial_indent_delta(self):
        """LLM provides partial indentation — add the delta."""
        matched = "            vtt = old"  # 12 spaces
        new = "    import re\n    vtt = new"  # 4 spaces
        expected = "            import re\n            vtt = new"  # 12 spaces
        assert self._align(matched, new) == expected

    def test_no_indent_needed(self):
        """Matched text has no indentation — no changes."""
        matched = "x = old"
        new = "import re\nx = new"
        assert self._align(matched, new) == new

    def test_qa_scenario_first_correct_rest_zero(self):
        """Real QA scenario: first line has 12-space indent, rest at col 0."""
        matched = "            vtt = \"WEBVTT\\n\\n\" + srt_content.replace(',', '.')"
        new = (
            "            import re\n"
            "vtt_lines = srt_content.split('\\n')\n"
            "vtt = [re.sub(r'pattern', fix, line) for line in vtt_lines]\n"
            "vtt = \"WEBVTT\\n\\n\" + '\\n'.join(vtt)"
        )
        expected = (
            "            import re\n"
            "            vtt_lines = srt_content.split('\\n')\n"
            "            vtt = [re.sub(r'pattern', fix, line) for line in vtt_lines]\n"
            "            vtt = \"WEBVTT\\n\\n\" + '\\n'.join(vtt)"
        )
        assert self._align(matched, new) == expected

    def test_subsequent_with_relative_indent(self):
        """Subsequent lines have relative indentation that should be preserved."""
        matched = "        x = old"
        new = "        if cond:\n    x = 1\nelse:\n    x = 2"
        expected = "        if cond:\n            x = 1\n        else:\n            x = 2"
        assert self._align(matched, new) == expected
