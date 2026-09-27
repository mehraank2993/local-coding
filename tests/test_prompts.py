"""Tests for agent.prompts — prompt structure and construction helpers."""

from __future__ import annotations

import pytest

from agent.prompts import (
    REPAIR_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    _REPAIR_PLACEHOLDERS,
    build_repair_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM_PROMPT structure
# ═══════════════════════════════════════════════════════════════════════════


class TestSystemPrompt:
    """Verify the system prompt covers all required instructions."""

    def test_is_nonempty_string(self):
        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 100

    @pytest.mark.parametrize(
        "keyword",
        [
            "read_file",
            "write_patch",
            "run_shell",
            "run_tests",
        ],
    )
    def test_mentions_all_tools(self, keyword: str):
        assert keyword in SYSTEM_PROMPT

    @pytest.mark.parametrize(
        "concept",
        [
            "inspect",       # 2. inspect files before editing
            "smallest",      # 4. smallest reasonable change
            "unrelated",     # 5. avoid unrelated modifications
            "test",          # 6. run tests after editing
            "evidence",      # 7. treat failures as evidence
            "failure",       # 8. repair based on failure output
            "verified",      # 9. stop when verified
            "never",         # 10. never claim success without verification
        ],
    )
    def test_covers_key_instructions(self, concept: str):
        assert concept.lower() in SYSTEM_PROMPT.lower()

    def test_no_format_placeholders(self):
        """The system prompt is static — no {placeholders} allowed."""
        # A left brace followed by a word character would indicate a
        # placeholder.  We allow {{ (escaped braces) but not {name}.
        import re
        placeholders = re.findall(r"\{[a-zA-Z_]\w*\}", SYSTEM_PROMPT)
        assert placeholders == [], f"Unexpected placeholders: {placeholders}"


# ═══════════════════════════════════════════════════════════════════════════
# REPAIR_PROMPT_TEMPLATE structure
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairPromptTemplate:
    """Verify the repair template has the expected placeholders."""

    def test_contains_all_placeholders(self):
        for name in _REPAIR_PLACEHOLDERS:
            assert f"{{{name}}}" in REPAIR_PROMPT_TEMPLATE, (
                f"Missing placeholder {{{name}}} in template"
            )

    def test_no_extra_placeholders(self):
        """Only the declared placeholders should appear."""
        import re
        found = set(re.findall(r"\{([a-zA-Z_]\w*)\}", REPAIR_PROMPT_TEMPLATE))
        assert found == _REPAIR_PLACEHOLDERS

    def test_mentions_read_file(self):
        assert "read_file" in REPAIR_PROMPT_TEMPLATE

    def test_mentions_write_patch(self):
        assert "write_patch" in REPAIR_PROMPT_TEMPLATE

    def test_mentions_run_tests(self):
        assert "run_tests" in REPAIR_PROMPT_TEMPLATE


# ═══════════════════════════════════════════════════════════════════════════
# build_repair_prompt
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildRepairPrompt:
    """Verify the repair-prompt builder produces correct output."""

    def test_all_evidence_included(self):
        result = build_repair_prompt(
            command="python -m pytest -q",
            exit_code=1,
            stdout="FAILED test_add",
            stderr="AssertionError: 5 != 3",
        )
        assert "python -m pytest -q" in result
        assert "1" in result
        assert "FAILED test_add" in result
        assert "AssertionError: 5 != 3" in result

    def test_empty_stdout_gets_placeholder(self):
        result = build_repair_prompt(
            command="pytest",
            exit_code=2,
            stdout="",
            stderr="error",
        )
        assert "(empty)" in result

    def test_empty_stderr_gets_placeholder(self):
        result = build_repair_prompt(
            command="pytest",
            exit_code=2,
            stdout="output",
            stderr="",
        )
        assert "(empty)" in result

    def test_none_exit_code_becomes_na(self):
        result = build_repair_prompt(
            command="pytest",
            exit_code=None,
            stdout="out",
            stderr="err",
        )
        assert "N/A" in result

    def test_returns_string(self):
        result = build_repair_prompt(
            command="cmd",
            exit_code=0,
            stdout="ok",
            stderr="",
        )
        assert isinstance(result, str)
        assert len(result) > 0
