"""Tests for main.py — CLI argument parsing, validation, and integration.

These tests exercise the CLI layer without requiring a live Ollama server.
Where the agent loop is needed, it is mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import build_parser, main, _print_summary
from agent.loop import AgentState
from agent.types import ToolResult


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildParser:
    """Tests for the argparse parser construction."""

    def test_task_required(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_task_positional(self):
        parser = build_parser()
        args = parser.parse_args(["Fix the bug"])
        assert args.task == "Fix the bug"

    def test_defaults(self):
        parser = build_parser()
        args = parser.parse_args(["some task"])
        assert args.model is None
        assert args.repair is True
        assert args.max_steps == 20
        assert args.max_repairs == 3
        assert args.repo == "."

    def test_all_flags(self):
        parser = build_parser()
        args = parser.parse_args([
            "--model", "codellama:13b",
            "--no-repair",
            "--max-steps", "50",
            "--max-repairs", "5",
            "--repo", "/tmp/project",
            "Do something",
        ])
        assert args.model == "codellama:13b"
        assert args.repair is False
        assert args.max_steps == 50
        assert args.max_repairs == 5
        assert args.repo == "/tmp/project"
        assert args.task == "Do something"

    def test_repair_flag_enabled(self):
        parser = build_parser()
        args = parser.parse_args(["--repair", "task"])
        assert args.repair is True

    def test_repair_flag_disabled(self):
        parser = build_parser()
        args = parser.parse_args(["--no-repair", "task"])
        assert args.repair is False


# ═══════════════════════════════════════════════════════════════════════════
# Repo path validation
# ═══════════════════════════════════════════════════════════════════════════


class TestRepoValidation:
    """Tests for repository path validation in main()."""

    def test_invalid_repo_returns_1(self):
        exit_code = main(["--repo", "/nonexistent_dir_xyz_abc", "task"])
        assert exit_code == 1

    def test_valid_repo_accepted(self, tmp_path: Path):
        """A valid repo path should not fail at the validation step."""
        # We mock OllamaModel and run_agent_loop so we don't need Ollama
        fake_state = AgentState(task="t", status="success")
        with patch("main.OllamaModel") as mock_model_cls, \
             patch("main.run_agent_loop", return_value=fake_state):
            mock_model_cls.return_value = MagicMock()
            exit_code = main(["--repo", str(tmp_path), "test task"])
        assert exit_code == 0


# ═══════════════════════════════════════════════════════════════════════════
# Exit codes
# ═══════════════════════════════════════════════════════════════════════════


class TestExitCodes:
    """main() must return 0 on success and 1 on failure."""

    def _run_with_state(self, status: str, tmp_path: Path, **kwargs) -> int:
        fake_state = AgentState(task="t", status=status, **kwargs)
        with patch("main.OllamaModel") as mock_cls, \
             patch("main.run_agent_loop", return_value=fake_state):
            mock_cls.return_value = MagicMock()
            return main(["--repo", str(tmp_path), "task"])

    def test_success_returns_0(self, tmp_path: Path):
        assert self._run_with_state("success", tmp_path) == 0

    def test_max_steps_returns_1(self, tmp_path: Path):
        assert self._run_with_state(
            "max_steps", tmp_path, error="limit"
        ) == 1

    def test_max_repairs_returns_1(self, tmp_path: Path):
        assert self._run_with_state(
            "max_repairs", tmp_path, error="exhausted"
        ) == 1

    def test_error_returns_1(self, tmp_path: Path):
        assert self._run_with_state(
            "error", tmp_path, error="model error"
        ) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Summary printing
# ═══════════════════════════════════════════════════════════════════════════


class TestPrintSummary:
    """Tests for the _print_summary output."""

    def test_success_summary(self, capsys):
        state = AgentState(task="t", status="success", step_count=3, repair_count=0)
        _print_summary(state)
        out = capsys.readouterr().out
        assert "SUCCESS" in out
        assert "Steps: 3" in out
        assert "Repairs: 0" in out

    def test_failure_summary(self, capsys):
        state = AgentState(
            task="t", status="max_repairs",
            step_count=5, repair_count=3,
            error="Exhausted 3 repair attempt(s)",
        )
        _print_summary(state)
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "Steps: 5" in out
        assert "Repairs: 3" in out
        assert "Exhausted" in out

    def test_error_summary(self, capsys):
        state = AgentState(
            task="t", status="error",
            step_count=1, repair_count=0,
            error="Model error: connection refused",
        )
        _print_summary(state)
        out = capsys.readouterr().out
        assert "ERROR" in out
        assert "connection refused" in out


# ═══════════════════════════════════════════════════════════════════════════
# Progress callback integration
# ═══════════════════════════════════════════════════════════════════════════


class TestProgressOutput:
    """Verify that main() passes the on_tool_result callback and that
    the progress lines appear in stdout."""

    def test_progress_lines_printed(self, tmp_path: Path, capsys):
        """Simulate a 2-step run and verify progress output."""
        fake_state = AgentState(
            task="t", status="success", step_count=2, repair_count=0,
            tool_results=[
                ToolResult(tool_name="read_file", success=True, output="x"),
                ToolResult(tool_name="write_patch", success=True, output="ok"),
            ],
        )

        def fake_run(*, task, model, config, max_steps, max_repairs,
                     repair_enabled, on_tool_result):
            # Simulate callbacks
            if on_tool_result:
                on_tool_result(1, "read_file",
                               ToolResult(tool_name="read_file", success=True, output="x"), 0)
                on_tool_result(2, "write_patch",
                               ToolResult(tool_name="write_patch", success=True, output="ok"), 0)
            return fake_state

        with patch("main.OllamaModel") as mock_cls, \
             patch("main.run_agent_loop", side_effect=fake_run):
            mock_cls.return_value = MagicMock()
            exit_code = main(["--repo", str(tmp_path), "test task"])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "Step 1" in out
        assert "read_file" in out
        assert "Step 2" in out
        assert "write_patch" in out
        assert "success" in out.lower()

    def test_repair_header_printed(self, tmp_path: Path, capsys):
        """Verify that a repair header is printed when repair_count changes."""
        fake_state = AgentState(
            task="t", status="success", step_count=3, repair_count=1,
        )

        def fake_run(*, task, model, config, max_steps, max_repairs,
                     repair_enabled, on_tool_result):
            if on_tool_result:
                on_tool_result(1, "run_tests",
                               ToolResult(tool_name="run_tests", success=False,
                                          output="FAIL", exit_code=1), 0)
                # repair_count jumps to 1
                on_tool_result(2, "write_patch",
                               ToolResult(tool_name="write_patch", success=True, output="ok"), 1)
                on_tool_result(3, "run_tests",
                               ToolResult(tool_name="run_tests", success=True,
                                          output="PASS", exit_code=0), 1)
            return fake_state

        with patch("main.OllamaModel") as mock_cls, \
             patch("main.run_agent_loop", side_effect=fake_run):
            mock_cls.return_value = MagicMock()
            exit_code = main(["--repo", str(tmp_path), "fix bug"])

        out = capsys.readouterr().out
        assert "Repair 1" in out
        assert "Step 2" in out
        assert "Step 3" in out


# ═══════════════════════════════════════════════════════════════════════════
# Model construction error
# ═══════════════════════════════════════════════════════════════════════════


class TestModelConstructionError:
    """If OllamaModel raises, main() should return 1 without a traceback."""

    def test_model_init_failure(self, tmp_path: Path):
        with patch("main.OllamaModel", side_effect=Exception("bad host")):
            exit_code = main(["--repo", str(tmp_path), "task"])
        assert exit_code == 1
