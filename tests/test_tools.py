"""Tests for agent.tools.

Every test uses temporary directories so the real repository is never
modified during the test run.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from agent.tools import (
    TOOL_REGISTRY,
    read_file,
    run_shell,
    run_tests,
    write_patch,
)
from agent.types import ToolResult


# ── helpers ────────────────────────────────────────────────────────────────


def _write(tmp: Path, name: str, content: str) -> Path:
    """Write a file into *tmp* and return its path."""
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


# ══════════════════════════════════════════════════════════════════════════
# read_file
# ══════════════════════════════════════════════════════════════════════════


class TestReadFile:
    """Tests for the read_file tool."""

    def test_existing_file(self, tmp_path: Path):
        _write(tmp_path, "hello.txt", "hello world")
        result = read_file("hello.txt", project_root=tmp_path)
        assert result.success is True
        assert result.output == "hello world"
        assert result.error is None
        assert result.tool_name == "read_file"

    def test_missing_file(self, tmp_path: Path):
        result = read_file("nope.txt", project_root=tmp_path)
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_directory_rejected(self, tmp_path: Path):
        (tmp_path / "subdir").mkdir()
        result = read_file("subdir", project_root=tmp_path)
        assert result.success is False
        assert "directory" in result.error.lower()

    def test_path_outside_project_root(self, tmp_path: Path):
        """Paths that escape the project root must be denied."""
        result = read_file("../../etc/passwd", project_root=tmp_path)
        assert result.success is False
        assert "denied" in result.error.lower() or "outside" in result.error.lower()

    def test_decode_error(self, tmp_path: Path):
        """Binary content that is not valid UTF-8 must fail gracefully."""
        binary_file = tmp_path / "binary.bin"
        binary_file.write_bytes(b"\x80\x81\x82\xff\xfe")
        result = read_file("binary.bin", project_root=tmp_path)
        assert result.success is False
        assert "utf-8" in result.error.lower()

    def test_no_raw_traceback_on_error(self, tmp_path: Path):
        """Error messages must not contain raw Python tracebacks."""
        result = read_file("does_not_exist.py", project_root=tmp_path)
        assert result.success is False
        assert "Traceback" not in (result.error or "")
        assert "Traceback" not in result.output


# ══════════════════════════════════════════════════════════════════════════
# write_patch
# ══════════════════════════════════════════════════════════════════════════


class TestWritePatch:
    """Tests for the write_patch tool."""

    def test_successful_patch(self, tmp_path: Path):
        _write(tmp_path, "code.py", "x = 1\ny = 2\nz = 3\n")
        result = write_patch("code.py", "y = 2", "y = 42", project_root=tmp_path)
        assert result.success is True

        patched = (tmp_path / "code.py").read_text(encoding="utf-8")
        assert "y = 42" in patched
        assert "y = 2" not in patched
        # Other lines untouched
        assert "x = 1" in patched
        assert "z = 3" in patched

    def test_zero_matches(self, tmp_path: Path):
        _write(tmp_path, "code.py", "x = 1\n")
        result = write_patch("code.py", "NOT_HERE", "anything", project_root=tmp_path)
        assert result.success is False
        assert "zero" in result.error.lower() or "not found" in result.error.lower()
        # File must be unchanged
        assert (tmp_path / "code.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_multiple_matches(self, tmp_path: Path):
        _write(tmp_path, "dup.py", "a = 1\na = 1\na = 1\n")
        result = write_patch("dup.py", "a = 1", "a = 99", project_root=tmp_path)
        assert result.success is False
        assert "ambiguous" in result.error.lower() or "3" in result.error
        # File must be unchanged
        assert (tmp_path / "dup.py").read_text(encoding="utf-8") == "a = 1\na = 1\na = 1\n"

    def test_missing_file(self, tmp_path: Path):
        result = write_patch("nope.py", "a", "b", project_root=tmp_path)
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_path_outside_root(self, tmp_path: Path):
        result = write_patch("../../evil.py", "a", "b", project_root=tmp_path)
        assert result.success is False
        assert "denied" in result.error.lower() or "outside" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════
# run_shell
# ══════════════════════════════════════════════════════════════════════════


class TestRunShell:
    """Tests for the run_shell tool."""

    def test_successful_command(self):
        result = run_shell("echo hello")
        assert result.success is True
        assert result.exit_code == 0
        assert "hello" in result.output
        assert result.duration is not None
        assert result.duration >= 0
        assert result.command == "echo hello"

    def test_failing_command(self):
        # 'cmd /c exit 1' works reliably on Windows PowerShell
        result = run_shell("cmd /c exit 1")
        assert result.success is False
        assert result.exit_code == 1

    def test_timeout(self):
        # A command that sleeps for far longer than the timeout
        if sys.platform == "win32":
            cmd = "ping -n 30 127.0.0.1"
        else:
            cmd = "sleep 30"
        result = run_shell(cmd, timeout=1)
        assert result.success is False
        assert "timed out" in result.error.lower()
        assert result.duration is not None

    def test_stderr_captured(self):
        """stderr must be captured even on success."""
        if sys.platform == "win32":
            cmd = 'python -c "import sys; sys.stderr.write(\'warn\\n\')"'
        else:
            cmd = "python3 -c \"import sys; sys.stderr.write('warn\\n')\""
        result = run_shell(cmd)
        assert result.success is True
        assert result.stderr is not None
        assert "warn" in result.stderr


# ══════════════════════════════════════════════════════════════════════════
# run_tests
# ══════════════════════════════════════════════════════════════════════════


class TestRunTests:
    """Tests for the run_tests tool."""

    def test_passing_tests(self, tmp_path: Path):
        """A trivial passing test suite must report success=True via exit code."""
        _write(
            tmp_path,
            "test_ok.py",
            textwrap.dedent("""\
                def test_pass():
                    assert True
            """),
        )
        result = run_tests(
            test_cmd=f"python -m pytest -q {tmp_path / 'test_ok.py'}",
            cwd=str(tmp_path),
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.command is not None
        assert result.duration is not None

    def test_failing_tests(self, tmp_path: Path):
        """A failing test must report success=False via exit code, not string search."""
        _write(
            tmp_path,
            "test_fail.py",
            textwrap.dedent("""\
                def test_fail():
                    assert False, "intentional failure"
            """),
        )
        result = run_tests(
            test_cmd=f"python -m pytest -q {tmp_path / 'test_fail.py'}",
            cwd=str(tmp_path),
        )
        assert result.success is False
        assert result.exit_code != 0
        assert result.duration is not None

    def test_timeout(self, tmp_path: Path):
        """Tests that exceed the timeout must fail gracefully."""
        _write(
            tmp_path,
            "test_hang.py",
            textwrap.dedent("""\
                import time
                def test_slow():
                    time.sleep(60)
            """),
        )
        result = run_tests(
            test_cmd=f"python -m pytest -q {tmp_path / 'test_hang.py'}",
            cwd=str(tmp_path),
            timeout=2,
        )
        assert result.success is False
        assert "timed out" in result.error.lower()


# ══════════════════════════════════════════════════════════════════════════
# TOOL_REGISTRY
# ══════════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """Tests for the TOOL_REGISTRY mapping."""

    def test_registry_keys(self):
        expected = {"read_file", "write_patch", "run_shell", "run_tests"}
        assert set(TOOL_REGISTRY.keys()) == expected

    def test_registry_values_are_callable(self):
        for name, func in TOOL_REGISTRY.items():
            assert callable(func), f"{name} is not callable"
