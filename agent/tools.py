"""Built-in tools available to the coding agent.

Each tool is a plain function that returns a structured ``ToolResult``.
All tools are **completely independent of Ollama** — they operate on the
local filesystem and subprocesses only.

Security note
-------------
``run_shell`` is intentionally **unsandboxed** in Phase 1.  It executes
arbitrary shell commands with the privileges of the host process.  This is
acceptable for a locally-operated agent but must be revisited before any
networked or multi-tenant deployment.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from agent.types import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_within_root(path: str, project_root: Path) -> Path:
    """Resolve *path* and verify it lives under *project_root*.

    Raises ``ValueError`` if the resolved path escapes the root.
    """
    root = project_root.resolve()
    resolved = (root / path).resolve()
    # Use os.path-style prefix check so "root" itself is allowed
    if not (resolved == root or str(resolved).startswith(str(root) + resolved.anchor[-1:])):
        raise ValueError(
            f"Path escapes project root: {path!r} resolves to {resolved}"
        )
    return resolved


def _is_within_root(resolved: Path, root: Path) -> bool:
    """Return True if *resolved* is equal to or a child of *root*."""
    try:
        resolved.relative_to(root)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

def read_file(path: str, *, project_root: Path) -> ToolResult:
    """Read and return the UTF-8 contents of a file.

    Parameters
    ----------
    path:
        Relative or absolute path to the file.
    project_root:
        The project root directory — access outside this tree is denied.

    Returns
    -------
    ToolResult
        ``success=True`` with file contents in ``output``, or
        ``success=False`` with a human-readable ``error``.
    """
    root = project_root.resolve()
    try:
        resolved = (root / path).resolve()
    except (OSError, ValueError):
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"Invalid path: {path!r}",
        )

    if not _is_within_root(resolved, root):
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"Access denied: path is outside project root",
        )

    if not resolved.exists():
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"File not found: {path!r}",
        )

    if resolved.is_dir():
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"Path is a directory, not a file: {path!r}",
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"File is not valid UTF-8: {path!r}",
        )
    except OSError as exc:
        return ToolResult(
            tool_name="read_file",
            success=False,
            output="",
            error=f"Cannot read file: {path!r}",
        )

    return ToolResult(tool_name="read_file", success=True, output=content)


# ---------------------------------------------------------------------------
# write_patch
# ---------------------------------------------------------------------------

def write_patch(
    path: str,
    old_str: str,
    new_str: str,
    *,
    project_root: Path,
) -> ToolResult:
    """Apply an exact search-and-replace patch to a file.

    Rules
    -----
    * Zero matches → fail (``old_str`` not found).
    * More than one match → fail (ambiguous).
    * Exactly one match → replace that single occurrence and write back.
    * The file is **never** silently overwritten in full.

    Parameters
    ----------
    path:
        Relative or absolute path to the target file.
    old_str:
        The exact substring to find.
    new_str:
        The replacement string.
    project_root:
        The project root directory.

    Returns
    -------
    ToolResult
    """
    root = project_root.resolve()
    try:
        resolved = (root / path).resolve()
    except (OSError, ValueError):
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"Invalid path: {path!r}",
        )

    if not _is_within_root(resolved, root):
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"Access denied: path is outside project root",
        )

    if not resolved.exists():
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"File not found: {path!r}",
        )

    if resolved.is_dir():
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"Path is a directory, not a file: {path!r}",
        )

    try:
        content = resolved.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"Cannot read file: {path!r}",
        )

    match_count = content.count(old_str)

    if match_count == 0:
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"old_str not found in {path!r} (zero matches)",
        )

    if match_count > 1:
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=(
                f"Ambiguous patch: old_str appears {match_count} times "
                f"in {path!r} (expected exactly 1)"
            ),
        )

    # Exactly one match — apply the patch
    new_content = content.replace(old_str, new_str, 1)

    try:
        resolved.write_text(new_content, encoding="utf-8")
    except OSError:
        return ToolResult(
            tool_name="write_patch",
            success=False,
            output="",
            error=f"Cannot write file: {path!r}",
        )

    return ToolResult(
        tool_name="write_patch",
        success=True,
        output=f"Patched {path!r}: replaced 1 occurrence",
    )


# ---------------------------------------------------------------------------
# run_shell
# ---------------------------------------------------------------------------

def run_shell(
    command: str,
    *,
    cwd: str | None = None,
    timeout: int = 60,
) -> ToolResult:
    """Execute a shell command and return structured results.

    .. warning::

        This tool is **intentionally unsandboxed** in Phase 1.  It runs
        commands with the full privileges of the host process.  This is
        appropriate for a locally-operated coding agent but must be
        hardened before any networked deployment.

    Parameters
    ----------
    command:
        The shell command string to execute.
    cwd:
        Working directory for the command (defaults to current dir).
    timeout:
        Maximum seconds before the process is killed.

    Returns
    -------
    ToolResult
        Contains ``exit_code``, ``stdout`` (in ``output``), ``stderr``,
        ``duration``, and ``command``.
    """
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        elapsed = time.monotonic() - start
        return ToolResult(
            tool_name="run_shell",
            success=(proc.returncode == 0),
            output=proc.stdout,
            error=proc.stderr if proc.returncode != 0 else None,
            exit_code=proc.returncode,
            stderr=proc.stderr,
            duration=round(elapsed, 3),
            command=command,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return ToolResult(
            tool_name="run_shell",
            success=False,
            output="",
            error=f"Command timed out after {timeout} seconds",
            exit_code=None,
            stderr=None,
            duration=round(elapsed, 3),
            command=command,
        )
    except OSError as exc:
        elapsed = time.monotonic() - start
        return ToolResult(
            tool_name="run_shell",
            success=False,
            output="",
            error=f"Execution error: {exc}",
            exit_code=None,
            stderr=None,
            duration=round(elapsed, 3),
            command=command,
        )


# ---------------------------------------------------------------------------
# run_tests
# ---------------------------------------------------------------------------

def run_tests(
    *,
    test_cmd: str = "python -m pytest -q",
    cwd: str | None = None,
    test_target: str | None = None,
    timeout: int = 120,
) -> ToolResult:
    """Run the project test suite and report pass/fail via exit code.

    Success is determined **solely** by the subprocess exit code — the
    tool never inspects stdout for strings like ``"passed"`` or
    ``"failed"``.

    Parameters
    ----------
    test_cmd:
        The base command to execute (default: ``python -m pytest -q``).
    cwd:
        Working directory (default: current directory).
    test_target:
        The specific test target to run (e.g., ``tests/``, ``tests/test_loop.py``).
        Prevents running unintended tests.
    timeout:
        Maximum seconds before the process is killed.

    Returns
    -------
    ToolResult
        Contains ``exit_code``, ``stdout``, ``stderr``, ``duration``,
        and ``command``.
    """
    full_cmd = test_cmd
    if test_target:
        full_cmd = f"{test_cmd} {test_target}"
        
    start = time.monotonic()
    try:
        proc = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        elapsed = time.monotonic() - start
        
        output_data = {
            "command": full_cmd,
            "working_directory": cwd or str(Path.cwd()),
            "test_target": test_target,
            "exit_code": proc.returncode,
            "success": (proc.returncode == 0),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration": round(elapsed, 3),
        }
        
        return ToolResult(
            tool_name="run_tests",
            success=(proc.returncode == 0),
            output=proc.stdout + f"\n[Executed: {full_cmd} in {output_data['working_directory']}]",
            error=proc.stderr if proc.returncode != 0 else None,
            exit_code=proc.returncode,
            stderr=proc.stderr,
            duration=round(elapsed, 3),
            command=full_cmd,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return ToolResult(
            tool_name="run_tests",
            success=False,
            output="",
            error=f"Tests timed out after {timeout} seconds",
            exit_code=None,
            stderr=None,
            duration=round(elapsed, 3),
            command=full_cmd,
        )
    except OSError as exc:
        elapsed = time.monotonic() - start
        return ToolResult(
            tool_name="run_tests",
            success=False,
            output="",
            error=f"Execution error: {exc}",
            exit_code=None,
            stderr=None,
            duration=round(elapsed, 3),
            command=full_cmd,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, callable] = {
    "read_file": read_file,
    "write_patch": write_patch,
    "run_shell": run_shell,
    "run_tests": run_tests,
}
