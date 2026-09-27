"""Agent loop — orchestrates model calls and tool execution.

This is the core harness that drives the coding agent through:

    user task → model → tool call → tool execution → observation
    → model → continue …

It delegates all model interaction to :mod:`agent.model` and all tool
execution to :mod:`agent.tools`.  No logic from those modules is
duplicated here.

No Colab, Docker, MCP, async, RAG, memory, or benchmark logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from agent.prompts import SYSTEM_PROMPT, build_repair_prompt
from agent.tools import TOOL_REGISTRY, read_file, run_shell, run_tests, write_patch
from agent.types import AgentConfig, ToolResult
from agent.trajectory import log_trajectory_step
import uuid

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocols — so the loop can work with any model (including mocks)
# ---------------------------------------------------------------------------

class ModelProtocol(Protocol):
    """Minimal interface the loop requires from a model object."""

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[Any] | None = None,
        temperature: float | None = None,
    ) -> Any: ...

    @staticmethod
    def has_tool_calls(response: Any) -> bool: ...

    @staticmethod
    def extract_tool_calls(response: Any) -> list[dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """Mutable state tracked throughout an agent run.

    Attributes
    ----------
    task:
        The original user task description.
    messages:
        Full conversation history (list of dicts).
    tool_calls:
        Chronological record of every tool call the model made.
    tool_results:
        Chronological record of every tool result produced.
    step_count:
        Total model-call steps executed.
    repair_count:
        Number of repair iterations consumed so far.
    status:
        Final outcome: ``"success"``, ``"max_steps"``, ``"max_repairs"``,
        or ``"error"``.
    error:
        Description of the terminal error, if any.
    """

    task: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    step_count: int = 0
    repair_count: int = 0
    status: str = "running"
    error: str | None = None


# ---------------------------------------------------------------------------
# Tool definitions passed to the model (Google-style docstrings)
# ---------------------------------------------------------------------------

def _tool_read_file(path: str) -> str:
    """Read the UTF-8 contents of a file.

    Args:
        path: Relative path to the file within the project.

    Returns:
        str: The file contents.
    """
    return ""  # never called — definition only


def _tool_write_patch(path: str, old_str: str, new_str: str) -> str:
    """Apply an exact search-and-replace edit to a file.

    Args:
        path: Relative path to the file within the project.
        old_str: The exact existing text to find (must appear exactly once).
        new_str: The replacement text.

    Returns:
        str: Confirmation message.
    """
    return ""


def _tool_run_shell(command: str) -> str:
    """Execute a shell command.

    Args:
        command: The shell command to run.

    Returns:
        str: Command output.
    """
    return ""


def _tool_run_tests(test_target: str | None = None) -> str:
    """Run the project test suite and report pass/fail.

    Args:
        test_target: The specific test file or directory to run (optional).

    Returns:
        str: Test results.
    """
    return ""


def _tool_finish() -> str:
    """Explicitly mark the task as finished and successful.
    
    Use this ONLY when you have verified your changes and no further action is needed,
    and you are not using run_tests (which finishes automatically on success).

    Returns:
        str: Confirmation.
    """
    return ""


# The list of tool *definitions* (callables) passed to the model.
# The model sees their names, docstrings, and parameter types but
# never executes them — the loop dispatches to the real tools.
TOOL_DEFINITIONS: list[Any] = [
    _tool_read_file,
    _tool_write_patch,
    _tool_run_shell,
    _tool_run_tests,
    _tool_finish,
]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    project_root: Path,
    shell_timeout: int,
) -> ToolResult:
    """Execute a tool by name and return a ``ToolResult``.

    Handles unknown tools, invalid arguments, and tool exceptions
    gracefully — always returns a ``ToolResult``, never raises.
    """
    if name == "read_file":
        path = arguments.get("path")
        if path is None:
            return ToolResult(
                tool_name=name, success=False, output="",
                error="Missing required argument: 'path'",
            )
        return read_file(str(path), project_root=project_root)

    if name == "write_patch":
        path = arguments.get("path")
        old_str = arguments.get("old_str")
        new_str = arguments.get("new_str")
        missing = [
            k for k, v in [("path", path), ("old_str", old_str), ("new_str", new_str)]
            if v is None
        ]
        if missing:
            return ToolResult(
                tool_name=name, success=False, output="",
                error=f"Missing required argument(s): {', '.join(missing)}",
            )
        return write_patch(str(path), str(old_str), str(new_str), project_root=project_root)

    if name == "run_shell":
        command = arguments.get("command")
        if command is None:
            return ToolResult(
                tool_name=name, success=False, output="",
                error="Missing required argument: 'command'",
            )
        return run_shell(str(command), timeout=shell_timeout, cwd=str(project_root))

    if name == "run_tests":
        test_cmd = arguments.get("test_cmd", "python -m pytest -q")
        test_target = arguments.get("test_target")
        return run_tests(test_cmd=str(test_cmd), test_target=str(test_target) if test_target else None, timeout=shell_timeout, cwd=str(project_root))

    if name == "finish":
        return ToolResult(tool_name="finish", success=True, output="Task marked as finished.")

    # Unknown tool
    return ToolResult(
        tool_name=name, success=False, output="",
        error=f"Unknown tool: {name!r}",
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _tool_result_to_content(result: ToolResult) -> str:
    """Format a ``ToolResult`` as a string to feed back to the model."""
    parts: list[str] = []
    if result.success:
        parts.append(f"[{result.tool_name}] SUCCESS")
    else:
        parts.append(f"[{result.tool_name}] FAILED")

    if result.output:
        parts.append(result.output)
    if result.error:
        parts.append(f"Error: {result.error}")
    if result.exit_code is not None:
        parts.append(f"Exit code: {result.exit_code}")
    if result.stderr:
        parts.append(f"Stderr: {result.stderr}")

    return "\n".join(parts)


def _build_repair_message(result: ToolResult) -> str:
    """Build a repair prompt from a failed ``run_tests`` result."""
    return build_repair_prompt(
        command=result.command or "python -m pytest -q",
        exit_code=result.exit_code if result.exit_code is not None else "N/A",
        stdout=result.output or "",
        stderr=result.stderr or result.error or "",
    )


def _build_repo_context(project_root: Path) -> str:
    """Build lightweight repository context string."""
    try:
        entries = []
        for p in project_root.iterdir():
            if p.name in (".git", "__pycache__", ".pytest_cache"):
                continue
            if p.is_dir():
                entries.append(f"{p.name}/")
            else:
                entries.append(p.name)
        structure = "\n".join(sorted(entries))
    except Exception:
        structure = "(Unable to read repository structure)"
        
    return f"## Repository Context\nRoot: {project_root.resolve()}\nStructure:\n{structure}\n"

def classify_task(task: str) -> str:
    """Classify user task as INSPECTION or MODIFICATION."""
    task_lower = task.lower()
    if task_lower.startswith("inspect ") or task_lower.startswith("read ") or "summarize" in task_lower:
        if not any(w in task_lower for w in ("fix", "change", "update", "modify", "write", "implement", "add")):
            return "INSPECTION"
    return "MODIFICATION"

# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_agent_loop(
    *,
    task: str,
    model: ModelProtocol,
    config: AgentConfig | None = None,
    max_steps: int = 20,
    max_repairs: int = 3,
    repair_enabled: bool = True,
    on_tool_result: Callable[[int, str, ToolResult, int], None] | None = None,
) -> AgentState:
    """Run the agent loop to completion.

    Parameters
    ----------
    task:
        The user's task description.
    model:
        An object satisfying :class:`ModelProtocol` (typically an
        :class:`~agent.model.OllamaModel` or a mock).
    config:
        Agent configuration.  Defaults to :class:`AgentConfig`.
    max_steps:
        Hard upper bound on model-call iterations (prevents infinite loops).
    max_repairs:
        Maximum number of test-failure → repair cycles.
    repair_enabled:
        If ``False``, test failures terminate the loop immediately.
    on_tool_result:
        Optional callback invoked after each tool execution with
        ``(step_count, tool_name, result, repair_count)``.  Useful for
        printing progress in the CLI.

    Returns
    -------
    AgentState
        The final agent state including all messages, tool calls/results,
        and the terminal status.
    """
    cfg = config or AgentConfig()
    state = AgentState(task=task)
    run_id = str(uuid.uuid4())
    log_path = cfg.project_root / "trajectory.jsonl"
    
    runtime = "local" # We don't have remote detection built into config yet, assume local or whatever model.host indicates.

    # Seed the conversation
    repo_context = _build_repo_context(cfg.project_root)
    state.messages.append({"role": "system", "content": SYSTEM_PROMPT})
    state.messages.append({"role": "user", "content": f"{repo_context}\nTask:\n{task}"})

    while state.step_count < max_steps:
        state.step_count += 1
        log.debug("Step %d / %d", state.step_count, max_steps)

        # ── 1. Call the model ──────────────────────────────────────────
        try:
            response = model.chat(
                state.messages,
                tools=TOOL_DEFINITIONS,
            )
        except Exception as exc:
            state.status = "error"
            state.error = f"Model error: {exc}"
            log.error("Model exception at step %d: %s", state.step_count, exc)
            break

        # ── 2. Check for tool calls ───────────────────────────────────
        if not model.has_tool_calls(response):
            content = getattr(response.message, "content", "") or ""
            state.messages.append({"role": "assistant", "content": content})
            
            # The model returned ordinary text instead of a tool call
            log.info("Model returned text instead of a tool call at step %d", state.step_count)
            
            corrective_msg = (
                "Please use your tools to perform the task. "
                "You must verify your work. If no tests are applicable and you have manually "
                "verified the fix, call the `finish` tool. Do not just reply with text."
            )
            state.messages.append({"role": "user", "content": corrective_msg})
            continue

        # ── 3. Process each tool call ─────────────────────────────────
        calls = model.extract_tool_calls(response)

        # Preserve assistant message (content + tool_calls metadata)
        assistant_content = getattr(response.message, "content", "") or ""
        state.messages.append({
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [{"function": call} for call in calls],
        })

        task_type = classify_task(task)

        for call in calls:
            tool_name = call.get("name", "")
            tool_args = call.get("arguments", {})

            # ── Enforce task-appropriate actions ──────────────────────
            if task_type == "INSPECTION" and tool_name in ("write_patch", "run_tests"):
                state.tool_calls.append(call)
                log.debug("Tool call rejected for INSPECTION task: %s(%s)", tool_name, tool_args)
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error=f"Tool {tool_name} is not allowed for INSPECTION tasks. Please continue with the original task."
                )
                state.tool_results.append(result)
                state.messages.append({
                    "role": "tool",
                    "content": _tool_result_to_content(result),
                })
                continue

            state.tool_calls.append(call)
            log.debug("Tool call: %s(%s)", tool_name, tool_args)

            # ── 3a. Execute the tool ──────────────────────────────────
            try:
                result = _dispatch_tool(
                    tool_name,
                    tool_args,
                    project_root=cfg.project_root,
                    shell_timeout=cfg.shell_timeout,
                )
            except Exception as exc:
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output="",
                    error=f"Tool exception: {exc}",
                )

            state.tool_results.append(result)

            # Notify observer (if any)
            if on_tool_result is not None:
                on_tool_result(
                    state.step_count, tool_name, result, state.repair_count,
                )

            # ── 3b. Feed the observation back ─────────────────────────
            state.messages.append({
                "role": "tool",
                "content": _tool_result_to_content(result),
            })

            # ── 3c. Log Trajectory ────────────────────────────────────────
            
            # Record trajectory
            verif_res = None
            if tool_name == "run_tests":
                verif_res = {"success": result.success, "exit_code": result.exit_code}
                
            log_trajectory_step(
                log_path=log_path,
                run_id=run_id,
                task=task,
                model=getattr(model, "model", "unknown"),
                runtime=runtime,
                step=state.step_count,
                action="tool_execution",
                tool_name=tool_name,
                tool_arguments=tool_args,
                tool_result={"success": result.success, "error": result.error},
                verification_result=verif_res,
                repair_attempt=state.repair_count,
                final_status=None,
            )

            # ── 3d. Repair workflow for failed tests ──────────────────
            if tool_name == "run_tests" and not result.success:
                has_attempted_mod = any(
                    tc.get("name") in ("write_patch", "run_shell")
                    for tc in state.tool_calls
                )
                # Only repair if it's a MODIFICATION task and we actually modified something
                if task_type == "MODIFICATION" and has_attempted_mod:
                    if not repair_enabled:
                        state.status = "max_repairs"
                        state.error = "Test failure and repair is disabled"
                        log.info("Tests failed, repair disabled — stopping")
                        
                        log_trajectory_step(log_path, run_id, task, getattr(model, "model", "unknown"), runtime, state.step_count, "terminate", None, None, None, None, state.repair_count, state.status)
                        return state

                    state.repair_count += 1
                    if state.repair_count > max_repairs:
                        state.status = "max_repairs"
                        state.error = (
                            f"Exhausted {max_repairs} repair attempt(s) — "
                            f"tests still failing"
                        )
                        log.info("Max repairs (%d) reached — stopping", max_repairs)
                        log_trajectory_step(log_path, run_id, task, getattr(model, "model", "unknown"), runtime, state.step_count, "terminate", None, None, None, None, state.repair_count, state.status)
                        return state

                    # Inject structured failure evidence
                    repair_msg = _build_repair_message(result)
                    state.messages.append({
                        "role": "user",
                        "content": repair_msg,
                    })
                    log.debug("Injected repair prompt (attempt %d/%d)",
                              state.repair_count, max_repairs)
                else:
                    log.debug("Test failed, but repair is irrelevant or task is INSPECTION.")

            # ── 3e. Stop immediately on verified test success ─────────
            if tool_name == "run_tests" and result.success:
                # Stop on test success only if it's a MODIFICATION task and modified something,
                # or if the user explicitly called run_tests for whatever reason.
                has_attempted_mod = any(
                    tc.get("name") in ("write_patch", "run_shell")
                    for tc in state.tool_calls
                )
                if task_type == "MODIFICATION" and not has_attempted_mod:
                    state.messages.append({
                        "role": "user",
                        "content": "Tests passed, but you haven't made any modifications. Please apply the fix first."
                    })
                else:
                    state.status = "success"
                    log.info("Tests passed at step %d — stopping", state.step_count)
                    log_trajectory_step(log_path, run_id, task, getattr(model, "model", "unknown"), runtime, state.step_count, "terminate", None, None, None, None, state.repair_count, state.status)
                    return state

            # ── 3f. Stop immediately if explicitly finished ───────────
            if tool_name == "finish":
                if task_type == "MODIFICATION":
                    # User requires actual work to have occurred before success
                    has_modifications = any(
                        tc.get("name") == "write_patch"
                        for tc in state.tool_calls
                    )
                    has_verification = any(
                        tc.get("name") in ("run_tests", "run_shell")
                        for tc in state.tool_calls
                    )
                    if not (has_modifications and has_verification):
                        state.messages.append({
                            "role": "user", 
                            "content": "You called finish but have not both modified code AND verified it. Please do the actual work first."
                        })
                        continue
                
                state.status = "success"
                log.info("Agent explicitly called finish at step %d — stopping", state.step_count)
                log_trajectory_step(log_path, run_id, task, getattr(model, "model", "unknown"), runtime, state.step_count, "terminate", None, None, None, None, state.repair_count, state.status)
                return state

    else:
        # While-loop exhausted without breaking
        state.status = "max_steps"
        state.error = f"Reached maximum step limit ({max_steps})"
        log.warning("Agent hit max_steps (%d) without completing", max_steps)
        log_trajectory_step(log_path, run_id, task, getattr(model, "model", "unknown"), runtime, state.step_count, "terminate", None, None, None, None, state.repair_count, state.status)

    return state
