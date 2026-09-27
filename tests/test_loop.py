"""Tests for agent.loop — fully mocked, no live Ollama required.

Every test constructs a ``MockModel`` that returns pre-scripted responses,
letting us verify loop behavior deterministically: tool dispatch, multi-step
workflows, repair cycles, error handling, and termination conditions.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import MagicMock, patch

import pytest

from agent.loop import (
    AgentState,
    _dispatch_tool,
    _tool_result_to_content,
    run_agent_loop,
)
from agent.types import AgentConfig, ToolResult


# ═══════════════════════════════════════════════════════════════════════════
# Mock model helpers
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class _FakeFunction:
    name: str
    arguments: dict[str, Any]


@dataclass
class _FakeToolCall:
    function: _FakeFunction


@dataclass
class _FakeMessage:
    content: str
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeResponse:
    message: _FakeMessage


def _make_tool_response(
    calls: list[dict[str, Any]],
    content: str = "",
) -> _FakeResponse:
    """Build a fake model response containing tool calls."""
    tcs = [
        _FakeToolCall(function=_FakeFunction(name=c["name"], arguments=c["arguments"]))
        for c in calls
    ]
    return _FakeResponse(message=_FakeMessage(content=content, tool_calls=tcs))


def _make_text_response(content: str) -> _FakeResponse:
    """Build a fake model response with text only (no tool calls)."""
    return _FakeResponse(message=_FakeMessage(content=content, tool_calls=None))


class MockModel:
    """A model that returns a pre-scripted sequence of responses.

    Satisfies the ``ModelProtocol`` used by ``run_agent_loop``.
    """

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self._call_index = 0

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[Any] | None = None,
        temperature: float | None = None,
    ) -> _FakeResponse:
        if self._call_index >= len(self._responses):
            # Safety: return a text response to end the loop
            return _make_text_response("(no more scripted responses)")
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp

    @staticmethod
    def has_tool_calls(response: _FakeResponse) -> bool:
        tc = response.message.tool_calls
        return tc is not None and len(tc) > 0

    @staticmethod
    def extract_tool_calls(response: _FakeResponse) -> list[dict[str, Any]]:
        tc = response.message.tool_calls
        if not tc:
            return []
        return [
            {"name": c.function.name, "arguments": dict(c.function.arguments)}
            for c in tc
        ]


# ── Helpers ────────────────────────────────────────────────────────────────

def _write(tmp: Path, name: str, content: str) -> Path:
    p = tmp / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def _config(tmp: Path) -> AgentConfig:
    return AgentConfig(project_root=tmp, shell_timeout=10)


# ═══════════════════════════════════════════════════════════════════════════
# Test: successful single tool call
# ═══════════════════════════════════════════════════════════════════════════


class TestSuccessfulToolCall:
    """Model reads a file, then responds with text."""

    def test_read_file_then_done(self, tmp_path: Path):
        _write(tmp_path, "hello.py", "print('hi')\n")

        model = MockModel([
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "hello.py"}},
            ]),
            _make_tool_response([
                {"name": "finish", "arguments": {}},
            ]),
        ])

        state = run_agent_loop(
            task="Read hello.py",
            model=model,
            config=_config(tmp_path),
            max_steps=10,
        )

        assert state.status == "success"
        assert state.step_count == 2

# ═══════════════════════════════════════════════════════════════════════════
# Test: Success criteria and explicit terminal states
# ═══════════════════════════════════════════════════════════════════════════

class TestSuccessCriteria:
    def test_ordinary_text_is_not_success(self, tmp_path: Path):
        model = MockModel([
            _make_text_response("I will fix this for you."),
            _make_text_response("Still thinking about it..."),
        ])
        state = run_agent_loop(task="Fix it", model=model, config=_config(tmp_path), max_steps=2)
        # Should not be 'success', should exhaust steps trying to correct the model
        assert state.status == "max_steps"
        
    def test_asks_for_details_is_not_success(self, tmp_path: Path):
        model = MockModel([
            _make_text_response("Understood! Please provide the details of the coding task you need assistance with."),
        ])
        state = run_agent_loop(task="Do something", model=model, config=_config(tmp_path), max_steps=1)
        assert state.status == "max_steps"

    def test_read_without_verify_is_not_success(self, tmp_path: Path):
        _write(tmp_path, "a.py", "x = 1")
        model = MockModel([
            _make_tool_response([{"name": "read_file", "arguments": {"path": "a.py"}}]),
            _make_tool_response([{"name": "finish", "arguments": {}}]), # Try to finish without modifying
        ])
        state = run_agent_loop(task="Fix it", model=model, config=_config(tmp_path), max_steps=2)
        assert state.status == "max_steps" # The finish tool fails its check, loop continues
        
    def test_finish_with_modifications_is_success(self, tmp_path: Path):
        _write(tmp_path, "a.py", "x = 1")
        model = MockModel([
            _make_tool_response([
                {"name": "write_patch", "arguments": {"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"}},
                {"name": "run_tests", "arguments": {}}
            ]),
            _make_tool_response([{"name": "finish", "arguments": {}}]),
        ])
        state = run_agent_loop(task="Fix it", model=model, config=_config(tmp_path), max_steps=3)
        assert state.status == "success"

# ═══════════════════════════════════════════════════════════════════════════
# Test: multiple tool calls in sequence
# ═══════════════════════════════════════════════════════════════════════════


class TestMultipleToolCalls:
    """Model reads a file, patches it, then responds with text."""

    def test_read_then_patch(self, tmp_path: Path):
        _write(tmp_path, "code.py", "x = 1\n")

        model = MockModel([
            # Step 1: read
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "code.py"}},
            ]),
            # Step 2: patch and verify
            _make_tool_response([
                {"name": "write_patch", "arguments": {
                    "path": "code.py", "old_str": "x = 1", "new_str": "x = 42",
                }},
                {"name": "run_tests", "arguments": {}}
            ]),
            # Step 3: finish
            _make_tool_response([
                {"name": "finish", "arguments": {}},
            ]),
        ])

        state = run_agent_loop(
            task="Change x to 42",
            model=model,
            config=_config(tmp_path),
        )

        assert state.status == "success"
        assert len(state.tool_calls) == 4
        assert state.tool_calls[0]["name"] == "read_file"
        assert state.tool_calls[1]["name"] == "write_patch"
        assert state.tool_calls[2]["name"] == "run_tests"
        assert state.tool_calls[3]["name"] == "finish"
        assert state.tool_results[1].success is True
        # Verify the actual file was patched
        assert "x = 42" in (tmp_path / "code.py").read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# Test: multiple tool calls in a single response
# ═══════════════════════════════════════════════════════════════════════════


class TestMultipleToolCallsInOneResponse:
    """Model returns two tool calls in a single response."""

    def test_two_reads(self, tmp_path: Path):
        _write(tmp_path, "a.txt", "aaa")
        _write(tmp_path, "b.txt", "bbb")

        model = MockModel([
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "a.txt"}},
                {"name": "read_file", "arguments": {"path": "b.txt"}},
            ]),
            _make_text_response("Read both files."),
        ])

        state = run_agent_loop(
            task="Read both files",
            model=model,
            config=_config(tmp_path),
        )

        assert state.status == "max_steps"
        assert len(state.tool_calls) >= 2
        assert len(state.tool_results) == 2
        assert state.tool_results[0].output == "aaa"
        assert state.tool_results[1].output == "bbb"


# ═══════════════════════════════════════════════════════════════════════════
# Test: test failure (no repair)
# ═══════════════════════════════════════════════════════════════════════════


class TestTestFailureNoRepair:
    """Test failure with repair disabled terminates immediately."""

    def test_failure_stops(self, tmp_path: Path):
        _write(
            tmp_path, "test_fail.py",
            "def test_bad():\n    assert False\n",
        )

        model = MockModel([
            _make_tool_response([
                {"name": "write_patch", "arguments": {"path": "test_fail.py", "old_str": "def test_bad():", "new_str": "def test_bad_still_fails():"}},
                {"name": "run_tests", "arguments": {
                    "test_cmd": f"python -m pytest -q {tmp_path / 'test_fail.py'}",
                }},
            ]),
        ])

        state = run_agent_loop(
            task="Run failing tests",
            model=model,
            config=_config(tmp_path),
            repair_enabled=False,
        )

        assert state.status == "max_repairs"
        assert "disabled" in state.error.lower()
        assert state.repair_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# Test: repair workflow
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairWorkflow:
    """Test failure → repair prompt injected → model fixes → tests pass."""

    def test_successful_repair(self, tmp_path: Path):
        # Write code with a bug
        _write(tmp_path, "lib.py", "def add(a, b):\n    return a - b\n")
        _write(
            tmp_path, "test_lib.py",
            textwrap.dedent("""\
                from lib import add
                def test_add():
                    assert add(2, 3) == 5
            """),
        )

        model = MockModel([
            # Step 1: model reads and patches but has bug
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "lib.py"}},
                {"name": "write_patch", "arguments": {
                    "path": "lib.py",
                    "old_str": "return a - b",
                    "new_str": "return a * b",
                }},
                {"name": "run_tests", "arguments": {
                    "test_cmd": f"python -m pytest -q {tmp_path / 'test_lib.py'}",
                }},
            ]),
            # Step 2: repair — model reads file
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "lib.py"}},
            ]),
            # Step 3: repair — model patches to correct
            _make_tool_response([
                {"name": "write_patch", "arguments": {
                    "path": "lib.py",
                    "old_str": "return a * b",
                    "new_str": "return a + b",
                }},
            ]),
            # Step 4: model runs tests again → pass
            _make_tool_response([
                {"name": "run_tests", "arguments": {
                    "test_cmd": f"python -m pytest -q {tmp_path / 'test_lib.py'}",
                }},
            ]),
        ])

        state = run_agent_loop(
            task="Fix the add function",
            model=model,
            config=_config(tmp_path),
            max_repairs=3,
        )

        assert state.status == "success"
        assert state.repair_count == 1
        # Verify the file was actually fixed
        fixed = (tmp_path / "lib.py").read_text(encoding="utf-8")
        assert "return a + b" in fixed

        # Verify repair prompt was injected with failure evidence
        repair_msgs = [
            m for m in state.messages
            if m["role"] == "user" and "tests failed" in m["content"].lower()
        ]
        assert len(repair_msgs) >= 1
        assert "exit code" in repair_msgs[0]["content"].lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: repair limit exhausted
# ═══════════════════════════════════════════════════════════════════════════


class TestRepairLimitExhausted:
    """After max_repairs, the loop stops even if tests still fail."""

    def test_max_repairs_reached(self, tmp_path: Path):
        _write(
            tmp_path, "test_always_fail.py",
            "def test_fail():\n    assert False\n",
        )

        test_cmd = f"python -m pytest -q {tmp_path / 'test_always_fail.py'}"

        # Model keeps making mods and running tests that always fail
        responses = []
        for _ in range(5):
            responses.append(
                _make_tool_response([
                    {"name": "write_patch", "arguments": {"path": "test_always_fail.py", "old_str": "def test_fail():", "new_str": "def test_fail2():"}},
                    {"name": "run_tests", "arguments": {"test_cmd": test_cmd}},
                ])
            )

        model = MockModel(responses)

        state = run_agent_loop(
            task="Fix something unfixable",
            model=model,
            config=_config(tmp_path),
            max_repairs=2,
        )

        assert state.status == "max_repairs"
        assert state.repair_count > 2  # exceeded the limit
        assert "exhaust" in state.error.lower() or "2" in state.error


# ═══════════════════════════════════════════════════════════════════════════
# Test: unknown tool
# ═══════════════════════════════════════════════════════════════════════════


class TestUnknownTool:
    """The loop handles an unknown tool name gracefully."""

    def test_unknown_tool_returns_error(self, tmp_path: Path):
        model = MockModel([
            _make_tool_response([
                {"name": "delete_everything", "arguments": {}},
            ]),
            _make_text_response("I see that tool doesn't exist."),
        ])

        state = run_agent_loop(
            task="Try unknown tool",
            model=model,
            config=_config(tmp_path),
        )

        assert state.status == "max_steps"  # loop finishes by hitting max_steps since text is no longer success
        assert len(state.tool_results) == 1
        assert state.tool_results[0].success is False
        assert "unknown" in state.tool_results[0].error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: invalid arguments
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidArguments:
    """The loop handles missing/invalid tool arguments gracefully."""

    def test_missing_path_arg(self, tmp_path: Path):
        model = MockModel([
            # read_file without 'path' argument
            _make_tool_response([
                {"name": "read_file", "arguments": {}},
            ]),
            _make_text_response("I see I forgot the path."),
        ])

        state = run_agent_loop(
            task="Try invalid args",
            model=model,
            config=_config(tmp_path),
        )

        assert state.status == "max_steps"
        assert state.tool_results[0].success is False
        assert "missing" in state.tool_results[0].error.lower()

    def test_missing_write_patch_args(self, tmp_path: Path):
        model = MockModel([
            _make_tool_response([
                {"name": "write_patch", "arguments": {"path": "foo.py"}},
            ]),
            _make_text_response("Oops, forgot old_str and new_str."),
        ])

        state = run_agent_loop(
            task="Bad patch args",
            model=model,
            config=_config(tmp_path),
        )

        assert state.tool_results[0].success is False
        assert "missing" in state.tool_results[0].error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: tool exception
# ═══════════════════════════════════════════════════════════════════════════


class TestToolException:
    """An unexpected exception inside a tool is caught gracefully."""

    def test_exception_caught(self, tmp_path: Path):
        model = MockModel([
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "some.py"}},
            ]),
            _make_text_response("File not found, I see."),
        ])

        # Patch the dispatch to raise an unexpected error
        with patch("agent.loop.read_file", side_effect=RuntimeError("boom")):
            state = run_agent_loop(
                task="trigger exception",
                model=model,
                config=_config(tmp_path),
            )

        assert state.status == "max_steps"  # loop continues and hits max_steps
        assert state.tool_results[0].success is False
        assert "exception" in state.tool_results[0].error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: model exception
# ═══════════════════════════════════════════════════════════════════════════


class TestModelException:
    """A model error (e.g. connection refused) terminates gracefully."""

    def test_model_error_stops_loop(self, tmp_path: Path):
        model = MockModel([])
        # Override chat to raise
        model.chat = MagicMock(side_effect=ConnectionError("Ollama down"))

        state = run_agent_loop(
            task="Try with broken model",
            model=model,
            config=_config(tmp_path),
        )

        assert state.status == "error"
        assert "model error" in state.error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: max-step termination
# ═══════════════════════════════════════════════════════════════════════════


class TestMaxStepTermination:
    """The loop terminates after max_steps even if the model keeps calling tools."""

    def test_max_steps_hit(self, tmp_path: Path):
        _write(tmp_path, "x.txt", "data")

        # Model always reads a file, never finishes
        responses = [
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "x.txt"}},
            ])
            for _ in range(20)
        ]
        model = MockModel(responses)

        state = run_agent_loop(
            task="Loop forever",
            model=model,
            config=_config(tmp_path),
            max_steps=5,
        )

        assert state.status == "max_steps"
        assert state.step_count == 5
        assert "max" in state.error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: state tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestStateTracking:
    """Verify all state fields are populated correctly."""

    def test_state_fields(self, tmp_path: Path):
        _write(tmp_path, "f.txt", "hello")

        model = MockModel([
            _make_tool_response([
                {"name": "read_file", "arguments": {"path": "f.txt"}},
            ]),
            _make_tool_response([
                {"name": "finish", "arguments": {}},
            ]),
        ])

        state = run_agent_loop(
            task="Read f.txt",
            model=model,
            config=_config(tmp_path),
        )

        assert state.task == "Read f.txt"
        assert len(state.messages) >= 4  # system, user, assistant+tool, tool, assistant
        assert len(state.tool_calls) >= 1
        assert len(state.tool_results) >= 1
        assert state.repair_count == 0
        assert state.status == "success" # finish tool will succeed for INSPECTION task

# ═══════════════════════════════════════════════════════════════════════════
# Test: Trajectory Logging
# ═══════════════════════════════════════════════════════════════════════════

class TestTrajectoryLogging:
    def test_trajectory_creation_and_ordering(self, tmp_path: Path):
        import json
        
        # We need a small task with tool calls
        _write(tmp_path, "code.py", "x = 1")
        
        model = MockModel([
            _make_tool_response([{"name": "read_file", "arguments": {"path": "code.py"}}]),
            _make_tool_response([{"name": "write_patch", "arguments": {"path": "code.py", "old_str": "x = 1", "new_str": "x = 2"}}]),
            _make_tool_response([{"name": "run_tests", "arguments": {}}]),
            _make_tool_response([{"name": "finish", "arguments": {}}]),
        ])
        
        # create fake tests that succeed
        _write(tmp_path, "test_code.py", "def test_ok(): pass")
        
        state = run_agent_loop(
            task="Fix the code",
            model=model,
            config=_config(tmp_path),
        )
        
        assert state.status == "success"
        
        log_path = tmp_path / "trajectory.jsonl"
        assert log_path.exists()
        
        lines = log_path.read_text("utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines]
        
        # Expecting records for read_file, write_patch, run_tests, finish, and a terminal record
        assert len(records) == 5
        
        # Order should match exactly
        assert records[0]["tool_name"] == "read_file"
        assert records[1]["tool_name"] == "write_patch"
        assert records[2]["tool_name"] == "run_tests"
        assert records[3]["tool_name"] == "finish"
        assert records[4]["action"] == "terminate"
        
        # Run ID should be consistent
        run_id = records[0]["run_id"]
        for r in records:
            assert r["run_id"] == run_id
            
        # Step count should increase or stay same
        steps = [r["step"] for r in records]
        assert steps == sorted(steps)


# ═══════════════════════════════════════════════════════════════════════════
# Test: Task Intent & Enforcements
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskIntent:
    def test_inspection_task_rejects_modifications(self, tmp_path: Path):
        model = MockModel([
            _make_tool_response([{"name": "write_patch", "arguments": {"path": "req.txt", "old_str": "", "new_str": ""}}]),
            _make_tool_response([{"name": "run_tests", "arguments": {}}]),
            _make_tool_response([{"name": "finish", "arguments": {}}]),
        ])
        
        state = run_agent_loop(
            task="Inspect requirements.txt",
            model=model,
            config=_config(tmp_path),
        )
        
        # Tools rejected -> tool result is success=False, error mentions "not allowed"
        assert len(state.tool_results) == 3
        assert state.tool_results[0].tool_name == "write_patch"
        assert state.tool_results[0].success is False
        assert "not allowed for INSPECTION tasks" in state.tool_results[0].error
        
        assert state.tool_results[1].tool_name == "run_tests"
        assert state.tool_results[1].success is False
        assert "not allowed for INSPECTION tasks" in state.tool_results[1].error
        
        assert state.status == "success"

    def test_finish_terminates_immediately(self, tmp_path: Path):
        # We simulate a modification task that first modifies, then calls finish and run_tests in ONE response.
        # It should exit before run_tests is executed.
        _write(tmp_path, "code.py", "x = 1")
        
        model = MockModel([
            _make_tool_response([
                {"name": "write_patch", "arguments": {"path": "code.py", "old_str": "x = 1", "new_str": "x = 2"}},
                {"name": "run_tests", "arguments": {}},
                {"name": "finish", "arguments": {}},
                {"name": "read_file", "arguments": {"path": "code.py"}}
            ])
        ])
        
        state = run_agent_loop(
            task="Fix code.py",
            model=model,
            config=_config(tmp_path),
        )
        
        assert state.status == "success"
        # Only write_patch, run_tests, and finish should be in tool_results!
        tool_names = [res.tool_name for res in state.tool_results]
        assert tool_names == ["write_patch", "run_tests", "finish"]
        assert "read_file" not in tool_names

    def test_modification_triggers_repair(self, tmp_path: Path):
        _write(tmp_path, "code.py", "x = 1")
        _write(tmp_path, "test_code.py", "def test_fail(): assert False")
        
        model = MockModel([
            _make_tool_response([
                {"name": "write_patch", "arguments": {"path": "code.py", "old_str": "x = 1", "new_str": "x = 2"}},
                {"name": "run_tests", "arguments": {}}
            ]),
            _make_tool_response([{"name": "finish", "arguments": {}}])
        ])
        
        state = run_agent_loop(
            task="Fix code.py",
            model=model,
            config=_config(tmp_path),
        )
        
        assert state.repair_count == 1
        # Loop continues after test failure, model then calls finish (but wait, finish fails without verified tests or another mod...
        # Wait, if finish is called, it succeeds because a modification WAS attempted!
        assert state.status == "success"

# ═══════════════════════════════════════════════════════════════════════════
# Test: _dispatch_tool directly

# ═══════════════════════════════════════════════════════════════════════════


class TestDispatchTool:
    """Direct tests of the tool dispatcher."""

    def test_read_file(self, tmp_path: Path):
        _write(tmp_path, "a.txt", "content")
        result = _dispatch_tool(
            "read_file", {"path": "a.txt"},
            project_root=tmp_path, shell_timeout=10,
        )
        assert result.success is True
        assert result.output == "content"

    def test_unknown(self, tmp_path: Path):
        result = _dispatch_tool(
            "nope", {},
            project_root=tmp_path, shell_timeout=10,
        )
        assert result.success is False
        assert "unknown" in result.error.lower()

    def test_run_shell_missing_command(self, tmp_path: Path):
        result = _dispatch_tool(
            "run_shell", {},
            project_root=tmp_path, shell_timeout=10,
        )
        assert result.success is False
        assert "missing" in result.error.lower()


# ═══════════════════════════════════════════════════════════════════════════
# Test: _tool_result_to_content
# ═══════════════════════════════════════════════════════════════════════════


class TestToolResultFormatting:
    """Verify the observation string fed back to the model."""

    def test_success_format(self):
        r = ToolResult(tool_name="read_file", success=True, output="hello")
        s = _tool_result_to_content(r)
        assert "SUCCESS" in s
        assert "hello" in s

    def test_failure_format(self):
        r = ToolResult(
            tool_name="run_tests", success=False, output="1 failed",
            error="tests failed", exit_code=1, stderr="FAIL",
        )
        s = _tool_result_to_content(r)
        assert "FAILED" in s
        assert "1 failed" in s
        assert "Exit code: 1" in s
