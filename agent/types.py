"""Small typed structures used throughout the agent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Default values sourced from environment, never hard-coded credentials.
# ---------------------------------------------------------------------------

_DEFAULT_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL: str = os.getenv("AGENT_MODEL", "qwen2.5-coder:3b")


@dataclass
class AgentConfig:
    """Configuration for the coding agent.

    All connection details are read from environment variables so nothing
    is hard-coded:

    ============  ==============================  ==========================
    Env var       Purpose                         Default
    ============  ==============================  ==========================
    OLLAMA_HOST   Ollama server URL               ``http://localhost:11434``
    AGENT_MODEL   Model name for completions      ``qwen2.5-coder:3b``
    ============  ==============================  ==========================
    """

    model: str = field(default_factory=lambda: _DEFAULT_MODEL)
    host: str = field(default_factory=lambda: _DEFAULT_HOST)
    max_iterations: int = 10
    temperature: float = 0.2
    project_root: Path = field(default_factory=lambda: Path.cwd())
    shell_timeout: int = 60  # seconds


@dataclass
class ToolResult:
    """Structured result returned by every tool invocation.

    Fields
    ------
    tool_name:
        Which tool produced this result.
    success:
        ``True`` when the operation completed without error.
    output:
        Primary textual output (file contents, stdout, etc.).
    error:
        Human-readable error description, or ``None`` on success.
    exit_code:
        Process exit code for shell / test tools; ``None`` for file tools.
    stderr:
        Captured stderr for shell / test tools; ``None`` for file tools.
    duration:
        Wall-clock seconds the operation took; ``None`` when not measured.
    command:
        The command string that was executed; ``None`` for file tools.
    """

    tool_name: str
    success: bool
    output: str
    error: str | None = None
    exit_code: int | None = None
    stderr: str | None = None
    duration: float | None = None
    command: str | None = None


@dataclass
class Message:
    """A single message in the agent conversation."""

    role: str  # "system", "user", or "assistant"
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
