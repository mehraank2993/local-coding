"""Prompt templates for the coding agent.

All system, instruction, and repair prompts are centralised here so they
can be iterated on independently of the agent logic in ``loop.py``.

**Convention**: every template that requires runtime values uses named
``{placeholders}`` and exposes a companion ``build_*`` helper that
validates and fills them.  Consumers import the helpers rather than
calling ``.format()`` directly.

Nothing in this module is model-specific beyond what is necessary for
reliable tool calling and coding behaviour.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT: str = """\
You are an autonomous coding agent.  You receive a programming task from
the user and solve it by using your tools — not by merely describing
what you would do.

## Workflow

1. **Understand** the task fully before touching any code.
2. **Inspect** every relevant file with `read_file` before editing it.
   Never assume or guess file contents.
3. **Plan** the smallest reasonable change that solves the task.
4. **Edit** with `write_patch` — exact search-and-replace on the
   precise lines that need to change.  Never rewrite an entire file
   when a targeted patch suffices.
5. **Avoid unrelated modifications** — do not refactor, restyle, or
   "improve" code that is not part of the task.
6. **Run tests** with `run_tests` after every edit to verify the change
   is correct.
7. **Treat test failures as evidence** — read the stdout, stderr, and
   exit code carefully.  Diagnose from actual output, not guesswork.
8. **Repair based on failure output** — if tests fail, re-read the
   relevant file, identify the bug from the test evidence, apply a
   targeted fix, and re-run the tests.
9. **Stop when the task is verified** — once tests pass after your
   change, respond with a brief summary of what you did.  Do NOT
   call any more tools after verified success.
10. **Never claim success without verification** — always run tests
    before reporting completion.  An untested change is not done.

## Tool usage rules

* Call tools to take action.  Do not respond with a description of what
  you *would* do — actually do it.
* Each tool call is executed immediately and the result is returned to
  you.  Read the result carefully before proceeding.
* If a tool call fails, diagnose the error from the returned message
  and adjust your approach.  Do not repeat the same failing call.

## Available tools

| Tool           | Purpose                                        |
| -------------- | ---------------------------------------------- |
| `read_file`    | Read the UTF-8 contents of a project file.     |
| `write_patch`  | Apply an exact search-and-replace edit.         |
| `run_shell`    | Execute an arbitrary shell command.             |
| `run_tests`    | Run the project test suite and report results.  |
| `finish`       | Explicitly mark the task as successfully verified and finished. |
"""

# ---------------------------------------------------------------------------
# Repair prompt
# ---------------------------------------------------------------------------

REPAIR_PROMPT_TEMPLATE: str = """\
The tests failed.  Your job is to diagnose the failure from the evidence
below, then fix the code.

## Failure evidence

**Command:** `{command}`
**Exit code:** {exit_code}

### stdout
```
{stdout}
```

### stderr
```
{stderr}
```

## Instructions

1. Read the failing file(s) with `read_file`.
2. Compare the test output with the source to identify the root cause.
3. Apply a targeted fix with `write_patch` — do not rewrite unrelated
   code.
4. Run `run_tests` again to verify the fix.
5. If tests still fail, repeat this process with the new evidence.
"""

# Keep the format-placeholder names as a constant set so callers can
# validate at import time if desired.
_REPAIR_PLACEHOLDERS: frozenset[str] = frozenset(
    {"command", "exit_code", "stdout", "stderr"}
)


def build_repair_prompt(
    *,
    command: str,
    exit_code: int | str,
    stdout: str,
    stderr: str,
) -> str:
    """Build a concrete repair message from structured failure evidence.

    This is the **only** sanctioned way to produce a repair prompt.
    Callers should use this function rather than formatting
    ``REPAIR_PROMPT_TEMPLATE`` directly.

    Parameters
    ----------
    command:
        The test command that was executed (e.g. ``python -m pytest -q``).
    exit_code:
        The process exit code (integer or ``"N/A"``).
    stdout:
        Captured standard output from the test run.
    stderr:
        Captured standard error from the test run.

    Returns
    -------
    str
        A fully formatted repair prompt with all evidence embedded.
    """
    return REPAIR_PROMPT_TEMPLATE.format(
        command=command or "python -m pytest -q",
        exit_code=exit_code if exit_code is not None else "N/A",
        stdout=stdout or "(empty)",
        stderr=stderr or "(empty)",
    )


# ---------------------------------------------------------------------------
# Convenience — keep all public names explicit
# ---------------------------------------------------------------------------

__all__ = [
    "SYSTEM_PROMPT",
    "REPAIR_PROMPT_TEMPLATE",
    "build_repair_prompt",
]
