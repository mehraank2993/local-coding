"""Entry point for the Local Coding Agent.

Usage::

    python main.py "Fix the bug in calculate_total()"
    python main.py --model qwen2.5-coder:7b --max-steps 30 "Add logging"
    python main.py --no-repair --repo ./my-project "Refactor utils.py"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent.loop import AgentState, run_agent_loop
from agent.model import OllamaModel
from agent.types import AgentConfig, ToolResult


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser."""
    parser = argparse.ArgumentParser(
        prog="local-coding-agent",
        description="Autonomous coding agent powered by Ollama.",
    )
    parser.add_argument(
        "task",
        help="The coding task to perform (e.g. 'Fix the bug in calculate_total()').",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Ollama model name (default: AGENT_MODEL env var or "
            "'qwen2.5-coder:3b')."
        ),
    )
    parser.add_argument(
        "--repair",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable the test-repair loop (default: enabled).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum model-call steps (default: 20).",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=3,
        help="Maximum repair attempts on test failure (default: 3).",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Project repository root (default: current directory).",
    )
    return parser


# ---------------------------------------------------------------------------
# Progress printing
# ---------------------------------------------------------------------------

_last_repair_count: int = 0


def _on_tool_result(
    step: int,
    tool_name: str,
    result: ToolResult,
    repair_count: int,
) -> None:
    """Callback invoked by the agent loop after each tool execution."""
    global _last_repair_count

    # Print a repair header when a new repair cycle starts
    if repair_count > _last_repair_count:
        _last_repair_count = repair_count
        print(f"\nRepair {repair_count}\n")

    status = "success" if result.success else "failed"
    print(f"  Step {step}")
    print(f"  Tool: {tool_name}")
    print(f"  Status: {status}")
    print()


def _print_summary(state: AgentState) -> None:
    """Print the final execution summary."""
    label = state.status.upper().replace("_", " ")
    # Map internal status to a user-friendly display
    if state.status == "success":
        display = "SUCCESS"
    elif state.status == "max_repairs":
        display = "FAILED (repair limit)"
    elif state.status == "max_steps":
        display = "FAILED (step limit)"
    elif state.status == "error":
        display = "ERROR"
    else:
        display = label

    print(f"Final status: {display}")
    print(f"Steps: {state.step_count}")
    print(f"Repairs: {state.repair_count}")

    if state.error:
        print(f"Error: {state.error}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Parse arguments, construct the agent, and run the loop.

    Parameters
    ----------
    argv:
        Command-line arguments.  Defaults to ``sys.argv[1:]`` when
        ``None`` (production).  Pass explicitly for testing.

    Returns
    -------
    int
        ``0`` on success, ``1`` on agent failure.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Validate repo path ─────────────────────────────────────────────
    repo = Path(args.repo).resolve()
    if not repo.is_dir():
        print(f"Error: repository path does not exist: {repo}", file=sys.stderr)
        return 1

    # ── Build configuration ────────────────────────────────────────────
    config_kwargs: dict = {
        "project_root": repo,
    }
    if args.model is not None:
        config_kwargs["model"] = args.model

    config = AgentConfig(**config_kwargs)

    # ── Print header ───────────────────────────────────────────────────
    print(f"Task: {args.task}")
    print(f"Model: {config.model}")
    print(f"Repo: {repo}")
    print()

    # ── Construct model ────────────────────────────────────────────────
    try:
        model = OllamaModel(config=config)
    except Exception as exc:
        print(f"Error: cannot create model client: {exc}", file=sys.stderr)
        return 1

    # ── Run the agent loop ─────────────────────────────────────────────
    global _last_repair_count
    _last_repair_count = 0

    try:
        state = run_agent_loop(
            task=args.task,
            model=model,
            config=config,
            max_steps=args.max_steps,
            max_repairs=args.max_repairs,
            repair_enabled=args.repair,
            on_tool_result=_on_tool_result,
        )
    except Exception as exc:
        print(f"Error: agent loop failed: {exc}", file=sys.stderr)
        return 1

    # ── Print summary ──────────────────────────────────────────────────
    _print_summary(state)

    return 0 if state.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
