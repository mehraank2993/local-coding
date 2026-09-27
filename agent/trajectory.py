import json
import uuid
import time
from pathlib import Path
from typing import Any

def log_trajectory_step(
    log_path: Path,
    run_id: str,
    task: str,
    model: str,
    runtime: str,
    step: int,
    action: str,
    tool_name: str | None,
    tool_arguments: dict[str, Any] | None,
    tool_result: dict[str, Any] | None,
    verification_result: dict[str, Any] | None,
    repair_attempt: int,
    final_status: str | None,
):
    """Append a structured step to the JSONL trajectory file."""
    record = {
        "run_id": run_id,
        "task": task,
        "model": model,
        "runtime": runtime,
        "step": step,
        "action": action,
        "tool_name": tool_name,
        "tool_arguments": tool_arguments,
        "tool_result": tool_result,
        "verification_result": verification_result,
        "repair_attempt": repair_attempt,
        "final_status": final_status,
        "timestamp": time.time(),
    }
    
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
