import pytest
import shlex
from scripts.run_remote_agent import build_remote_script

def test_remote_script_preserves_task_argument():
    # User's task with spaces, punctuation, and quotes
    task = "Fix calculate_total() so empty lists return 0"
    
    # How run_remote_agent handles unknown_args
    unknown_args = [task, "--repair"]
    remote_args_str = " ".join(shlex.quote(arg) for arg in unknown_args)
    
    script = build_remote_script(remote_args_str)
    
    # We expect the exact task string to exist in the generated script
    assert "Fix calculate_total() so empty lists return 0" in script
    
    # Simulate what happens inside the remote script when it parses it back
    import ast
    
    # Find the line that declares remote_args = shlex.split(...)
    # We can parse the ast to find it or just extract it
    lines = script.splitlines()
    shlex_line = next(line for line in lines if line.startswith("remote_args = shlex.split("))
    
    # Extract the literal string that is passed to shlex.split
    # it looks like: remote_args = shlex.split("'Fix calculate_total() so empty lists return 0' --repair")
    # let's just evaluate it
    # We can use a local dict to execute just that line safely
    local_vars = {"shlex": shlex}
    exec(shlex_line, {}, local_vars)
    
    parsed_args = local_vars["remote_args"]
    assert parsed_args == [task, "--repair"]
