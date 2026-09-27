import argparse
import subprocess
import sys
import shlex
from pathlib import Path


def _run_colab_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a colab CLI command inside WSL."""
    cmd = ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "bash", "-c",
           f"cd /mnt/d/local-coding && /root/.local/bin/colab {' '.join(shlex.quote(a) for a in args)}"]
    return subprocess.run(cmd, capture_output=True, text=True, check=check, encoding="utf-8")

def build_remote_script(remote_args_str: str) -> str:
    return f"""import os
import sys
import subprocess
import logging

# We don't want debug logging for the final E2E test to keep output clean, unless needed.
# logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

print("Extracting repository...")
subprocess.run("mkdir -p /content/local-coding && tar -xzf /content/repo.tar.gz -C /content/local-coding", shell=True, check=True)

print("Starting Ollama (just in case)...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import time
import urllib.request
for _ in range(10):
    try:
        urllib.request.urlopen("http://localhost:11434/", timeout=1)
        break
    except Exception:
        time.sleep(1)

os.chdir("/content/local-coding")
sys.path.insert(0, "/content/local-coding")

# We must set this env var so the agent uses 7b model
os.environ["AGENT_MODEL"] = "qwen2.5-coder:7b"

import main
import shlex

remote_args = shlex.split({repr(remote_args_str)})
print(f"\\n>>> RUNNING AGENT WITH ARGS: {{remote_args}} <<<\\n")

sys.exit(main.main(remote_args))
"""

def main():
    # Pass all unknown arguments to the remote main.py
    parser = argparse.ArgumentParser(description="Run agent in Colab.", add_help=False)
    parser.add_argument("-s", "--session", required=True, help="Session name")
    args, unknown_args = parser.parse_known_args()

    print(f"Creating archive of the current codebase...")
    subprocess.run([
        "wsl", "-d", "Ubuntu-24.04", "-u", "root", "bash", "-c",
        'cd /mnt/d/local-coding && tar --exclude=".git" --exclude="__pycache__" --exclude=".pytest_cache" -czf /tmp/repo.tar.gz .'
    ], check=True)

    print("Uploading archive to Colab...")
    _run_colab_cmd(["upload", "-s", args.session, "/tmp/repo.tar.gz", "/content/repo.tar.gz"], check=True)

    # Reconstruct the arguments for main.py
    remote_args_str = " ".join(shlex.quote(arg) for arg in unknown_args)
    remote_script = build_remote_script(remote_args_str)

    remote_path = Path("remote_agent_runner.py")
    remote_path.write_text(remote_script, encoding="utf-8")

    print("Restarting remote kernel to ensure fresh imports...")
    _run_colab_cmd(["restart-kernel", "-s", args.session], check=False)

    print(f"Executing agent remotely...")
    res = _run_colab_cmd(["exec", "--timeout", "900", "-s", args.session, "-f", "remote_agent_runner.py"], check=False)
    
    remote_path.unlink(missing_ok=True)
    
    print("\\n--- Remote Output ---")
    
    def safe_print(text):
        print(text.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8'))
        
    safe_print(res.stdout.strip())
    if res.stderr:
        print("\\n--- Remote Stderr ---")
        safe_print(res.stderr.strip())
    print("---------------------\\n")
    
    if res.returncode != 0:
        print("Error: Remote agent execution failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
