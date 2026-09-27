"""Bootstrap a Google Colab GPU runtime for the Local Coding Agent.

This script provisions an ephemeral Colab T4 GPU instance using the Colab CLI,
installs project dependencies, and runs a remote verification check.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def _run_colab_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a colab CLI command inside WSL."""
    cmd = ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "/root/.local/bin/colab"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def _check_auth():
    """Verify Colab CLI authentication."""
    print("Checking Colab CLI authentication...")
    result = _run_colab_cmd(["usage"], check=False)
    if result.returncode != 0:
        print("Error: Colab CLI authentication failed or not configured.", file=sys.stderr)
        print("Please run 'colab auth' in WSL to authenticate.", file=sys.stderr)
        sys.exit(1)


def bootstrap_colab(session_name: str):
    """Run the complete bootstrap sequence."""
    _check_auth()

    print(f"Creating new Colab session '{session_name}' with T4 GPU...")
    result = _run_colab_cmd(["new", "-s", session_name, "--gpu", "T4"], check=False)
    if result.returncode != 0:
        print("Error: Failed to allocate GPU session.", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print("Checking session status...")
    # Wait for the session to be ready (up to 5 minutes)
    ready = False
    for i in range(60):
        res = _run_colab_cmd(["status", "-s", session_name], check=False)
        if res.returncode == 0:
            status = res.stdout.strip().lower()
            if "ready" in status or "idle" in status:
                ready = True
                break
        time.sleep(5)
    
    if not ready:
        print("Error: Session did not become ready.", file=sys.stderr)
        sys.exit(1)

    print("Installing project dependencies...")
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("Error: requirements.txt not found locally.", file=sys.stderr)
        sys.exit(1)
    
    res = _run_colab_cmd(["install", "-s", session_name, "-r", "requirements.txt"], check=False)
    if res.returncode != 0:
        print("Error: Dependency installation failed.", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(1)

    print("Running GPU verification script...")
    verify_script = """import sys
import platform
try:
    import torch
    has_torch = True
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
    torch_version = torch.__version__
except ImportError:
    has_torch = False
    cuda_available = False
    gpu_name = "None"
    torch_version = "Not installed"

print(f"Python version: {sys.version.split()[0]}")
print(f"CUDA availability: {cuda_available}")
print(f"GPU name: {gpu_name}")
print(f"PyTorch version: {torch_version}")
"""
    verify_path = Path("remote_verify.py")
    verify_path.write_text(verify_script, encoding="utf-8")
    
    res = _run_colab_cmd(["exec", "-s", session_name, "-f", "remote_verify.py"], check=False)
    verify_path.unlink(missing_ok=True)
    
    if res.returncode != 0:
        print("Error: Remote execution failed.", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(1)
    
    print("\n--- Remote GPU Verification Result ---")
    print(res.stdout.strip())
    print("--------------------------------------\n")

    print(f"Bootstrap complete. Session '{session_name}' is active and ready.")
    print(f"To cleanup, run: colab stop -s {session_name}  (or use 'wsl -d Ubuntu-24.04 -u root /root/.local/bin/colab stop -s {session_name}')")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap Colab GPU session.")
    parser.add_argument("-s", "--session", required=True, help="Session name to create/use")
    args = parser.parse_args()
    bootstrap_colab(args.session)


if __name__ == "__main__":
    main()
