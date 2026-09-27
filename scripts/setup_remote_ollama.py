"""Configure Ollama inside the remote Colab GPU session."""

import argparse
import subprocess
import sys
from pathlib import Path

# The remote python script that runs inside Colab
REMOTE_SCRIPT = """\
import os
import subprocess
import time
import sys
import json
import urllib.request

try:
    import torch
    has_torch = True
    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else "None"
except ImportError:
    has_torch = False
    cuda_available = False
    gpu_name = "None"

print(f"GPU name: {gpu_name}")
print(f"CUDA availability: {cuda_available}")

# 1. Verify/Install Ollama
try:
    res = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
    ollama_version = res.stdout.strip()
    print(f"Ollama already installed: {ollama_version}")
except (FileNotFoundError, subprocess.CalledProcessError):
    print("Ollama not found. Installing...")
    print("Installing zstd prerequisite...")
    subprocess.run(["apt-get", "update"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["apt-get", "install", "-y", "zstd"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)
    res = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=True)
    ollama_version = res.stdout.strip()
    print(f"Ollama installed: {ollama_version}")

# 3. Start server
print("Starting Ollama server...")
ollama_proc = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# 4. Wait for API
print("Waiting for Ollama HTTP API...")
api_ready = False
for _ in range(60):
    try:
        urllib.request.urlopen("http://localhost:11434/", timeout=1)
        api_ready = True
        break
    except Exception:
        time.sleep(1)

if not api_ready:
    print("Error: Ollama API failed to start.")
    sys.exit(1)
print("Ollama API is reachable.")

# 5. Pull model
model_name = "qwen2.5-coder:7b"
print(f"Pulling model {model_name} (this may take a few minutes)...")
subprocess.run(["ollama", "pull", model_name], check=True)

# 6. Verify model exists
print("Verifying model exists...")
import ollama
models = ollama.list()
model_names = [m.get("name") or m.get("model") for m in models.get("models", [])]
print(f"Available models: {model_names}")
if not any(model_name in n for n in model_names):
    print(f"Error: {model_name} not found in model list.")
    sys.exit(1)

# 7. Live inference request
print("Running live inference...")
resp = ollama.chat(model=model_name, messages=[{"role": "user", "content": "Say hello!"}])
print("Inference response:", resp["message"]["content"])

# 8. Tool calling test
print("Running native tool-calling test...")
def add(a: int, b: int) -> int:
    \"\"\"Add two integers.\"\"\"
    return a + b

resp = ollama.chat(
    model=model_name,
    messages=[{"role": "user", "content": "Call the add tool with a=20 and b=22. Do NOT answer the question directly. You MUST call the tool exactly as provided."}],
    tools=[add]
)

message = getattr(resp, "message", None) or resp.get("message", {})
tool_calls = getattr(message, "tool_calls", None) or message.get("tool_calls", [])
content = getattr(message, "content", "") or message.get("content", "")

if not tool_calls:
    try:
        # Fallback to parsing from content
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "name" in parsed:
            tool_calls = [{"function": parsed}]
    except json.JSONDecodeError:
        pass

print("Tool calls returned:", json.dumps(tool_calls, indent=2))

if not tool_calls:
    print("Error: Model did not return any tool calls.")
    print("Raw message content was:", content)
    sys.exit(1)

tc = tool_calls[0]
name = tc.get("function", {}).get("name")
args = tc.get("function", {}).get("arguments", {})

if name != "add":
    print(f"Error: Wrong tool name: {name}")
    sys.exit(1)

if "a" not in args or "b" not in args:
    print(f"Error: Missing arguments: {args}")
    sys.exit(1)

print("Tool calling test PASSED.")
"""

def _run_colab_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a colab CLI command inside WSL."""
    cmd = ["wsl", "-d", "Ubuntu-24.04", "-u", "root", "/root/.local/bin/colab"] + args
    return subprocess.run(cmd, capture_output=True, text=True, check=check)

def main():
    parser = argparse.ArgumentParser(description="Configure Ollama inside Colab.")
    parser.add_argument("-s", "--session", required=True, help="Session name")
    args = parser.parse_args()

    verify_path = Path("remote_ollama_setup.py")
    verify_path.write_text(REMOTE_SCRIPT, encoding="utf-8")
    
    print(f"Executing Ollama setup on remote Colab session '{args.session}'...")
    res = _run_colab_cmd(["exec", "--timeout", "600", "-s", args.session, "-f", "remote_ollama_setup.py"], check=False)
    
    verify_path.unlink(missing_ok=True)
    
    if res.returncode != 0:
        print("Error: Remote execution failed.", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(1)
        
    print("\n--- Remote Output ---")
    print(res.stdout.strip())
    print("---------------------\n")


if __name__ == "__main__":
    main()
