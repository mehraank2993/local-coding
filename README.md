# Local Coding Agent - Development Environment

This repository hosts the controller and development environment for the **Local Coding Agent** project.

## Architecture

```text
Windows Host
└── Antigravity (IDE)
    └── D:\local-coding (Local Git Repository)
        └── WSL2 / Colab CLI (Linux Execution Environment)
            ▼
        Google Colab VM
            ├── GPU (T4 / A100)
            ├── Ollama
            ├── Qwen2.5-Coder
            └── Coding Agent
```

---

## Infrastructure Milestone: WSL2 → Colab CLI → T4 GPU

The local machine has no discrete GPU and acts strictly as the controller. Because `google-colab-cli` requires POSIX terminal support (`termios`) and native Windows support is still upstream work-in-progress, **WSL2** is the designated host for the Colab CLI.

Before building any agent logic or execution abstractions, complete this 4-step infrastructure milestone to prove the raw remote execution path.

### Step 1: Install WSL2 on Windows Host
Open PowerShell as **Administrator**:
```powershell
wsl --install
```
Restart Windows when prompted. Once rebooted, verify:
```powershell
wsl --status
wsl -l -v
```
*(Ensure Ubuntu or your target distribution is running under WSL version 2).*

### Step 2: Set Up WSL2 Environment
Inside your WSL terminal (e.g., Ubuntu):
```bash
sudo apt update && sudo apt install -y git python3 python3-pip python3-venv curl
```
*(Use Python 3.11 or 3.12 inside WSL to ensure maximum stability and compatibility).*

### Step 3: Install Colab CLI via `uv` in WSL
Install Astral's `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

Install `google-colab-cli` via `uv tool`:
```bash
uv tool install google-colab-cli
```

Verify installation and inspect supported authentication / flags:
```bash
colab version
colab --help
colab auth --help
```

### Step 4: Validate Raw Remote GPU Execution
From your repository root inside WSL (`/mnt/d/local-coding`), run:
```bash
colab run --gpu T4 gpu_test.py
```

`gpu_test.py` validates remote CUDA availability and device reporting:
```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

> [!NOTE]
> **No Premature Abstractions**: Do not build a custom `ColabExecutor` wrapper yet. First prove the raw CLI pathway (`colab run --gpu T4`). Once raw execution is validated, we proceed to Phase 1.
