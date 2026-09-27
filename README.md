# Local Coding Agent - Development Environment

This repository hosts the controller and development environment for the **Local Coding Agent** project.

## Architecture

The environment is structured as a split controller/remote-execution model:

```text
Antigravity (IDE)
    ↓
Local Git repository (Windows / Controller)
    ↓
Colab CLI (WSL2 Linux Environment)
    ↓
Colab GPU runtime (Remote Execution)
    ↓
Ollama (GPU-accelerated models)
    ↓
coding agent
```

## Component Status Checklist

| Component | Target Version / Role | Status | Notes |
| :--- | :--- | :--- | :--- |
| **Git** | Distributed Version Control | **WORKING** | `v2.51.2`, repository initialized on branch `main` |
| **Python** | 3.11+ (Host Controller) | **WORKING** | `Python 3.13.7` (64-bit) installed |
| **WSL2** | Linux Subsystem for Tooling | **NOT INSTALLED** | Windows Subsystem for Linux is not yet installed on the host |
| **Ollama** | Model Engine (Colab GPU) | **NOT READY** | Python client `ollama 0.6.1` is installed locally; Ollama engine will run remotely on Colab GPU |
| **Google Colab CLI** | Remote Compute Orchestration | **BLOCKED (NEEDS WSL2)** | `google-colab-cli 0.7.4` is installed, but requires POSIX (`termios`). Must execute inside WSL2 |

---

## Environment Verification

Run the verification checklist at any time:

```powershell
python verify_env.py
```

## Required Actions to Complete Environment Setup

1. **Install WSL2 on Windows Host**:
   Open PowerShell as **Administrator** and run:
   ```powershell
   wsl --install
   ```
   Reboot the system when prompted.

2. **Set up Google Colab CLI in WSL2**:
   Once WSL2 is running (e.g. Ubuntu):
   ```bash
   pip install google-colab-cli
   colab login
   ```

3. **Connect to Colab GPU Runtime & Ollama**:
   - Provision Colab GPU instance via Colab CLI.
   - Run and expose Ollama within the Colab GPU environment.
