# Environment & Infrastructure Setup Guide

This document contains all details for configuring, verifying, and maintaining the development and remote-compute environment for the **Local Coding Agent** project.

> [!IMPORTANT]
> **Project Rule**: Any future infrastructure, environment dependencies, WSL configurations, or compute runtime updates **must be documented in this file**, keeping `README.md` reserved strictly for project architecture and agent design.

---

## Architecture Overview

```text
Windows Host
└── Antigravity (IDE)
    └── D:\local-coding (Local Controller & Git Repository)
        └── WSL2 (Ubuntu 24.04 LTS / Linux Tooling)
            └── Google Colab CLI (v0.7.4 via uv)
                ▼
            Google Colab VM
                ├── GPU (Tesla T4 / A100)
                ├── Ollama (Inference Backend)
                ├── Qwen2.5-Coder (Model)
                └── Coding Agent Execution
```

---

## Current Status Checklist

| Component | Target Version / Role | Status | Notes |
| :--- | :--- | :---: | :--- |
| **Git** | Host Version Control | **PASS** | `v2.51.2.windows.1`, initialized on branch `main` |
| **Host Python** | 3.11+ Controller | **PASS** | `Python 3.13.7` (Windows host) |
| **WSL2** | Linux Execution Subsystem | **PASS** | `Ubuntu-24.04 LTS` (Python 3.12.3) |
| **Colab CLI** | Remote VM Orchestrator | **PASS** | `google-colab-cli 0.7.4` via `uv` in WSL |
| **Remote GPU** | Cloud Compute | **PASS** | Ephemeral Tesla T4 tested & verified |
| **Ollama** | Model Engine | **PASS** | Runs inside remote Colab GPU VM |

---

## Setup & Verification Instructions

### 1. Environment Verification Script
Run the automated multi-component verification check from the Windows host:
```powershell
python verify_env.py
```

### 2. WSL2 Linux Environment
* Distribution: `Ubuntu-24.04 LTS`
* Default Python: `Python 3.12.3`
* System packages installed: `git`, `python3`, `python3-pip`, `python3-venv`, `curl`

To access the environment from PowerShell:
```powershell
wsl -d Ubuntu-24.04
```
From within WSL, the project directory is located at:
```bash
cd /mnt/d/local-coding
```

### 3. Google Colab CLI (`uv` Tool)
Installed inside WSL2 via Astral's `uv`:
```bash
# Path to executable: /root/.local/bin/colab
colab version
colab usage
colab --help
colab auth --help
```

### 4. Remote GPU Smoke Test
To test provisioning, remote script upload, and teardown of an ephemeral GPU runtime:
```bash
# Run inside WSL
cd /mnt/d/local-coding
/root/.local/bin/colab run --gpu T4 gpu_test.py
```
Expected output:
```text
[colab] Creating session 'run-...'...
[colab] Session READY. Executing gpu_test.py...
CUDA available: True
GPU: Tesla T4
[colab] Stopping session...
[colab] Session terminated.
```
