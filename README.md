# Local Coding Agent

An autonomous coding agent system operated from a local development environment, powered by remote GPU compute and local LLM backends (Ollama / Qwen2.5-Coder on Google Colab).

---

## Overview

The **Local Coding Agent** project separates the development/controller surface from heavy model execution:
- **Local Controller**: Windows + Antigravity IDE managing task orchestration, codebase modifications, and Git version control.
- **Compute Plane**: Google Colab GPU runtimes provisioned on-demand via the Colab CLI (WSL2), hosting Ollama and open-weights coding models.

For all environment installation, WSL2 configuration, and Colab CLI setup details, refer to the **[Setup Guide](file:///d:/local-coding/SETUP.md)**.

## Hardened Agent Loop

The agent loop has been hardened to prevent common LLM pitfalls:
- **Task Intent Classification**: The agent strictly distinguishes between `INSPECTION` (read-only) and `MODIFICATION` tasks, physically blocking destructive tools (like `write_patch` or `run_tests`) when merely asked to inspect a file.
- **Truthful Completion**: A natural-language response alone never constitutes completion. The `finish` tool requires concrete verification (`run_tests` or `run_shell`) after any code modification.
- **Strict Repair Scoping**: The autonomous repair loop is only triggered if the agent actually attempted modifications, preventing pre-existing environment or test failures from hijacking the task objective.
- **Test Isolation**: Test tools support targeted test scopes to prevent unintentional broad test suite executions.

---

## Architecture

```text
Windows Host (Antigravity IDE)
    └── Local Git Repository (d:\local-coding)
            └── WSL2 / Colab CLI
                    ▼
            Google Colab GPU VM
                ├── Tesla T4 / A100 GPU
                ├── Ollama Server
                ├── Qwen2.5-Coder
                └── Autonomous Agent Engine
```

---

## Development Guidelines & Rules

1. **Setup Documentation**: All setup instructions, environment dependencies, and runtime configurations must reside in [`SETUP.md`](file:///d:/local-coding/SETUP.md). Do not clutter this main README with installation steps.
2. **Security & Credential Scanning**: 
   - No credentials, tokens, or API keys (`creds.txt`, `.env`, OAuth tokens) may ever be committed or pushed.
   - All commits and pushes are scanned by [`scripts/scan_secrets.py`](file:///d:/local-coding/scripts/scan_secrets.py).
3. **No Heavy Premature Dependencies**: Avoid adding unneeded framework abstractions (no LangChain, LangGraph, Docker, MCP, FastAPI, RAG, DBs, frontend). Keep the agent lean and purpose-built.

---

## Quick Reference

| Resource | Description |
| :--- | :--- |
| **[`SETUP.md`](file:///d:/local-coding/SETUP.md)** | Infrastructure setup, WSL2 configuration, and Colab CLI guide |
| **[`scripts/run_remote_agent.py`](file:///d:/local-coding/scripts/run_remote_agent.py)** | Script to upload the codebase and run the agent remotely via an existing Colab session |
| **[`scripts/setup_remote_ollama.py`](file:///d:/local-coding/scripts/setup_remote_ollama.py)** | Automatically install and boot Ollama + `qwen2.5-coder:7b` inside the remote Colab session |
| **[`verify_env.py`](file:///d:/local-coding/verify_env.py)** | Script to verify Git, Python, WSL2, Ollama, and Colab CLI |
| **[`gpu_test.py`](file:///d:/local-coding/gpu_test.py)** | Remote Colab GPU validation script |
| **[`scripts/scan_secrets.py`](file:///d:/local-coding/scripts/scan_secrets.py)** | Pre-commit and pre-push secret scanning tool |
