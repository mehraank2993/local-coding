# Local Coding Agent

An autonomous coding agent system operated from a local development environment, powered by remote GPU compute and local LLM backends (Ollama / Qwen2.5-Coder on Google Colab).

---

## Overview

The **Local Coding Agent** project separates the development/controller surface from heavy model execution:
- **Local Controller**: Windows + Antigravity IDE managing task orchestration, codebase modifications, and Git version control.
- **Compute Plane**: Google Colab GPU runtimes provisioned on-demand via the Colab CLI (WSL2), hosting Ollama and open-weights coding models.

For all environment installation, WSL2 configuration, and Colab CLI setup details, refer to the **[Setup Guide](file:///d:/local-coding/SETUP.md)**.

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
| **[`verify_env.py`](file:///d:/local-coding/verify_env.py)** | Script to verify Git, Python, WSL2, Ollama, and Colab CLI |
| **[`gpu_test.py`](file:///d:/local-coding/gpu_test.py)** | Remote Colab GPU validation script |
| **[`scripts/scan_secrets.py`](file:///d:/local-coding/scripts/scan_secrets.py)** | Pre-commit and pre-push secret scanning tool |
