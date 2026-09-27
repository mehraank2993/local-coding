# Project Rules: Local Coding Agent

## 1. Documentation Separation Rule
- **Setup & Infrastructure**: ALL instructions, runtime requirements, WSL configurations, and environment installation details MUST be documented exclusively in [`SETUP.md`](file:///d:/local-coding/SETUP.md).
- **Project Scope**: The root [`README.md`](file:///d:/local-coding/README.md) is reserved strictly for the core Local Coding Agent project, architecture, components, and usage.
- Never place installation walkthroughs, WSL setup steps, or environment troubleshooting directly into `README.md`.

## 2. Credential Security & Pre-Push Scanning Rule
- **Zero Credentials in Git**: Never commit, track, or push API keys, OAuth tokens, authorization codes, passwords, or secrets (e.g. `creds.txt`, `.env`, service account JSONs, or private keys).
- **Mandatory Scan**: Before staging, committing, or pushing any changes to Git, all files must pass the secret scanner:
  ```powershell
  python scripts/scan_secrets.py --all
  ```
- **Automated Enforcement**: Git hooks (`pre-commit` and `pre-push`) are enabled and will automatically abort any commit or push if secret patterns or unignored credential files are detected.
- **Git Ignore**: Any new files containing tokens or sensitive runtime configurations must immediately be added to [`.gitignore`](file:///d:/local-coding/.gitignore).
