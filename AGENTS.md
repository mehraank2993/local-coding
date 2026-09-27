# Agent Operating Rules & Constraints

## Rule 1: Setup Documentation Boundary
- **`SETUP.md`**: All environment setup, infrastructure configuration, WSL2 instructions, Colab CLI commands, and prerequisite installations belong exclusively in [`SETUP.md`](file:///d:/local-coding/SETUP.md).
- **`README.md`**: Reserved solely for the Local Coding Agent project overview, architecture, and core agent capabilities.

## Rule 2: Credential & Secret Protection
- **No Secrets in Repository**: Never commit API keys, authorization codes, tokens, or credential files.
- **Pre-Push & Pre-Commit Scanning**: Always run or verify against [`scripts/scan_secrets.py`](file:///d:/local-coding/scripts/scan_secrets.py) before committing and pushing code:
  ```powershell
  python scripts/scan_secrets.py --all
  ```
- **Automated Git Hooks**: Active hooks in `.git/hooks/` automatically reject any commit or push containing secrets.
- **Git Ignore**: Credential files (e.g. `creds.txt`, `.env`, `*secret*`) must remain excluded in [`.gitignore`](file:///d:/local-coding/.gitignore).
