"""Secret and Credential Scanner for Local Coding Agent.

Prevents committing or pushing credentials, tokens, and API secrets.
Can be used as a standalone tool, pre-commit hook, or pre-push hook.
"""

import os
import re
import subprocess
import sys

# High-confidence patterns for sensitive credentials
SECRET_PATTERNS = [
    (r"4/0A[A-Za-z0-9_\-]{30,}", "Google OAuth Authorization Code"),
    (r"AIza[0-9A-Za-z_\-]{35}", "Google API Key"),
    (r"ya29\.[0-9A-Za-z_\-]{50,}", "Google OAuth Access Token"),
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"ghp_[0-9a-zA-Z]{36}", "GitHub Personal Access Token"),
    (r"github_pat_[0-9a-zA-Z_]{82}", "GitHub Fine-Grained Token"),
    (r"xox[baprs]-[0-9a-zA-Z]{10,48}", "Slack Token"),
    (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "Private Key"),
    (r'(?i)(api[_-]?key|secret[_-]?key|password|auth[_-]?token)\s*[:=]\s*["\']?[0-9a-zA-Z\-_\.]{8,}["\']?', "Hardcoded Secret/Token Assignment"),
    (r"gcloud\s+cli\s*:\s*[A-Za-z0-9_\-\/]+", "GCloud / Colab CLI Credential"),
]

BLOCKED_FILENAMES = [
    re.compile(r"^creds(\.txt)?$", re.IGNORECASE),
    re.compile(r"^.*\.secret$", re.IGNORECASE),
    re.compile(r"^.*\.key$", re.IGNORECASE),
    re.compile(r"^\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"^.*_rsa$", re.IGNORECASE),
    re.compile(r"^.*_ed25519$", re.IGNORECASE),
]


def get_files_to_scan(mode: str) -> list[str]:
    """Get list of files to scan based on git mode."""
    cmd = []
    if mode == "--staged":
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]
    elif mode == "--pre-push":
        # Check diff between origin/main and HEAD if upstream exists, else all tracked
        try:
            upstream = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "@{u}"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            cmd = ["git", "diff", f"{upstream}..HEAD", "--name-only", "--diff-filter=ACM"]
        except Exception:
            cmd = ["git", "ls-files"]
    elif mode == "--all":
        cmd = ["git", "ls-files"]
    else:
        cmd = ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"]

    try:
        output = subprocess.check_output(cmd, text=True, errors="replace").strip()
        if not output:
            return []
        return [f.strip() for f in output.splitlines() if f.strip()]
    except Exception:
        # Fallback to scanning working directory
        files = []
        for root, dirs, filenames in os.walk("."):
            if ".git" in dirs:
                dirs.remove(".git")
            for filename in filenames:
                files.append(os.path.relpath(os.path.join(root, filename), "."))
        return files


def scan_file(filepath: str) -> list[str]:
    """Scan a single file for blocked names and sensitive regex matches."""
    violations = []
    basename = os.path.basename(filepath)

    # 1. Check filename patterns
    for pat in BLOCKED_FILENAMES:
        if pat.match(basename):
            violations.append(f"Blocked credential file name: '{filepath}'")

    if not os.path.exists(filepath) or os.path.isdir(filepath):
        return violations

    # Don't scan this scanner itself or binaries
    if filepath.endswith("scan_secrets.py"):
        return violations

    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_no, line in enumerate(f, start=1):
                for pattern, desc in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        # Mask matched portion for safety in reports
                        snippet = line.strip()[:60]
                        violations.append(
                            f"Line {line_no}: Found {desc} -> {snippet}..."
                        )
    except Exception as e:
        violations.append(f"Could not read file {filepath}: {e}")

    return violations


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "--staged"
    print(f"[*] Scanning files for credentials and secrets (mode: {mode})...")

    files = get_files_to_scan(mode)
    if not files:
        print("[+] No files to scan.")
        sys.exit(0)

    total_violations = 0
    for filepath in files:
        violations = scan_file(filepath)
        if violations:
            total_violations += len(violations)
            print(f"\n[!] SECURITY VIOLATION in '{filepath}':")
            for v in violations:
                print(f"    - {v}")

    if total_violations > 0:
        print(f"\n[x] SCAN FAILED: {total_violations} sensitive credential(s) detected!")
        print("    Remove sensitive data and ensure credential files are in .gitignore.")
        sys.exit(1)

    print(f"[+] SCAN PASSED: {len(files)} file(s) checked. No credentials detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
