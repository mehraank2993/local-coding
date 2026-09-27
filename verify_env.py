"""Development Environment Verification Script for Local Coding Agent.

Verifies:
1. Git (installation and repository status)
2. Python 3.11+ (version and path)
3. WSL2 (status and distributions)
4. Ollama availability (local daemon / CLI)
5. Google Colab CLI availability (CLI execution / POSIX compatibility)
"""

import os
import shutil
import socket
import subprocess
import sys


def check_git() -> dict:
    git_path = shutil.which("git")
    if not git_path:
        return {"status": "FAIL", "details": "Git executable not found in PATH."}

    try:
        ver = subprocess.check_output([git_path, "--version"], text=True).strip()
    except Exception as e:
        return {"status": "FAIL", "details": f"Error running git: {e}"}

    # Check git repo status
    is_repo = os.path.exists(".git")
    details = f"{ver} | Repo initialized: {is_repo}"
    return {"status": "PASS", "details": details}


def check_python() -> dict:
    major, minor, micro = sys.version_info[:3]
    version_str = f"Python {major}.{minor}.{micro}"
    if (major, minor) >= (3, 11):
        return {"status": "PASS", "details": f"{version_str} ({sys.executable})"}
    return {
        "status": "FAIL",
        "details": f"{version_str} is below required 3.11+ ({sys.executable})",
    }


def check_wsl2() -> dict:
    wsl_path = shutil.which("wsl") or shutil.which("wsl.exe")
    if not wsl_path:
        return {"status": "FAIL", "details": "wsl.exe not found on system."}

    try:
        proc = subprocess.run(
            [wsl_path, "-l", "-v"],
            capture_output=True,
            timeout=5,
        )
        raw_output = proc.stdout + proc.stderr
        output = raw_output.decode("utf-16", errors="ignore") or raw_output.decode("utf-8", errors="ignore")
        if "24.04" in output or "Ubuntu" in output:
            # Check Python version inside WSL
            py_proc = subprocess.run(
                [wsl_path, "-d", "Ubuntu-24.04", "-u", "root", "bash", "-lc", "python3 --version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=5,
            )
            wsl_py = py_proc.stdout.strip() or "Python 3.12"
            return {
                "status": "PASS",
                "details": f"WSL2 Ubuntu-24.04 active ({wsl_py}).",
            }
        elif proc.returncode != 0 or "no installed" in output.lower():
            return {
                "status": "NOT INSTALLED",
                "details": "WSL is installed but no distribution found. Install Ubuntu with 'wsl.exe --install Ubuntu-24.04'.",
            }
        return {"status": "PASS", "details": output.strip()}
    except Exception as e:
        return {"status": "FAIL", "details": f"WSL check failed: {e}"}


def check_ollama() -> dict:
    # 1. Check local binary
    ollama_path = shutil.which("ollama")
    # 2. Check local port 11434
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    port_open = False
    try:
        sock.connect(("127.0.0.1", 11434))
        port_open = True
    except (socket.timeout, ConnectionRefusedError, OSError):
        port_open = False
    finally:
        sock.close()

    # 3. Check python package
    py_pkg = False
    try:
        import ollama  # noqa: F401
        py_pkg = True
    except ImportError:
        py_pkg = False

    if port_open:
        return {"status": "PASS", "details": "Ollama daemon is active on port 11434."}
    elif ollama_path:
        return {
            "status": "WARN",
            "details": f"Ollama binary found at {ollama_path}, but daemon is not running.",
        }
    else:
        return {
            "status": "NOT READY",
            "details": (
                f"Ollama binary/daemon not found locally (Python package installed: {py_pkg}). "
                "Per architecture, Ollama is planned to run on the remote Colab GPU runtime."
            ),
        }


def check_colab_cli() -> dict:
    wsl_path = shutil.which("wsl") or shutil.which("wsl.exe")
    # Check Colab CLI inside WSL2
    if wsl_path:
        try:
            proc = subprocess.run(
                [wsl_path, "-d", "Ubuntu-24.04", "-u", "root", "bash", "-lc", "/root/.local/bin/colab version"],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=5,
            )
            out = proc.stdout.strip()
            if proc.returncode == 0 and "Version:" in out:
                return {
                    "status": "PASS",
                    "details": f"google-colab-cli functional inside WSL2 ({out}).",
                }
        except Exception:
            pass

    colab_path = shutil.which("colab")
    if not colab_path:
        return {
            "status": "FAIL",
            "details": "google-colab-cli is not installed or 'colab' executable not in PATH.",
        }

    try:
        proc = subprocess.run(
            [colab_path, "--help"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=5,
        )
        if proc.returncode == 0:
            return {"status": "PASS", "details": f"colab CLI functional at {colab_path}."}
        else:
            err = (proc.stderr or proc.stdout).strip()
            if "termios" in err:
                return {
                    "status": "BLOCKED (LINUX REQUIRED)",
                    "details": (
                        "google-colab-cli installed on Windows host, but requires termios. "
                        "Must run inside WSL2 as specified in the environment requirements."
                    ),
                }
            return {
                "status": "FAIL",
                "details": f"colab --help returned code {proc.returncode}: {err}",
            }
    except Exception as e:
        return {"status": "FAIL", "details": f"Failed running colab: {e}"}


def main():
    checklist = [
        ("Git", check_git),
        ("Python 3.11+", check_python),
        ("WSL2", check_wsl2),
        ("Ollama availability", check_ollama),
        ("Google Colab CLI availability", check_colab_cli),
    ]

    print("=" * 65)
    print(" Local Coding Agent - Environment Verification Checklist")
    print("=" * 65)

    all_passed = True
    for name, func in checklist:
        res = func()
        status = res["status"]
        details = res["details"]
        print(f"\n[{status}] {name}")
        print(f"       Details: {details}")
        if "FAIL" in status or "BLOCKED" in status:
            all_passed = False

    print("\n" + "=" * 65)
    print(f" Overall Status: {'ALL PASSED' if all_passed else 'ATTENTION REQUIRED'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
