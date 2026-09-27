import subprocess
print("Running curl install...")
res = subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, capture_output=True, text=True)
print("Return code:", res.returncode)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
