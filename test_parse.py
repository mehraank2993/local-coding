import json

content = '''{"name": "run_shell", "arguments": {"command": "find . -name '*.py' | wc -l > count.txt"}}'''

clean_content = content.strip()
if clean_content.startswith("```json"):
    clean_content = clean_content[7:]
elif clean_content.startswith("```"):
    clean_content = clean_content[3:]
if clean_content.endswith("```"):
    clean_content = clean_content[:-3]
clean_content = clean_content.strip()

print("Clean content:", repr(clean_content))

if clean_content.startswith("{") and clean_content.endswith("}"):
    try:
        parsed = json.loads(clean_content)
        if isinstance(parsed, dict) and "name" in parsed:
            print("TRUE")
        else:
            print("FALSE - dict/name check failed")
    except json.JSONDecodeError as e:
        print("FALSE - decode error:", e)
else:
    print("FALSE - start/end check failed")
