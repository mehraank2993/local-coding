import ollama
import json

def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

print("Sending request...")
resp = ollama.chat(
    model="qwen2.5-coder:7b",
    messages=[{"role": "user", "content": "Calculate 20 plus 22 using the tool."}],
    tools=[add]
)

tc = getattr(resp.message, "tool_calls", None)
print("Tool calls:", tc)
print("Content:", resp.message.content)
