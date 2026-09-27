"""Thin wrapper around the official ``ollama`` Python package.

This module provides :class:`OllamaModel` — a minimal, framework-free
interface to the Ollama chat API with tool-calling support.

The agent will eventually execute inside **Google Colab**, so the host is
fully configurable (default: ``OLLAMA_HOST`` env var or
``http://localhost:11434``).  No localhost assumption is baked in.

Design constraints
------------------
* No retry / repair logic — the caller is responsible.
* No LangChain / LangGraph / framework abstractions.
* No tool *execution* — tool definitions are forwarded to the model, and
  tool-call metadata is preserved in the response, but execution is the
  caller's responsibility.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import ollama
from ollama import ChatResponse

from agent.types import AgentConfig


class OllamaModel:
    """Thin, stateless wrapper around an ``ollama.Client``.

    Parameters
    ----------
    config:
        Agent configuration (model name, host URL, temperature, …).
        When omitted a default :class:`AgentConfig` is used, which reads
        ``OLLAMA_HOST`` and ``AGENT_MODEL`` from the environment.
    """

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        self._client: ollama.Client = ollama.Client(host=self._config.host)

    # -- public properties --------------------------------------------------

    @property
    def model(self) -> str:
        """Return the model name used for completions."""
        return self._config.model

    @property
    def host(self) -> str:
        """Return the Ollama server URL."""
        return self._config.host

    # -- core API -----------------------------------------------------------

    def chat(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        tools: Sequence[Callable[..., Any] | dict[str, Any]] | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        """Send a chat request and return the raw ``ChatResponse``.

        Parameters
        ----------
        messages:
            Conversation history as a sequence of dicts with at minimum
            ``{"role": ..., "content": ...}`` keys.  Tool-call and tool-
            result messages are passed through unmodified.
        tools:
            Optional tool definitions.  Each element may be a Python
            callable (the SDK converts it automatically using Google-style
            docstrings) or an explicit JSON-schema dict.
        temperature:
            Override the default temperature for this request.

        Returns
        -------
        ChatResponse
            The full Ollama response — including ``message.content``,
            ``message.tool_calls``, and timing metadata.
        """
        temp = temperature if temperature is not None else self._config.temperature

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": list(messages),
            "options": {"temperature": temp},
        }
        if tools is not None:
            kwargs["tools"] = list(tools)

        response: ChatResponse = self._client.chat(**kwargs)
        return response

    # -- convenience helpers ------------------------------------------------

    @staticmethod
    def has_tool_calls(response: ChatResponse) -> bool:
        """Return ``True`` if the response contains at least one tool call."""
        import json
        tool_calls = getattr(response.message, "tool_calls", None)
        if tool_calls:
            return True
        content = getattr(response.message, "content", "") or ""
        print(f"DEBUG has_tool_calls: raw content={repr(content)}", flush=True)
        # Extract JSON block if present
        import re
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            clean_content = match.group(1).strip()
        else:
            clean_content = content.strip()

        print(f"DEBUG has_tool_calls: clean_content={repr(clean_content)}", flush=True)

        if (clean_content.startswith("{") and clean_content.endswith("}")) or (clean_content.startswith("[") and clean_content.endswith("]")):
            try:
                parsed = json.loads(clean_content)
                if isinstance(parsed, dict) and "name" in parsed:
                    return True
                elif isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict) and "name" in parsed[0]:
                    return True
            except json.JSONDecodeError:
                pass
            
            # Try JSON Lines
            try:
                lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
                if all(l.startswith("{") and l.endswith("}") for l in lines):
                    parsed_lines = [json.loads(l) for l in lines]
                    if len(parsed_lines) > 0 and isinstance(parsed_lines[0], dict) and "name" in parsed_lines[0]:
                        return True
            except json.JSONDecodeError:
                pass
        return False

    @staticmethod
    def extract_tool_calls(
        response: ChatResponse,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from a response as plain dicts.

        Each returned dict has the shape::

            {
                "name": "tool_name",
                "arguments": {"arg1": value1, ...},
            }

        Returns an empty list when no tool calls are present.
        """
        import json
        raw_calls = getattr(response.message, "tool_calls", None)
        
        extracted: list[dict[str, Any]] = []
        if raw_calls:
            for tc in raw_calls:
                fn = tc.function
                extracted.append({
                    "name": fn.name,
                    "arguments": dict(fn.arguments) if fn.arguments else {},
                })
            return extracted
            
        content = getattr(response.message, "content", "") or ""
        import re
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            clean_content = match.group(1).strip()
        else:
            clean_content = content.strip()

        if (clean_content.startswith("{") and clean_content.endswith("}")) or (clean_content.startswith("[") and clean_content.endswith("]")):
            try:
                parsed = json.loads(clean_content)
                if isinstance(parsed, dict) and "name" in parsed:
                    extracted.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {})
                    })
                elif isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "name" in item:
                            extracted.append({
                                "name": item["name"],
                                "arguments": item.get("arguments", {})
                            })
                return extracted
            except json.JSONDecodeError:
                pass
                
            # Try JSON Lines
            try:
                lines = [l.strip() for l in clean_content.splitlines() if l.strip()]
                if all(l.startswith("{") and l.endswith("}") for l in lines):
                    for l in lines:
                        parsed = json.loads(l)
                        if isinstance(parsed, dict) and "name" in parsed:
                            extracted.append({
                                "name": parsed["name"],
                                "arguments": parsed.get("arguments", {})
                            })
                return extracted
            except json.JSONDecodeError:
                pass
                
        return extracted

    def is_available(self, *, timeout: float = 5.0) -> bool:
        """Return ``True`` if the Ollama server is reachable.

        This is a best-effort connectivity check — useful for deciding
        whether to skip live tests.
        """
        try:
            self._client.list()
            return True
        except Exception:
            return False
