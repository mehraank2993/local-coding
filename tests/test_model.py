"""Tests for agent.model — unit tests (mocked) and an optional live smoke test.

Unit tests run without an Ollama server.  The live smoke test is
automatically **skipped** when Ollama is unreachable, so running
``python -m pytest`` always succeeds regardless of infrastructure.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent.model import OllamaModel
from agent.types import AgentConfig


# ═══════════════════════════════════════════════════════════════════════════
# Unit tests (no server required)
# ═══════════════════════════════════════════════════════════════════════════


class TestOllamaModelInit:
    """OllamaModel construction and properties."""

    def test_default_config(self):
        model = OllamaModel()
        assert model.model == AgentConfig().model
        assert model.host == AgentConfig().host

    def test_custom_config(self):
        cfg = AgentConfig(model="custom:latest", host="http://colab-host:11434")
        model = OllamaModel(config=cfg)
        assert model.model == "custom:latest"
        assert model.host == "http://colab-host:11434"


class TestChatMocked:
    """OllamaModel.chat with a mocked client — verifies call forwarding."""

    def _make_model(self) -> tuple[OllamaModel, MagicMock]:
        model = OllamaModel()
        mock_client = MagicMock()
        model._client = mock_client  # swap in mock
        return model, mock_client

    def test_basic_chat(self):
        model, mock_client = self._make_model()
        # Build a fake ChatResponse-like object
        fake_msg = MagicMock()
        fake_msg.content = "Hello!"
        fake_msg.tool_calls = None
        fake_resp = MagicMock()
        fake_resp.message = fake_msg
        mock_client.chat.return_value = fake_resp

        messages = [{"role": "user", "content": "hi"}]
        resp = model.chat(messages)

        mock_client.chat.assert_called_once()
        call_kwargs = mock_client.chat.call_args
        assert call_kwargs.kwargs["model"] == model.model
        assert resp.message.content == "Hello!"

    def test_chat_with_tools(self):
        model, mock_client = self._make_model()
        # Simulate a tool-call response
        fake_fn = MagicMock()
        fake_fn.name = "add"
        fake_fn.arguments = {"a": 3, "b": 5}
        fake_tc = MagicMock()
        fake_tc.function = fake_fn
        fake_msg = MagicMock()
        fake_msg.content = ""
        fake_msg.tool_calls = [fake_tc]
        fake_resp = MagicMock()
        fake_resp.message = fake_msg
        mock_client.chat.return_value = fake_resp

        def add(a: int, b: int) -> int:
            """Add two integers.

            Args:
              a: First number
              b: Second number

            Returns:
              int: Sum
            """
            return a + b

        messages = [{"role": "user", "content": "use the add tool"}]
        resp = model.chat(messages, tools=[add])

        # tools should be forwarded to the client
        call_kwargs = mock_client.chat.call_args.kwargs
        assert "tools" in call_kwargs
        assert OllamaModel.has_tool_calls(resp) is True

    def test_chat_preserves_tool_call_metadata(self):
        model, mock_client = self._make_model()
        fake_fn = MagicMock()
        fake_fn.name = "multiply"
        fake_fn.arguments = {"x": 7, "y": 6}
        fake_tc = MagicMock()
        fake_tc.function = fake_fn
        fake_msg = MagicMock()
        fake_msg.content = "I'll use the multiply tool."
        fake_msg.tool_calls = [fake_tc]
        fake_resp = MagicMock()
        fake_resp.message = fake_msg
        mock_client.chat.return_value = fake_resp

        resp = model.chat([{"role": "user", "content": "7*6"}])
        # Content is preserved
        assert resp.message.content == "I'll use the multiply tool."
        # Tool-call metadata is preserved
        calls = OllamaModel.extract_tool_calls(resp)
        assert len(calls) == 1
        assert calls[0]["name"] == "multiply"
        assert calls[0]["arguments"]["x"] == 7
        assert calls[0]["arguments"]["y"] == 6


class TestExtractToolCalls:
    """Tests for the static helper methods."""

    def test_no_tool_calls(self):
        resp = MagicMock()
        resp.message.content = ""
        resp.message.tool_calls = None
        assert OllamaModel.has_tool_calls(resp) is False
        assert OllamaModel.extract_tool_calls(resp) == []

    def test_empty_tool_calls(self):
        resp = MagicMock()
        resp.message.content = ""
        resp.message.tool_calls = []
        assert OllamaModel.has_tool_calls(resp) is False
        assert OllamaModel.extract_tool_calls(resp) == []


class TestIsAvailable:
    """Tests for the connectivity check helper."""

    def test_available_when_list_succeeds(self):
        model = OllamaModel()
        model._client = MagicMock()
        model._client.list.return_value = {"models": []}
        assert model.is_available() is True

    def test_unavailable_when_list_fails(self):
        model = OllamaModel()
        model._client = MagicMock()
        model._client.list.side_effect = Exception("connection refused")
        assert model.is_available() is False


# ═══════════════════════════════════════════════════════════════════════════
# Live smoke test — skipped when Ollama is not reachable
# ═══════════════════════════════════════════════════════════════════════════


def _ollama_is_available() -> bool:
    """Best-effort check whether an Ollama server is reachable."""
    try:
        return OllamaModel().is_available()
    except Exception:
        return False


@pytest.mark.skipif(
    not _ollama_is_available(),
    reason="Ollama server is not available — skipping live smoke test",
)
class TestLiveSmokeTest:
    """Live integration test against a running Ollama instance.

    The prompt is crafted so the model **must** call the ``add`` tool —
    a question it cannot answer without invoking the tool.
    """

    def test_tool_call_with_add(self):
        model = OllamaModel()

        def add(a: int, b: int) -> int:
            """Add two integers together.

            Args:
              a: First integer to add
              b: Second integer to add

            Returns:
              int: The sum of a and b
            """
            return a + b

        messages = [
            {
                "role": "user",
                "content": (
                    "Use the 'add' tool to compute the sum of 4021 and 1783. "
                    "You MUST call the tool — do NOT compute the answer yourself."
                ),
            },
        ]

        response = model.chat(messages, tools=[add], temperature=0.0)

        # 1. Response exists
        assert response is not None
        assert response.message is not None

        # 2. Tool call exists
        assert OllamaModel.has_tool_calls(response), (
            f"Expected a tool call but got none.  "
            f"Content: {response.message.content!r}"
        )

        # 3. Extract and validate
        calls = OllamaModel.extract_tool_calls(response)
        assert len(calls) >= 1

        call = calls[0]

        # 4. Tool name is 'add'
        assert call["name"] == "add", f"Expected tool 'add', got {call['name']!r}"

        # 5. Arguments contain a and b with valid values
        args = call["arguments"]
        assert "a" in args, f"Missing argument 'a' in {args}"
        assert "b" in args, f"Missing argument 'b' in {args}"

        # The values should be the integers from the prompt
        a_val = args["a"]
        b_val = args["b"]
        assert isinstance(a_val, (int, float)), f"'a' is not numeric: {a_val!r}"
        assert isinstance(b_val, (int, float)), f"'b' is not numeric: {b_val!r}"

        # Verify the model picked up the right numbers
        assert {int(a_val), int(b_val)} == {4021, 1783}, (
            f"Expected arguments {{4021, 1783}}, got {{{int(a_val)}, {int(b_val)}}}"
        )

        print(
            f"\n  ✓ Live smoke test passed"
            f"\n    Model : {model.model}"
            f"\n    Host  : {model.host}"
            f"\n    Tool  : {call['name']}"
            f"\n    Args  : a={a_val}, b={b_val}"
        )
