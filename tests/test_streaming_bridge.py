import email.message
import io
import json
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

from hermes_aipass.bridge import Handler, RUNTIME


def _make_headers(mapping: dict[str, str]) -> Any:
    msg = email.message.Message()
    for k, v in mapping.items():
        msg[k] = v
    return msg


class DummyRequest:
    def __init__(self, body_bytes: bytes = b""):
        self.body_bytes = body_bytes

    def makefile(self, *args, **kwargs):
        return io.BytesIO(self.body_bytes)


class TestStreamingBridge(unittest.TestCase):
    @patch.object(Handler, "send_response")
    @patch.object(Handler, "send_header")
    @patch.object(Handler, "end_headers")
    def test_stream_completion_incremental_dispatch(self, mock_end, mock_header, mock_resp):
        handler = Handler.__new__(Handler)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = _make_headers({"X-AiPASS-Session-Id": "test-stream-sess"})

        # Mock execute returning a multi-chunk delta generator
        deltas = ["Hello ", "world, ", "this is ", "realtime streaming!"]
        mock_execute = MagicMock(return_value=("conv-123", iter(deltas), set(), "test-stream-sess", "prompt"))
        handler._execute = mock_execute
        handler._complete_state = MagicMock()
        handler._session_id = MagicMock(return_value="test-stream-sess")

        body = {"model": "gemini-3.8-flash", "stream": True}
        messages = [{"role": "user", "content": "hello"}]

        handler._stream_completion(body, messages, "gemini-3.8-flash")

        written = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("data: ", written)
        self.assertIn("Hello ", written)
        self.assertIn("realtime streaming!", written)
        self.assertIn("[DONE]", written)
        handler._complete_state.assert_called_once()

    @patch.object(Handler, "send_response")
    @patch.object(Handler, "send_header")
    @patch.object(Handler, "end_headers")
    def test_stream_completion_emits_reasoning_content_delta(self, mock_end, mock_header, mock_resp):
        handler = Handler.__new__(Handler)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = _make_headers({"X-AiPASS-Session-Id": "test-stream-sess"})

        deltas = [
            "**Researching Card Applications**\n\nI'm currently investigating.\n\n",
            "Here is the final answer in Thai."
        ]
        mock_execute = MagicMock(return_value=("conv-123", iter(deltas), set(), "test-stream-sess", "prompt"))
        handler._execute = mock_execute
        handler._complete_state = MagicMock()
        handler._session_id = MagicMock(return_value="test-stream-sess")

        body = {"model": "gemini-3.8-flash", "stream": True}
        messages = [{"role": "user", "content": "hello"}]

        handler._stream_completion(body, messages, "gemini-3.8-flash")

        written = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("reasoning_content", written)
        self.assertIn("Researching Card Applications", written)

        # Content delta should NOT contain the thought, and reconstructed content matches
        collected_content = []
        for line in written.split("\n"):
            if line.startswith("data: ") and not line.strip().endswith("[DONE]"):
                payload = json.loads(line[6:])
                delta = payload["choices"][0]["delta"]
                if "content" in delta:
                    self.assertNotIn("Researching Card Applications", delta["content"])
                    collected_content.append(delta["content"])

        self.assertEqual("".join(collected_content), "Here is the final answer in Thai.")

    @patch.object(Handler, "send_response")
    @patch.object(Handler, "send_header")
    @patch.object(Handler, "end_headers")
    def test_stream_completion_recovers_from_rai_422(self, mock_end, mock_header, mock_resp):
        from hermes_aipass.client import ContentPolicyViolationError
        handler = Handler.__new__(Handler)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = _make_headers({"X-AiPASS-Session-Id": "test-stream-rai"})

        deltas = ["Recovered response after RAI policy retry."]
        mock_client = MagicMock()
        mock_client.create_conversation.return_value = "new-rai-cid"
        mock_client.send_stream.return_value = iter(deltas)

        handler._execute = MagicMock(side_effect=ContentPolicyViolationError("Content policy violation SCAM_FRAUD", code=422))
        handler._complete_state = MagicMock()
        handler._session_id = MagicMock(return_value="test-stream-rai")

        with patch("hermes_aipass.bridge.RUNTIME.get_client", return_value=mock_client):
            body = {"model": "gemini-4-argon", "stream": True}
            messages = [{"role": "user", "content": "กู้เงิน บัตรกดเงินสด มิจฉาชีพ"}]
            handler._stream_completion(body, messages, "gemini-4-argon")

        written = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("Recovered response after RAI policy retry.", written)
        self.assertIn("[DONE]", written)



    @patch.object(Handler, "send_response")
    @patch.object(Handler, "send_header")
    @patch.object(Handler, "end_headers")
    def test_stream_completion_tool_call_incremental(self, mock_end, mock_header, mock_resp):
        handler = Handler.__new__(Handler)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = _make_headers({"X-AiPASS-Session-Id": "test-stream-tool"})

        deltas = [
            "Let me check that.\n",
            "<tool_call>",
            '{"name": "test_tool", "arguments": {"x": 1}}',
            "</tool_call>\nFinished.",
        ]
        mock_execute = MagicMock(return_value=("conv-tool-123", iter(deltas), {"test_tool"}, "test-stream-tool", "prompt"))
        handler._execute = mock_execute
        handler._complete_state = MagicMock()
        handler._session_id = MagicMock(return_value="test-stream-tool")

        body = {"model": "gemini-3.8-flash", "stream": True, "tools": [{"type": "function", "function": {"name": "test_tool"}}]}
        messages = [{"role": "user", "content": "run test_tool"}]

        handler._stream_completion(body, messages, "gemini-3.8-flash")

        written = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("test_tool", written)
        self.assertIn('"finish_reason":"tool_calls"', written)
        self.assertIn("[DONE]", written)
        handler._complete_state.assert_called_once()

    @patch.object(Handler, "send_response")
    @patch.object(Handler, "send_header")
    @patch.object(Handler, "end_headers")
    def test_stream_completion_transparent_refusal_retry(self, mock_end, mock_header, mock_resp):
        handler = Handler.__new__(Handler)
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()
        handler.headers = _make_headers({"X-AiPASS-Session-Id": "test-refusal-retry"})

        # First attempt returns a refusal snippet, retry returns valid text
        first_deltas = ["ขออภัย ไม่สามารถตอบคำถามนี้ได้"]
        second_deltas = ["แน่นอนครับ ", "นี่คือคำตอบที่ถูกต้อง"]

        mock_execute = MagicMock(return_value=("conv-refusal", iter(first_deltas), set(), "test-refusal-retry", "prompt"))
        handler._execute = mock_execute
        handler._complete_state = MagicMock()
        handler._session_id = MagicMock(return_value="test-refusal-retry")
        handler._sanitize_for_retry = MagicMock(return_value="sanitized prompt")

        mock_client = MagicMock()
        mock_client.create_conversation.return_value = "conv-retried"
        mock_client.send_stream.return_value = iter(second_deltas)

        with patch("hermes_aipass.bridge.RUNTIME.get_client", return_value=mock_client):
            body = {"model": "gemini-3.8-flash", "stream": True}
            messages = [{"role": "user", "content": "refusal trigger"}]

            handler._stream_completion(body, messages, "gemini-3.8-flash")

        written = handler.wfile.getvalue().decode("utf-8")
        self.assertNotIn("ขออภัย", written)
        self.assertIn("นี่คือคำตอบที่ถูกต้อง", written)
        self.assertIn("[DONE]", written)
        mock_client.create_conversation.assert_called_once()
