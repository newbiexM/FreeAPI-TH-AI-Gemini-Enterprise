import json
import unittest

from hermes_aipass.anthropic import (
    anthropic_to_openai_request,
    estimate_tokens,
    openai_to_anthropic_response,
)


class TestAnthropicAdapter(unittest.TestCase):
    def test_estimate_tokens(self):
        body = {
            "messages": [{"role": "user", "content": "Hello Anthropic!"}],
            "system": "You are a helpful assistant.",
        }
        tokens = estimate_tokens(body)
        self.assertTrue(tokens > 0)

    def test_anthropic_to_openai_request_translation(self):
        anthropic_body = {
            "model": "claude-3-7-sonnet-20250219",
            "system": [{"type": "text", "text": "System instruction"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Run check"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_123", "name": "bash", "input": {"cmd": "ls"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_123", "content": "file.txt"}
                    ],
                },
            ],
            "tools": [
                {
                    "name": "bash",
                    "description": "Run bash command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                }
            ],
            "stream": True,
        }

        openai_body, messages, model_id = anthropic_to_openai_request(anthropic_body)
        # Model should map cleanly to default or normalized model
        self.assertIn("gemini", model_id)
        # System instruction moved into messages
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System instruction")
        # Assistant tool call converted
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["tool_calls"][0]["function"]["name"], "bash")
        # Tool result converted
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "toolu_123")
        # Tools converted to OpenAI function spec
        self.assertEqual(openai_body["tools"][0]["function"]["name"], "bash")
        self.assertEqual(openai_body["tools"][0]["function"]["parameters"]["required"], ["cmd"])

    def test_openai_to_anthropic_response_translation(self):
        openai_resp = {
            "id": "chatcmpl-456",
            "model": "gemini-3.8-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "I am executing the tool.",
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"/tmp/a.txt"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

        anth_resp = openai_to_anthropic_response(openai_resp)
        self.assertEqual(anth_resp["type"], "message")
        self.assertEqual(anth_resp["role"], "assistant")
        self.assertEqual(anth_resp["stop_reason"], "tool_use")
        self.assertEqual(len(anth_resp["content"]), 2)
        self.assertEqual(anth_resp["content"][0]["type"], "text")
        self.assertEqual(anth_resp["content"][0]["text"], "I am executing the tool.")
        self.assertEqual(anth_resp["content"][1]["type"], "tool_use")
        self.assertEqual(anth_resp["content"][1]["name"], "read_file")
        self.assertEqual(anth_resp["content"][1]["input"], {"path": "/tmp/a.txt"})

    def test_bridge_count_tokens_and_messages_endpoints(self):
        import io
        from unittest.mock import MagicMock, patch
        from hermes_aipass.bridge import Handler

        handler = Handler.__new__(Handler)
        handler.command = "POST"
        handler.requestline = "POST /v1/messages/count_tokens HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.rfile = io.BytesIO(b'{"messages":[{"role":"user","content":"count me"}],"system":"test"}')
        handler.headers = {"Content-Length": str(len(handler.rfile.getvalue()))}
        handler.wfile = io.BytesIO()
        handler.path = "/v1/messages/count_tokens"

        handler.do_POST()
        res_raw = handler.wfile.getvalue().decode("utf-8")
        self.assertIn("input_tokens", res_raw)
