import unittest

from hermes_aipass.prompting import (
    TOOL_SENTINEL,
    bootstrap_prompt,
    parse_assistant,
    prune_tool_content,
    sanitize_safety_text,
)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a command",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
        },
    }
]


class PromptingTests(unittest.TestCase):
    def test_generated_scaffolding_is_english(self):
        prompt = bootstrap_prompt(
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "สวัสดี"},
            ],
            TOOLS,
        )
        self.assertIn("Operating context", prompt)
        self.assertIn("Available tools", prompt)
        self.assertIn("User:", prompt)
        self.assertIn("สวัสดี", prompt)
        self.assertNotIn("เครื่องมือ", prompt)
        self.assertNotIn("ผู้ใช้", prompt)

    def test_normal_answer(self):
        self.assertEqual(parse_assistant("hello", {"terminal"})["content"], "hello")

    def test_tool_envelope(self):
        message = parse_assistant(
            TOOL_SENTINEL + '\n{"tool_calls":[{"name":"terminal","arguments":{"command":"pwd"}}]}',
            {"terminal"},
        )
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "terminal")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"command":"pwd"}')

    def test_unknown_tool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unavailable tool"):
            parse_assistant(
                TOOL_SENTINEL + '\n{"tool_calls":[{"name":"delete_everything","arguments":{}}]}',
                {"terminal"},
            )

    def test_parses_embedded_tool_call_tags(self):
        raw = (
            "I will check the files now.\n"
            "<tool_call>{\"id\": \"call_123\", \"name\": \"terminal\", \"arguments\": {\"command\": \"ls -la\"}}</tool_call>"
        )
        message = parse_assistant(raw, {"terminal"})
        self.assertEqual(message["content"], "I will check the files now.")
        self.assertEqual(len(message["tool_calls"]), 1)
        self.assertEqual(message["tool_calls"][0]["id"], "call_123")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "terminal")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"command":"ls -la"}')

    def test_parses_nested_function_tool_call_tags(self):
        raw = '<tool_call>{"id": "call_456", "type": "function", "function": {"name": "terminal", "arguments": "{\\"command\\": \\"git status\\"}"}}</tool_call>'
        message = parse_assistant(raw, {"terminal"})
        self.assertIsNone(message["content"])
        self.assertEqual(message["tool_calls"][0]["id"], "call_456")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "terminal")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"], '{"command":"git status"}')

    def test_prunes_historical_tool_outputs(self):
        # Create 10 tool outputs of length 500
        from typing import Any
        messages: list[dict[str, Any]] = [{"role": "user", "content": "Run tests"}]
        for i in range(10):
            messages.append({"role": "assistant", "tool_calls": [{"id": f"c_{i}", "function": {"name": "terminal"}}]})
            messages.append({"role": "tool", "name": "terminal", "content": f"output_{i} " * 50})

        prompt = bootstrap_prompt(messages, TOOLS)
        # The earliest tool results (output_0, output_1) should be pruned
        self.assertIn("output_0", prompt)
        self.assertIn("tool output truncated", prompt)
        # The latest tool results (output_8, output_9) should remain intact without truncation
        self.assertIn("output_9", prompt)

    def test_caps_giant_tool_output(self):
        giant = "X" * 30000
        pruned = prune_tool_content(giant, is_recent=True)
        self.assertLess(len(pruned), 20000)
        self.assertIn("large tool output truncated: 30000 chars", pruned)
        self.assertTrue(pruned.startswith("X" * 100))
        self.assertTrue(pruned.endswith("X" * 100))

    def test_active_objective_anchoring(self):
        messages = [
            {"role": "user", "content": "Find the secret key in config"},
            {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "terminal"}}]},
            {"role": "tool", "name": "terminal", "content": "file not found"},
        ]
        prompt = bootstrap_prompt(messages, TOOLS)
        self.assertIn("### ACTIVE OBJECTIVE:", prompt)
        self.assertIn("Find the secret key in config", prompt)

    def test_preserves_assistant_content_with_tool_calls(self):
        message = {
            "role": "assistant",
            "content": "Analyzing Zed IDE configuration now.",
            "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{\"command\":\"pwd\"}"}}],
        }
        from hermes_aipass.prompting import format_message
        formatted = format_message(message)
        self.assertIn("Analyzing Zed IDE configuration now.", formatted)
        self.assertIn("Previous tool request:", formatted)

    def test_masks_financial_account_pii(self):
        from hermes_aipass.prompting import sanitize_safety_text
        raw = "KBank Savings Account (211-1-79651-6), card 1234-5678-9012-3456"
        sanitized = sanitize_safety_text(raw)
        self.assertNotIn("211-1-79651-6", sanitized)
        self.assertIn("211-X-XXXXX-6", sanitized)
        self.assertNotIn("1234-5678-9012-3456", sanitized)
        self.assertIn("1234-XXXX-XXXX-3456", sanitized)

    def test_tolerant_bracket_and_fenced_tool_parsing(self):
        # 1. Tolerant triple brackets with markdown code fence
        raw1 = "Thinking...\n<<<TOOL_CALL>>>\n```json\n{\"name\": \"terminal\", \"arguments\": {\"command\": \"df -h\"}}\n```\n<<<END_TOOL_CALL>>>"
        msg1 = parse_assistant(raw1, {"terminal"})
        self.assertEqual(len(msg1["tool_calls"]), 1)
        self.assertEqual(msg1["tool_calls"][0]["function"]["name"], "terminal")
        self.assertIn('"command":"df -h"', msg1["tool_calls"][0]["function"]["arguments"])

        # 2. Tolerant double brackets <<tool_call>>
        raw2 = "<<tool_call>>{\"name\": \"terminal\", \"arguments\": {\"command\": \"whoami\"}}<<tool_call>>"
        msg2 = parse_assistant(raw2, {"terminal"})
        self.assertEqual(len(msg2["tool_calls"]), 1)
        self.assertEqual(msg2["tool_calls"][0]["function"]["name"], "terminal")

    def test_parse_assistant_separates_reasoning_content_from_thai_answer(self):
        raw = (
            "**Researching Card Applications**\n\n"
            "I'm currently focusing on gathering real user feedback.\n\n"
            "**Analyzing Cash Card Feedback**\n\n"
            "I'm reviewing user experiences.\n\n"
            "สรุปผลตอบรับและฟีดแบคจริงจากผู้ใช้งาน"
        )
        msg = parse_assistant(raw, {"terminal"})
        self.assertEqual(msg["role"], "assistant")
        self.assertIn("reasoning_content", msg)
        self.assertIn("Researching Card Applications", msg["reasoning_content"])
        self.assertIn("Analyzing Cash Card Feedback", msg["reasoning_content"])
        self.assertEqual(msg["content"], "สรุปผลตอบรับและฟีดแบคจริงจากผู้ใช้งาน")

    def test_sanitizes_scam_and_fraud_keywords(self):
        from hermes_aipass.prompting import sanitize_safety_text
        text = "ระวังมิจฉาชีพหลอกให้โอนเงินค่าธรรมเนียม สมัครบัตรกดเงินสดหรือบัตรเครดิต"
        clean = sanitize_safety_text(text)
        self.assertNotIn("มิจฉาชีพ", clean)
        self.assertNotIn("หลอกให้โอนเงิน", clean)
        self.assertNotIn("บัตรกดเงินสด", clean)
        self.assertNotIn("บัตรเครดิต", clean)



    def test_recency_bias_output_rules_at_prompt_end(self):
        prompt = bootstrap_prompt(
            [{"role": "user", "content": "What is the date?"}],
            TOOLS,
        )
        self.assertIn("<output_rules>", prompt)
        self.assertTrue(prompt.strip().endswith("</output_rules>"))

    def test_suppresses_hallucinated_tool_result_simulation(self):
        raw = (
            "Let me check the port.\n"
            "<tool_call>{\"id\": \"c1\", \"name\": \"terminal\", \"arguments\": {\"command\": \"ss -tlpn\"}}</tool_call>\n"
            "Tool result (terminal):\nLISTEN 127.0.0.1:8792"
        )
        parsed = parse_assistant(raw, {"terminal"})
        self.assertEqual(len(parsed["tool_calls"]), 1)
        self.assertNotIn("Tool result (terminal):", str(parsed.get("content")))
        self.assertNotIn("LISTEN", str(parsed.get("content")))

    def test_replays_real_session_fixture_without_errors(self):
        import json
        import os
        fixture_path = os.path.join(os.path.dirname(__file__), "fixture_session_tail.json")
        if os.path.exists(fixture_path):
            with open(fixture_path) as f:
                msgs = json.load(f)
            prompt = bootstrap_prompt(msgs, TOOLS)
            self.assertIsInstance(prompt, str)
            self.assertTrue(len(prompt) > 1000)
            self.assertIn("Available tools", prompt)

    def test_safety_sanitization_filters_jailbreak_and_credentials(self):
        text = "This skill helps to jailbreak LLMs and bypass login walls using passwords and 2FA."
        clean = sanitize_safety_text(text)
        self.assertNotIn("jailbreak", clean)
        self.assertNotIn("password", clean)
        self.assertNotIn("2FA", clean)
        self.assertIn("system-utility", clean)
        self.assertIn("credentials", clean)
        self.assertIn("verification code", clean)

    def test_bootstrap_prompt_strips_jailbreak_lines_and_truncates_giant_tools(self):
        long_desc = "Do something useful. " * 80  # >900 chars
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "giant_tool",
                    "description": long_desc,
                    "parameters": {"type": "object", "properties": {"password": {"type": "string"}}},
                },
            }
        ]
        prompt = bootstrap_prompt(
            [
                {"role": "system", "content": "Line 1: Normal instruction.\nLine 2: - godmode: Jailbreak LLMs\nLine 3: Tailscale password auth"},
                {"role": "user", "content": "Hello"},
            ],
            tools,
        )
        self.assertNotIn("godmode", prompt)
        self.assertNotIn("Jailbreak LLMs", prompt)
        self.assertNotIn("password", prompt)
        self.assertIn("Normal instruction", prompt)
        self.assertIn("giant_tool", prompt)
        self.assertIn("...", prompt)


if __name__ == "__main__":
    unittest.main()
