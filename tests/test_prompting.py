import unittest

from hermes_aipass.prompting import TOOL_SENTINEL, bootstrap_prompt, parse_assistant


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
        self.assertIn("System instructions", prompt)
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


if __name__ == "__main__":
    unittest.main()

