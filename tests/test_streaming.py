import unittest
from hermes_aipass.streaming import find_marker

class TestMarkerMatcher(unittest.TestCase):
    def test_find_exact_and_tolerant_tool_start_markers(self):
        # Exact standard XML
        self.assertEqual(find_marker("Hello <tool_call>{\"name\":", "start"), (6, 11))
        # Tolerant double bracket XML
        self.assertEqual(find_marker("Hello <<tool_call>>{\"name\":", "start"), (6, 13))
        # Qwen modern triple bracket
        self.assertEqual(find_marker("Hello <<<TOOL_CALL>>>{\"name\":", "start"), (6, 15))
        # Quadruple bracket
        self.assertEqual(find_marker("Hello <<<<TOOL_CALL>>>>{\"name\":", "start"), (6, 17))
        # Sentinel
        self.assertEqual(find_marker("Hello\n@@HERMES_TOOL_CALL@@\n{", "start"), (6, 20))

    def test_find_end_markers(self):
        self.assertEqual(find_marker("{\"name\":\"bash\"}</tool_call> Done", "end"), (15, 12))
        self.assertEqual(find_marker("{\"name\":\"bash\"}<<<END_TOOL_CALL>>> Done", "end"), (15, 19))
        self.assertEqual(find_marker("{\"name\":\"bash\"}<<END_TOOL_CALL>> Done", "end"), (15, 17))

    def test_incomplete_marker_at_stream_boundary(self):
        # When not final and marker touches end without closing brackets, return incomplete (-2, 0)
        self.assertEqual(find_marker("Thinking... <<<TOOL_", "start", final=False), (-2, 0))
        self.assertEqual(find_marker("Thinking... <tool_", "start", final=False), (-2, 0))
        # If final, it's treated as plain text (-1, 0)
        self.assertEqual(find_marker("Thinking... <<<TOOL_", "start", final=True), (-1, 0))


class TestStreamInterceptor(unittest.TestCase):
    def test_pure_text_streaming_with_holdback(self):
        from hermes_aipass.streaming import StreamInterceptor
        interceptor = StreamInterceptor()
        chunk1 = interceptor.feed("Hello world! This is a long conversational sentence that flows smoothly.")
        # Some text should immediately emit while retaining a safe tail
        self.assertTrue(len(chunk1.text) > 0)
        self.assertEqual(chunk1.tool_calls, [])
        chunk2 = interceptor.finish()
        self.assertTrue(chunk2.text.endswith("flows smoothly."))

    def test_tool_call_interception_without_text_leakage(self):
        from hermes_aipass.streaming import StreamInterceptor
        interceptor = StreamInterceptor(allowed_tools={"read_file"})
        c1 = interceptor.feed("I will read the file.\n<tool_call>")
        # The tool call tag must NOT leak into c1.text
        self.assertNotIn("<tool_call>", c1.text)
        
        c2 = interceptor.feed('{"name": "read_file", "arguments": {"path": "/tmp/test.txt"}}</tool_call>\nAll done.')
        # Complete tool call emitted
        self.assertEqual(len(c2.tool_calls), 1)
        self.assertEqual(c2.tool_calls[0]["function"]["name"], "read_file")
        self.assertIn('"path":"/tmp/test.txt"', c2.tool_calls[0]["function"]["arguments"])
        
        c3 = interceptor.finish()
        self.assertIn("All done.", c3.text)

    def test_strip_markdown_code_fences_inside_tool_call(self):
        from hermes_aipass.streaming import StreamInterceptor
        interceptor = StreamInterceptor(allowed_tools={"run_cmd"})
        interceptor.feed("<<<TOOL_CALL>>>\n```json\n{\"name\": \"run_cmd\", \"arguments\": {\"cmd\": \"ls\"}}\n```\n<<<END_TOOL_CALL>>>")
        chunk = interceptor.finish()
        self.assertEqual(len(chunk.tool_calls), 1)
        self.assertEqual(chunk.tool_calls[0]["function"]["name"], "run_cmd")
