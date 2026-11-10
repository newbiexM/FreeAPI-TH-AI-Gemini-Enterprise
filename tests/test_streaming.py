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
