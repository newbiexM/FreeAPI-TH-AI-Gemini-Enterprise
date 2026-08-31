import json
import math
import unittest

from hermes_aipass.codec import decode_loader_bytes, extract_stream_events, text_delta
from hermes_aipass.models import Model, extract_models, is_conversation_model, map_reasoning


class CodecModelTests(unittest.TestCase):
    def test_decodes_flattened_model_loader(self):
        payload = [
            {"_1": 2},
            "data",
            [3],
            {"_4": 5, "_6": 7, "_8": 9, "_10": 11, "_12": 13},
            "id",
            "gpt-5.6-sol",
            "displayName",
            "GPT-5.6 Sol",
            "provider",
            "openai",
            "ready",
            True,
            "thinkingConfig",
            {"_14": 15, "_16": 17},
            "supportedLevels",
            [18, 17, 19],
            "defaultLevel",
            "medium",
            "low",
            "high",
        ]
        decoded = decode_loader_bytes(json.dumps(payload).encode())
        models = extract_models(decoded)
        self.assertEqual([model.id for model in models], ["gpt-5.6-sol"])
        self.assertEqual(models[0].thinking_levels, ("low", "medium", "high"))

    def test_filters_non_conversation_models(self):
        excluded = [
            "gpt-image-2",
            "gemini-3-pro-image",
            "seedance-2.0",
            "seedream-5.0-lite",
            "veo-3.1-fast-generate-001",
            "lyria-3-pro-preview",
            "sonar-deep-research",
            "openai-deep-research",
        ]
        self.assertTrue(all(not is_conversation_model(item) for item in excluded))
        self.assertTrue(is_conversation_model("DeepSeek-V3.2"))
        self.assertTrue(is_conversation_model("sonar-reasoning-pro"))

    def test_reasoning_is_clamped(self):
        model = Model(
            id="test",
            display_name="Test",
            provider="test",
            thinking_levels=("low", "medium", "high"),
            default_thinking_level="medium",
        )
        self.assertEqual(map_reasoning("ultra", model), "high")
        self.assertEqual(map_reasoning("low", model), "low")
        self.assertIsNone(map_reasoning("none", model))
        self.assertEqual(map_reasoning(None, model), "medium")

    def test_parses_modern_and_legacy_streams(self):
        lines = [
            b'data: {"type":"text-delta","id":"0","delta":"hello"}\n',
            b'0:" world"\n',
            b'data: [DONE]\n',
        ]
        events = list(extract_stream_events(lines))
        self.assertEqual("".join(text_delta(event) for event in events), "hello world")
        self.assertEqual(events[-1]["type"], "done")


if __name__ == "__main__":
    unittest.main()

