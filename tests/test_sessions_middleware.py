import os
import tempfile
import unittest

from hermes_aipass.middleware import attach_session_header
from hermes_aipass.bridge import _normalize_model_id
from hermes_aipass.prompting import message_fingerprint
from hermes_aipass.sessions import SessionStore


class SessionMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_home = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = self.temp.name

    def tearDown(self):
        if self.original_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = self.original_home
        self.temp.cleanup()

    def test_session_delta_skips_previous_assistant(self):
        first = [{"role": "system", "content": "system"}, {"role": "user", "content": "one"}]
        response = {"role": "assistant", "content": "answer"}
        store = SessionStore()
        store.complete(
            "session-a",
            conversation_id="chat-1",
            model="gpt-5.6-sol",
            request_messages=first,
            response_message=response,
        )
        current = first + [response, {"role": "user", "content": "two"}]
        session, tail, bootstrap = store.prepare("session-a", current)
        self.assertFalse(bootstrap)
        self.assertEqual(session.conversation_id, "chat-1")
        self.assertEqual(tail, [{"role": "user", "content": "two"}])

    def test_history_rewrite_bootstraps_new_chat(self):
        store = SessionStore()
        store.complete(
            "session-a",
            conversation_id="chat-1",
            model="model",
            request_messages=[{"role": "user", "content": "old"}],
            response_message={"role": "assistant", "content": "answer"},
        )
        _, _, bootstrap = store.prepare("session-a", [{"role": "user", "content": "compressed"}])
        self.assertTrue(bootstrap)

    def test_state_contains_fingerprints_not_transcript(self):
        store = SessionStore()
        message = {"role": "user", "content": "private transcript marker"}
        store.complete(
            "session-a",
            conversation_id="chat-1",
            model="model",
            request_messages=[message],
            response_message={"role": "assistant", "content": "secret answer"},
        )
        with open(os.path.join(self.temp.name, "aipass", "sessions.json"), encoding="utf-8") as handle:
            state = handle.read()
        self.assertNotIn("private transcript marker", state)
        self.assertNotIn("secret answer", state)
        self.assertIn(message_fingerprint(message), state)

    def test_middleware_only_modifies_aipass(self):
        result = attach_session_header(
            request={"model": "gpt"},
            provider="aipass",
            base_url="http://127.0.0.1:8792/v1",
            session_id="session-123",
        )
        self.assertEqual(result["request"]["extra_headers"]["X-AiPASS-Session-Id"], "session-123")
        self.assertIsNone(
            attach_session_header(
                request={"model": "gpt"},
                provider="openrouter",
                base_url="https://openrouter.ai/api/v1",
                session_id="session-123",
            )
        )

    def test_provider_prefixed_model_is_normalized(self):
        self.assertEqual(_normalize_model_id("aipass/gemini-3.1-flash-lite"), "gemini-3.1-flash-lite")
        self.assertEqual(_normalize_model_id("gpt-5.6-sol"), "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
