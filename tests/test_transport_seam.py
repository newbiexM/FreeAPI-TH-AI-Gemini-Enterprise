import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

from hermes_aipass.auth import AuthState, KeychainStore
from hermes_aipass.models import Model


class TestTransportSeam(unittest.TestCase):
    def setUp(self):
        self.mock_auth = AuthState(cookies={"__Host-C_OSES": "mock_val", "__Secure-ai_passport_auth.session_token": "mock_token"})
        self.load_patch = patch.object(KeychainStore, "load", return_value=self.mock_auth)
        self.load_patch.start()

    def tearDown(self):
        self.load_patch.stop()

    def test_transport_factory_selection(self):
        from hermes_aipass.transport import Transport, get_transport
        from hermes_aipass.client import AiPassClient
        from hermes_aipass.discovery_engine import DiscoveryEngineTransport

        # Explicit 'web' selects AiPassClient
        web_transport = get_transport("web")
        self.assertIsInstance(web_transport, AiPassClient)
        self.assertIsInstance(web_transport, Transport)

        # Explicit 'discovery_engine' or 'enterprise' selects DiscoveryEngineTransport
        ge_transport = get_transport("discovery_engine")
        self.assertIsInstance(ge_transport, DiscoveryEngineTransport)
        self.assertIsInstance(ge_transport, Transport)

        # Environment variable override
        with patch.dict(os.environ, {"AIPASS_TRANSPORT": "discovery_engine"}):
            env_transport = get_transport()
            self.assertIsInstance(env_transport, DiscoveryEngineTransport)

    def test_discovery_engine_transport_models(self):
        from hermes_aipass.discovery_engine import DiscoveryEngineTransport
        transport = DiscoveryEngineTransport()
        models = transport.models()
        self.assertTrue(len(models) >= 4)
        model_ids = [m.id for m in models]
        self.assertIn("gemini-4-argon", model_ids)
        self.assertIn("gemini-3.8-flash", model_ids)

        # Model lookup with normalization
        m = transport.model("gemini-enterprise/gemini-4-argon")
        self.assertEqual(m.id, "gemini-4-argon")

    def test_discovery_engine_create_conversation_and_stream(self):
        from hermes_aipass.discovery_engine import DiscoveryEngineTransport

        transport = DiscoveryEngineTransport(tenant_id="test-cid-1234", csesidx="123456789")
        self.assertEqual(transport.tenant_id, "test-cid-1234")
        self.assertEqual(transport.csesidx, "123456789")

        # Mock _get_jwt response for getoxsrf and widgetCreateSession
        mock_oxsrf_resp = MagicMock()
        mock_oxsrf_resp.read.return_value = b")]}'\n{\"xsrfToken\": \"mock_xsrf_token_32_bytes_long_1234567890=\", \"keyId\": \"mock_key_id\"}"

        mock_session_resp = MagicMock()
        mock_session_resp.read.return_value = json.dumps({
            "session": {"name": "collections/default_collection/engines/thai-citizen-app-68/sessions/mock-session-123"}
        }).encode("utf-8")

        # Mock Discovery Engine widgetStreamAssist chunk response
        mock_stream_data = json.dumps([
            {
                "streamAssistResponse": {
                    "answer": {
                        "replies": [
                            {"groundedContent": {"content": {"role": "model", "text": "Hello from Google "}}}
                        ]
                    }
                }
            },
            {
                "streamAssistResponse": {
                    "answer": {
                        "replies": [
                            {"groundedContent": {"content": {"role": "model", "text": "Gemini Enterprise!"}}}
                        ]
                    }
                }
            }
        ]).encode("utf-8")

        mock_assist_resp = MagicMock()
        chunks = [mock_stream_data[:100], mock_stream_data[100:], b""]
        mock_assist_resp.read.side_effect = chunks

        with patch.object(transport, "_get_jwt", return_value="mock.jwt.token"):
            with patch.object(transport, "_open") as mock_open:
                mock_open.return_value = mock_session_resp
                cid = transport.create_conversation("Hello Google Discovery Engine", "gemini-3.8-flash")
                self.assertEqual(cid, "collections/default_collection/engines/thai-citizen-app-68/sessions/mock-session-123")

                mock_open.return_value = mock_assist_resp
                deltas = list(transport.send_stream(cid, "How are you?", "gemini-3.8-flash"))
                self.assertEqual(deltas, [
                    "Hello from Google ",
                    "Gemini Enterprise!",
                ])

