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

        transport = DiscoveryEngineTransport(tenant_id="test-cid-1234")
        self.assertEqual(transport.tenant_id, "test-cid-1234")

        cid = transport.create_conversation("Hello Google Discovery Engine", "gemini-3.8-flash")
        self.assertTrue(cid.startswith("ge-"))

        # Mock Discovery Engine SSE response format
        mock_sse_lines = [
            b'data: {"answer": {"reply": "Hello from Google "}}\n\n',
            b'data: {"answer": {"thought": "Processing request in Vertex..."}}\n\n',
            b'data: {"answer": {"reply": "Gemini Enterprise!"}}\n\n',
            b'data: [DONE]\n\n',
        ]
        mock_resp = MagicMock()
        mock_resp.__iter__.return_value = iter(mock_sse_lines)
        mock_resp.headers = MagicMock()
        mock_resp.headers.get_all.return_value = []

        with patch.object(transport, "_open", return_value=mock_resp):
            deltas = list(transport.send_stream(cid, "How are you?", "gemini-3.8-flash"))
            self.assertEqual(deltas, [
                "Hello from Google ",
                "<thought>Processing request in Vertex...</thought>",
                "Gemini Enterprise!",
            ])

