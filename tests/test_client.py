import email.message
import io
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from hermes_aipass.auth import AuthError
from hermes_aipass.client import AiPassClient, ProtocolError


class ClientErrorTests(unittest.TestCase):
    @staticmethod
    def _client():
        client = object.__new__(AiPassClient)
        client.timeout = 1
        return client

    @staticmethod
    def _http_error(status, body=b""):
        return HTTPError(
            "https://de.aipass.net/chat.data",
            status,
            "error",
            email.message.Message(),
            io.BytesIO(body),
        )

    def test_401_is_an_authentication_error(self):
        with patch("hermes_aipass.client.urlopen", side_effect=self._http_error(401)):
            with self.assertRaisesRegex(AuthError, "session expired"):
                self._client()._open(Request("https://de.aipass.net/chat.data"))

    def test_403_is_an_access_error_not_expired_auth(self):
        with patch("hermes_aipass.client.urlopen", side_effect=self._http_error(403)):
            with self.assertRaisesRegex(ProtocolError, "account access or paid credits"):
                self._client()._open(Request("https://de.aipass.net/chat.data"))

    def test_422_rai_violation_raises_content_policy_error(self):
        from hermes_aipass.client import ContentPolicyViolationError
        body = b'{"error":"Chat service error: Unprocessable Entity","code":"CHAT_RAI_VIOLATION","category":"SCAM_FRAUD"}'
        with patch("hermes_aipass.client.urlopen", side_effect=self._http_error(422, body)):
            with self.assertRaises(ContentPolicyViolationError) as ctx:
                self._client()._open(Request("https://de.aipass.net/chat.data"))
            self.assertEqual(ctx.exception.category, "SCAM_FRAUD")
            self.assertEqual(ctx.exception.code, 422)


    def test_legacy_or_unknown_model_falls_back_to_default(self):
        client = self._client()
        model = client.model("glm-5.3")
        self.assertEqual(model.id, "gemini-3.8-flash")
        model2 = client.model("gpt-6.1-sol")
        self.assertEqual(model2.id, "gemini-3.8-flash")


if __name__ == "__main__":
    unittest.main()
