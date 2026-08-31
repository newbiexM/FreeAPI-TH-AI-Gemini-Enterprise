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
            {},
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


if __name__ == "__main__":
    unittest.main()
