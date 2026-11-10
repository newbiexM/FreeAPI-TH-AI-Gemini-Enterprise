import unittest

from hermes_aipass.auth import (
    AuthState,
    DISCOVERY_ENGINE_COOKIES,
    state_from_cdp_cookies,
    validate_auth,
)


class TestEnterpriseAuth(unittest.TestCase):
    def test_state_from_cdp_cookies_supports_discovery_engine(self):
        cookies = [
            {
                "name": "__Host-C_OSES",
                "value": "mock_google_session_token_12345",
                "domain": "vertexaisearch.cloud.google",
                "expires": 1893456000,
            },
            {
                "name": "__Secure-C_SES",
                "value": "mock_secure_ses_67890",
                "domain": "vertexaisearch.cloud.google",
                "expires": 1893456000,
            },
        ]
        state = state_from_cdp_cookies(cookies)
        self.assertIn("__Host-C_OSES", state.cookies)
        self.assertEqual(state.cookies["__Host-C_OSES"], "mock_google_session_token_12345")
        self.assertIn("__Secure-C_SES", state.cookies)

    def test_state_from_cdp_cookies_supports_unified_multi_domain(self):
        cookies = [
            {
                "name": "__Secure-ai_passport_auth.session_token",
                "value": "aipass_token_abc",
                "domain": "de.aipass.net",
                "expires": 1893456000,
            },
            {
                "name": "__Host-C_OSES",
                "value": "mock_google_token_xyz",
                "domain": "vertexaisearch.cloud.google",
                "expires": 1893456000,
            },
        ]
        state = state_from_cdp_cookies(cookies)
        self.assertIn("__Secure-ai_passport_auth.session_token", state.cookies)
        self.assertIn("__Host-C_OSES", state.cookies)

    def test_validate_auth_with_discovery_engine_cookies(self):
        state = AuthState(
            cookies={"__Host-C_OSES": "valid_token", "__Secure-C_SES": "valid_ses"},
            expires_at="2030-01-01T00:00:00Z",
        )
        result = validate_auth(state)
        self.assertTrue(result.get("valid"))
