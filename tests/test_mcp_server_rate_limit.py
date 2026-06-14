import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from core import mcp_server
from core.enterprise_security import SlidingWindowRateLimiter


class McpServerRateLimitTests(unittest.TestCase):
    def setUp(self):
        self.original_rate_limiter = mcp_server.rate_limiter
        mcp_server.rate_limiter = SlidingWindowRateLimiter(limit=1)
        self.client = TestClient(mcp_server.app)

    def tearDown(self):
        mcp_server.rate_limiter = self.original_rate_limiter

    def test_health_checks_do_not_consume_rate_limit(self):
        responses = [self.client.get("/health") for _ in range(3)]

        self.assertEqual([response.status_code for response in responses], [200, 200, 200])

    def test_non_exempt_endpoint_still_uses_rate_limit(self):
        with patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
            first = self.client.get("/metrics", headers={"X-SoulDrive-Token": "test-token"})
            second = self.client.get("/metrics", headers={"X-SoulDrive-Token": "test-token"})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)


if __name__ == "__main__":
    unittest.main()
