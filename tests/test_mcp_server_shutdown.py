import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from core import mcp_server


class McpServerShutdownTests(unittest.TestCase):
    def tearDown(self):
        if hasattr(mcp_server.app.state, "shutdown_handler"):
            delattr(mcp_server.app.state, "shutdown_handler")

    def test_shutdown_uses_registered_sidecar_handler(self):
        handler = Mock()
        mcp_server.app.state.shutdown_handler = handler

        with patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
            response = TestClient(mcp_server.app).post(
                "/shutdown",
                headers={"X-SoulDrive-Token": "test-token"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "shutdown_requested")
        handler.assert_called_once_with("api shutdown requested")


if __name__ == "__main__":
    unittest.main()
