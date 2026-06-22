import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from core import mcp_server
import core.runtime_state as runtime_state
from core.workspace import SoulDriveWorkspace


class McpServerSecurityTests(unittest.TestCase):
    def test_security_init_requires_no_recovery_acknowledgement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                response = TestClient(mcp_server.app).post(
                    "/security/init",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase", "acknowledge_no_recovery": False},
                )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "no recovery acknowledgement required")

    def test_security_init_and_unlock_enable_sensitive_workflows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                client = TestClient(mcp_server.app)
                init_response = client.post(
                    "/security/init",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase", "acknowledge_no_recovery": True},
                )
                lock_response = client.post(
                    "/security/lock",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={},
                )
                unlock_response = client.post(
                    "/security/unlock",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase"},
                )
                state = runtime_state.get_runtime_state()

        self.assertEqual(init_response.status_code, 200)
        self.assertEqual(lock_response.status_code, 200)
        self.assertEqual(unlock_response.status_code, 200)
        self.assertFalse(state["locked"])
        self.assertTrue(state["software_unlocked"])

    def test_security_unlock_rejects_wrong_passphrase(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict("os.environ", {"SOULDRIVE_API_TOKEN": "test-token"}, clear=False):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                client = TestClient(mcp_server.app)
                client.post(
                    "/security/init",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "secret-passphrase", "acknowledge_no_recovery": True},
                )
                response = client.post(
                    "/security/unlock",
                    headers={"X-SoulDrive-Token": "test-token"},
                    json={"passphrase": "wrong-passphrase"},
                )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"], "incorrect passphrase")


if __name__ == "__main__":
    unittest.main()
