import tempfile
import unittest
from base64 import b64decode
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from core import mcp_server
import core.runtime_state as runtime_state
from core.security_context import WORKSPACE_DATA_KEY_ENV, set_workspace_keys
from core.workspace import SoulDriveWorkspace
from core.workspace_crypto import initialize_keystore, unlock_keystore


class McpServerIndexingTests(unittest.TestCase):
    def setUp(self):
        self.original_indexer_process = mcp_server.indexer_process
        mcp_server.indexer_process = None

    def tearDown(self):
        mcp_server.indexer_process = self.original_indexer_process

    def test_index_run_starts_worker_for_active_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            fake_process = Mock()
            fake_process.poll.return_value = None

            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ), patch(
                "core.mcp_server.subprocess.Popen",
                return_value=fake_process,
            ) as popen:
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                runtime_state.mark_software_unlocked()
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )
                state = runtime_state.get_runtime_state()

        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["started"])
        self.assertEqual(state["indexing"]["status"], "queued")
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertIn("--workspace-path", command)
        self.assertIn(workspace.root_path, command)

    def test_index_run_passes_workspace_data_key_to_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            initialize_keystore(workspace, "secret-passphrase")
            keys = unlock_keystore(workspace, "secret-passphrase")
            set_workspace_keys(workspace.root_path, keys)
            fake_process = Mock()
            fake_process.poll.return_value = None

            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ), patch(
                "core.mcp_server.subprocess.Popen",
                return_value=fake_process,
            ) as popen:
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                runtime_state.mark_software_unlocked()
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 202)
        worker_env = popen.call_args.kwargs["env"]
        self.assertIn(WORKSPACE_DATA_KEY_ENV, worker_env)
        self.assertEqual(b64decode(worker_env[WORKSPACE_DATA_KEY_ENV].encode("ascii")), keys.workspace_data_key)

    def test_index_run_rejects_when_worker_is_already_running(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            running_process = Mock()
            running_process.poll.return_value = None
            mcp_server.indexer_process = running_process

            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                runtime_state.mark_software_unlocked()
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["status"], "already_running")

    def test_index_run_rejects_hardware_only_workspace_before_software_unlock(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()

            with patch.object(runtime_state, "STATE_PATH", state_path), patch.object(
                mcp_server,
                "current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 423)
        self.assertEqual(response.json()["reason"], "workspace passphrase required")

    def test_index_run_rejects_locked_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = str(Path(temp_dir) / "runtime_state.json")

            with patch.object(runtime_state, "STATE_PATH", state_path), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 423)
        self.assertEqual(response.json()["status"], "locked")


if __name__ == "__main__":
    unittest.main()
