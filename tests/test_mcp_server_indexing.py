import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from core import mcp_server
import core.runtime_state as runtime_state
from core.workspace import SoulDriveWorkspace


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
                response = TestClient(mcp_server.app).post(
                    "/index/run",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["status"], "already_running")

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
