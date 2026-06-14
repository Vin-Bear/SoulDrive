import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import indexer_worker
import core.runtime_state as runtime_state
from core.workspace import SoulDriveWorkspace


class IndexerWorkerTests(unittest.TestCase):
    def test_workspace_mode_writes_runtime_state_inside_workspace(self):
        original_state_path = runtime_state.STATE_PATH
        captured = {}

        class FakeIndexer:
            def sync_workspace(self, workspace, auth_level):
                captured["state_path"] = runtime_state.runtime_state_path()
                captured["workspace_path"] = workspace.root_path
                captured["auth_level"] = auth_level

            def close(self):
                pass

        try:
            runtime_state.STATE_PATH = None
            with tempfile.TemporaryDirectory() as temp_dir:
                workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()

                with patch("core.indexer_worker.DriveIndexer", return_value=FakeIndexer()):
                    exit_code = indexer_worker.main([
                        "--workspace-path",
                        workspace.root_path,
                        "--auth-level",
                        "LITE",
                    ])

                expected_state_path = str(Path(workspace.runtime_path) / "runtime_state.json")
        finally:
            runtime_state.STATE_PATH = original_state_path

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured["workspace_path"], workspace.root_path)
        self.assertEqual(captured["auth_level"], "LITE")
        self.assertEqual(captured["state_path"], expected_state_path)


if __name__ == "__main__":
    unittest.main()
