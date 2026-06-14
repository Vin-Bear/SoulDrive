import tempfile
import unittest
from pathlib import Path

from core.workspace import SoulDriveWorkspace, is_souldrive_workspace


class WorkspaceTests(unittest.TestCase):
    def test_workspace_creates_drive_workspace_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir)
            workspace.ensure()

            self.assertTrue((Path(temp_dir) / "SoulDrive" / "data" / "papers").exists())
            self.assertTrue(workspace.chroma_path.endswith(str(Path("SoulDrive") / "index" / "chroma")))
            self.assertTrue(workspace.graph_db_path.endswith(str(Path("SoulDrive") / "index" / "knowledge_graph.sqlite")))
            self.assertTrue(workspace.audit_log_path.endswith(str(Path("SoulDrive") / "audit" / "audit_log.jsonl")))
            self.assertTrue(Path(workspace.manifest_path).exists())
            self.assertTrue(is_souldrive_workspace(temp_dir))

    def test_workspace_diagnose_reports_product_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            diagnostics = workspace.diagnose()

        self.assertTrue(diagnostics["ready"])
        self.assertEqual(diagnostics["root_name"], "SoulDrive")
        self.assertTrue(diagnostics["checks"]["manifest"])
        self.assertIn("free_bytes", diagnostics["disk"])
        self.assertIn("minimum_free_bytes", diagnostics["disk"])


if __name__ == "__main__":
    unittest.main()
