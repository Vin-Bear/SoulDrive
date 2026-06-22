import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from core import mcp_server
import core.runtime_state as runtime_state
from core.parent_doc_store import ParentDocumentStore
from core.workspace import SoulDriveWorkspace


class McpServerPapersTests(unittest.TestCase):
    def test_documents_list_reads_only_workspace_documents_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            managed = Path(workspace.papers_path) / "managed.pdf"
            loose = Path(temp_dir) / "loose.pdf"
            managed.write_bytes(b"%PDF managed")
            loose.write_bytes(b"%PDF loose")

            store = ParentDocumentStore(workspace.parent_doc_path)
            try:
                store.upsert_parent(
                    "managed_parent_0",
                    "content",
                    {"source_filename": "managed.pdf"},
                )
            finally:
                store.close()

            with patch("core.mcp_server.current_workspace", return_value=workspace), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                response = TestClient(mcp_server.app).get(
                    "/documents/list",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document_count"], 1)
        self.assertEqual(payload["documents"][0]["name"], "managed.pdf")
        self.assertTrue(payload["documents"][0]["indexed"])

    def test_documents_import_copies_pdf_into_active_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            source_dir = Path(temp_dir) / "incoming"
            source_dir.mkdir()
            source_pdf = source_dir / "paper.pdf"
            source_pdf.write_bytes(b"%PDF imported")
            state_path = str(Path(temp_dir) / "runtime_state.json")

            with patch.object(runtime_state, "STATE_PATH", state_path), patch(
                "core.mcp_server.current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                response = TestClient(mcp_server.app).post(
                    "/documents/import",
                    json={"source_paths": [str(source_pdf)]},
                    headers={"X-SoulDrive-Token": "test-token"},
                )
                imported_pdf_exists = (Path(workspace.papers_path) / "paper.pdf").exists()

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["imported_count"], 1)
        self.assertEqual(payload["items"][0]["status"], "imported")
        self.assertEqual(payload["items"][0]["name"], "paper.pdf")
        self.assertTrue(imported_pdf_exists)

    def test_papers_list_keeps_legacy_response_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            legacy = Path(workspace.papers_path) / "legacy.pdf"
            legacy.write_bytes(b"%PDF legacy")

            with patch("core.mcp_server.current_workspace", return_value=workspace), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                response = TestClient(mcp_server.app).get(
                    "/papers/list",
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("paper_count", payload)
        self.assertIn("papers", payload)

    def test_papers_import_rejects_non_pdf_without_copying(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = SoulDriveWorkspace.from_drive(temp_dir).ensure()
            source_txt = Path(temp_dir) / "notes.txt"
            source_txt.write_text("not a pdf", encoding="utf-8")
            state_path = str(Path(temp_dir) / "runtime_state.json")

            with patch.object(runtime_state, "STATE_PATH", state_path), patch(
                "core.mcp_server.current_workspace",
                return_value=workspace,
            ), patch.dict(
                "os.environ",
                {"SOULDRIVE_API_TOKEN": "test-token"},
                clear=False,
            ):
                runtime_state.unlock_runtime("PRO", "SN-1", temp_dir, workspace.root_path)
                response = TestClient(mcp_server.app).post(
                    "/papers/import",
                    json={"source_paths": [str(source_txt)]},
                    headers={"X-SoulDrive-Token": "test-token"},
                )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["imported_count"], 0)
        self.assertEqual(payload["items"][0]["status"], "rejected")
        self.assertEqual(payload["items"][0]["error_code"], "UNSUPPORTED_FILE_TYPE")
        self.assertFalse((Path(workspace.papers_path) / "notes.txt").exists())


if __name__ == "__main__":
    unittest.main()
