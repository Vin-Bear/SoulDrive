import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.hardware_watcher import UDriveWatcher
from core.workspace import SoulDriveWorkspace


class HardwareWatcherTests(unittest.TestCase):
    def test_choose_souldrive_prefers_manifest_marked_drive(self):
        with tempfile.TemporaryDirectory() as first_drive, tempfile.TemporaryDirectory() as second_drive:
            SoulDriveWorkspace.from_drive(second_drive).ensure()
            watcher = UDriveWatcher()

            chosen = watcher.choose_souldrive([first_drive, second_drive])

        self.assertEqual(chosen, second_drive)

    def test_choose_souldrive_allows_first_time_initialization(self):
        with tempfile.TemporaryDirectory() as drive:
            watcher = UDriveWatcher()

            chosen = watcher.choose_souldrive([drive])

        self.assertEqual(chosen, drive)

    def test_prepare_workspace_imports_root_pdfs_on_first_use(self):
        with tempfile.TemporaryDirectory() as drive:
            root_pdf = Path(drive) / "paper.pdf"
            root_pdf.write_bytes(b"%PDF root")
            watcher = UDriveWatcher()

            workspace = watcher.prepare_workspace(drive)

            imported_pdf = Path(workspace.papers_path) / "paper.pdf"
            self.assertTrue(imported_pdf.exists())
            self.assertEqual(imported_pdf.read_bytes(), b"%PDF root")

    def test_prepare_workspace_copies_missing_model_assets_to_drive_workspace(self):
        with tempfile.TemporaryDirectory() as drive, tempfile.TemporaryDirectory() as model_root:
            source_root = Path(model_root)
            (source_root / "qwen2.5-3b-instruct-q4_k_m.gguf").write_text("chat model", encoding="utf-8")
            embedding_source = source_root / "bge-small-zh-v1.5"
            embedding_source.mkdir()
            (embedding_source / "config.json").write_text("embedding config", encoding="utf-8")
            watcher = UDriveWatcher()

            with patch.dict("os.environ", {"SOULDRIVE_MODEL_DIR": str(source_root)}):
                workspace = watcher.prepare_workspace(drive)

            workspace_models = Path(workspace.models_path)
            chat_destination = workspace_models / "qwen2.5-3b-instruct-q4_k_m.gguf"
            embedding_config_destination = workspace_models / "bge-small-zh-v1.5" / "config.json"
            self.assertTrue(chat_destination.exists())
            self.assertTrue(embedding_config_destination.exists())
            self.assertEqual(
                chat_destination.read_text(encoding="utf-8"),
                "chat model",
            )
            self.assertEqual(
                embedding_config_destination.read_text(encoding="utf-8"),
                "embedding config",
            )


if __name__ == "__main__":
    unittest.main()
