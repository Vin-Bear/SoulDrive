import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import core.runtime_state as runtime_state
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

    def test_prepare_workspace_does_not_auto_import_root_pdfs(self):
        with tempfile.TemporaryDirectory() as drive:
            root_pdf = Path(drive) / "paper.pdf"
            root_pdf.write_bytes(b"%PDF root")
            watcher = UDriveWatcher()

            workspace = watcher.prepare_workspace(drive)

            imported_pdf = Path(workspace.papers_path) / "paper.pdf"
            self.assertFalse(imported_pdf.exists())

    def test_monitor_loop_does_not_auto_start_indexer_on_mount(self):
        original_state_path = runtime_state.STATE_PATH
        with tempfile.TemporaryDirectory() as drive:
            watcher = UDriveWatcher()
            watcher.is_running = True

            def stop_after_first_iteration(_seconds):
                watcher.is_running = False

            try:
                with patch.object(watcher, "get_removable_drives", return_value=[drive]), patch.object(
                    watcher.authenticator,
                    "verify_environment",
                    return_value=("PRO", "SN-1"),
                ), patch(
                    "core.hardware_watcher.authorization_from_hardware_and_license",
                    return_value=(
                        "HARDWARE_PLUS_PASSWORD",
                        "SN-1",
                        SimpleNamespace(reason="valid", level="HARDWARE_PLUS_PASSWORD"),
                    ),
                ), patch(
                    "core.hardware_watcher.sync_workspace_models",
                    return_value=[],
                ), patch.object(
                    watcher,
                    "_notify_runtime_api",
                ), patch.object(
                    watcher,
                    "_start_indexer_worker",
                ) as start_indexer, patch(
                    "core.hardware_watcher.time.sleep",
                    side_effect=stop_after_first_iteration,
                ):
                    watcher.monitor_loop()
            finally:
                runtime_state.STATE_PATH = original_state_path

        start_indexer.assert_not_called()

    def test_prepare_workspace_copies_missing_model_assets_to_drive_workspace(self):
        with tempfile.TemporaryDirectory() as drive, tempfile.TemporaryDirectory() as model_root:
            source_root = Path(model_root)
            (source_root / "qwen2.5-3b-instruct-q4_k_m.gguf").write_text("chat model", encoding="utf-8")
            embedding_source = source_root / "bge-small-zh-v1.5"
            embedding_source.mkdir()
            (embedding_source / "config.json").write_text("embedding config", encoding="utf-8")
            reranker_source = source_root / "mmarco-mMiniLMv2-L6-H384-v1"
            reranker_source.mkdir()
            (reranker_source / "config.json").write_text("reranker config", encoding="utf-8")
            watcher = UDriveWatcher()

            with patch.dict("os.environ", {"SOULDRIVE_MODEL_DIR": str(source_root)}):
                workspace = watcher.prepare_workspace(drive)

            workspace_models = Path(workspace.models_path)
            chat_destination = workspace_models / "qwen2.5-3b-instruct-q4_k_m.gguf"
            embedding_config_destination = workspace_models / "bge-small-zh-v1.5" / "config.json"
            reranker_config_destination = workspace_models / "mmarco-mMiniLMv2-L6-H384-v1" / "config.json"
            self.assertTrue(chat_destination.exists())
            self.assertTrue(embedding_config_destination.exists())
            self.assertTrue(reranker_config_destination.exists())
            self.assertEqual(
                chat_destination.read_text(encoding="utf-8"),
                "chat model",
            )
            self.assertEqual(
                embedding_config_destination.read_text(encoding="utf-8"),
                "embedding config",
            )
            self.assertEqual(
                reranker_config_destination.read_text(encoding="utf-8"),
                "reranker config",
            )


if __name__ == "__main__":
    unittest.main()
