import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.paths import resolve_model_path


class PathResolutionTests(unittest.TestCase):
    def test_resolve_model_path_prefers_configured_model_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            model_file = model_dir / "model.gguf"
            model_file.write_text("model", encoding="utf-8")

            with patch.dict(os.environ, {"SOULDRIVE_MODEL_DIR": str(model_dir)}):
                resolved = resolve_model_path("model.gguf")

        self.assertEqual(resolved, str(model_file))

    def test_resolve_model_path_can_use_workspace_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "SoulDrive"
            model_dir = workspace / "models"
            model_dir.mkdir(parents=True)
            model_file = model_dir / "model.gguf"
            model_file.write_text("model", encoding="utf-8")

            resolved = resolve_model_path("model.gguf", workspace_path=str(workspace))

        self.assertEqual(resolved, str(model_file))


if __name__ == "__main__":
    unittest.main()
