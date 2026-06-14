import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import core.runtime_state as runtime_state
from core.diagnostics import product_diagnostics
from core.license import LicenseStatus


class DiagnosticsTests(unittest.TestCase):
    def test_product_diagnostics_reports_local_runtime_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _create_minimal_project_root(root)

            with patch.dict(os.environ, {
                "SOULDRIVE_APP_ROOT": str(root),
                "SOULDRIVE_MODEL_DIR": str(root / "models"),
            }):
                report = product_diagnostics()

        self.assertIn("checks", report)
        self.assertTrue(report["checks"]["policy"])
        self.assertTrue(report["checks"]["workspace"])
        self.assertTrue(report["checks"]["audit_chain"])
        self.assertTrue(report["models"]["ready"])
        self.assertNotIn("audit_path", report["audit"])
        self.assertNotIn("package", report)
        self.assertNotIn("release", report)
        self.assertFalse(report["license"]["valid"])

    def test_product_diagnostics_redacts_license_source_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            license_path = root / "config" / "license.json"
            _create_minimal_project_root(root)

            with patch.dict(os.environ, {
                "SOULDRIVE_APP_ROOT": str(root),
                "SOULDRIVE_MODEL_DIR": str(root / "models"),
            }), patch(
                "core.diagnostics.verify_license_for_workspace",
                return_value=LicenseStatus(
                    True,
                    "PRO",
                    "license verified",
                    source=str(license_path),
                    license_id="lic-1",
                ),
            ):
                report = product_diagnostics()

        self.assertEqual(report["license"]["source"], "license.json")
        self.assertNotIn(str(root), json.dumps(report["license"], ensure_ascii=False))

    def test_product_diagnostics_requires_mounted_unlocked_workspace_for_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _create_minimal_project_root(root)
            state_path = str(root / "runtime_state.json")

            with patch.dict(os.environ, {
                "SOULDRIVE_APP_ROOT": str(root),
                "SOULDRIVE_MODEL_DIR": str(root / "models"),
            }), patch.object(runtime_state, "STATE_PATH", state_path):
                report = product_diagnostics()

        self.assertFalse(report["checks"]["runtime_unlocked"])
        self.assertFalse(report["ready"])


def _create_minimal_project_root(root: Path):
    (root / "models" / "bge-small-zh-v1.5" / "1_Pooling").mkdir(parents=True)
    (root / "models" / "qwen2.5-3b-instruct-q4_k_m.gguf").write_text("model", encoding="utf-8")
    for relative_path in (
        "bge-small-zh-v1.5/config.json",
        "bge-small-zh-v1.5/model.safetensors",
        "bge-small-zh-v1.5/tokenizer.json",
        "bge-small-zh-v1.5/vocab.txt",
        "bge-small-zh-v1.5/1_Pooling/config.json",
    ):
        (root / "models" / relative_path).write_text("{}", encoding="utf-8")
    (root / "config").mkdir()
    (root / "config" / "enterprise-policy.json").write_text(json.dumps({
        "require_signed_license": False,
        "model_manifest_required": True,
    }), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
