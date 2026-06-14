import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ScriptTests(unittest.TestCase):
    def test_check_script_uses_windows_npm_entrypoint_without_audit(self):
        script = (ROOT / "scripts" / "check.ps1").read_text(encoding="utf-8")

        self.assertIn("npm.cmd run build", script)
        self.assertNotIn("npm run build", script)
        self.assertNotIn("npm audit", script)

    def test_audit_script_uses_runtime_dependency_audit(self):
        script = (ROOT / "scripts" / "audit.ps1").read_text(encoding="utf-8")

        self.assertIn("npm.cmd audit --omit=dev", script)

    def test_package_sidecar_collects_offline_document_runtime_assets(self):
        script_path = ROOT / "scripts" / "package-sidecar.ps1"

        self.assertTrue(script_path.exists())
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("--collect-all", script)
        self.assertIn("docling", script)
        self.assertIn("rapidocr", script)
        self.assertIn("--collect-all chromadb", script)
        self.assertIn("souldrive-ui\\src-tauri\\sidecars", script)

    def test_package_sidecar_prunes_only_nonessential_cuda_and_video_runtime_bloat(self):
        script = (ROOT / "scripts" / "package-sidecar.ps1").read_text(encoding="utf-8")

        self.assertIn("Remove-PortableSidecarBloat", script)
        self.assertNotIn('"_internal\\llama_cpp\\lib\\ggml-cuda.dll"', script)
        self.assertIn('"_internal\\llama_cpp\\lib\\ggml-cuda.lib"', script)
        self.assertIn('"_internal\\cublasLt64_12.dll"', script)
        self.assertIn("opencv_videoio_ffmpeg*.dll", script)

    def test_package_sidecar_includes_chromadb_dynamic_runtime_imports(self):
        script = (ROOT / "scripts" / "package-sidecar.ps1").read_text(encoding="utf-8")

        self.assertIn("--hidden-import chromadb.telemetry.product.posthog", script)
        self.assertIn("--hidden-import chromadb.api.rust", script)


if __name__ == "__main__":
    unittest.main()
