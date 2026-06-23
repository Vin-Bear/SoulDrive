import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.gpu_runtime import GpuAccelerationStrategy
from core.model_runtime import (
    LlamaRuntimeConfig,
    llama_runtime_config,
    load_llama_with_gpu_fallback,
    model_runtime_diagnostics,
    preferred_chat_model_filename,
    resolve_chat_model_path,
)


CPU_STRATEGY = GpuAccelerationStrategy(
    n_gpu_layers=0,
    mode="cpu",
    device_name=None,
    vendor=None,
    memory_mb=None,
    reason="test cpu fallback",
    detection_source=None,
    backend_available=False,
)


class ModelRuntimeTests(unittest.TestCase):
    def tearDown(self):
        for name in (
            "SOULDRIVE_CHAT_MODEL",
            "SOULDRIVE_LLAMA_GPU_LAYERS",
            "SOULDRIVE_LLAMA_CHAT_CTX",
            "SOULDRIVE_LLAMA_MAX_TOKENS",
            "SOULDRIVE_MODEL_DIR",
        ):
            os.environ.pop(name, None)

    def test_llama_runtime_config_is_bounded(self):
        with patch.dict(os.environ, {
            "SOULDRIVE_LLAMA_GPU_LAYERS": "999",
            "SOULDRIVE_LLAMA_CHAT_CTX": "999999",
            "SOULDRIVE_LLAMA_MAX_TOKENS": "1",
        }):
            config = llama_runtime_config()

        self.assertEqual(config.n_gpu_layers, 128)
        self.assertEqual(config.chat_n_ctx, 32768)
        self.assertEqual(config.max_tokens, 128)

    @patch("core.model_runtime.llama_gpu_strategy", return_value=CPU_STRATEGY)
    def test_llama_runtime_config_defaults_to_demo_safe_context(self, _strategy):
        config = llama_runtime_config()

        self.assertEqual(config.chat_n_ctx, 8192)
        self.assertEqual(config.graph_n_ctx, 8192)
        self.assertEqual(config.max_tokens, 900)

    def test_preferred_chat_model_keeps_standard_3b_default_when_7b_is_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "qwen2.5-3b-instruct-q4_k_m.gguf").write_text("model", encoding="utf-8")
            (model_dir / "qwen2.5-7b-instruct-q4_k_m.gguf").write_text("model", encoding="utf-8")

            with patch.dict(os.environ, {"SOULDRIVE_MODEL_DIR": str(model_dir)}):
                model_filename = preferred_chat_model_filename()

        self.assertEqual(model_filename, "qwen2.5-3b-instruct-q4_k_m.gguf")

    def test_preferred_chat_model_falls_back_to_3b(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "qwen2.5-3b-instruct-q4_k_m.gguf").write_text("model", encoding="utf-8")

            with patch.dict(os.environ, {"SOULDRIVE_MODEL_DIR": str(model_dir)}):
                model_filename = preferred_chat_model_filename()

        self.assertEqual(model_filename, "qwen2.5-3b-instruct-q4_k_m.gguf")

    def test_llama_runtime_config_auto_gpu_uses_strategy(self):
        strategy = GpuAccelerationStrategy(
            n_gpu_layers=16,
            mode="cuda_auto",
            device_name="NVIDIA RTX",
            vendor="NVIDIA",
            memory_mb=6144,
            reason="test auto",
            detection_source="nvidia-smi",
            backend_available=True,
        )

        with patch("core.model_runtime.llama_gpu_strategy", return_value=strategy):
            config = llama_runtime_config()

        self.assertEqual(config.n_gpu_layers, 16)
        self.assertEqual(config.gpu_mode, "cuda_auto")
        self.assertEqual(config.gpu_device_name, "NVIDIA RTX")

    def test_resolve_chat_model_uses_configured_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            model_file = model_dir / "custom.gguf"
            model_file.write_text("model", encoding="utf-8")

            with patch.dict(os.environ, {
                "SOULDRIVE_MODEL_DIR": str(model_dir),
                "SOULDRIVE_CHAT_MODEL": "custom.gguf",
            }):
                resolved = resolve_chat_model_path()

        self.assertEqual(resolved, str(model_file))

    def test_model_runtime_diagnostics_reports_missing_and_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {
                "SOULDRIVE_MODEL_DIR": temp_dir,
                "SOULDRIVE_CHAT_MODEL": "missing-chat-model.gguf",
            }):
                report = model_runtime_diagnostics()

        self.assertFalse(report["ready"])
        self.assertIn("config", report)
        self.assertFalse(report["chat_model"]["ready"])

    def test_load_llama_with_gpu_fallback_smoke_tests_generation(self):
        class FakeLlama:
            instances = []

            def __init__(self, model_path, n_gpu_layers, n_ctx, verbose):
                _ = model_path
                _ = n_ctx
                _ = verbose
                self.n_gpu_layers = n_gpu_layers
                FakeLlama.instances.append(self)

            def create_chat_completion(self, messages, max_tokens, temperature):
                _ = messages
                _ = max_tokens
                _ = temperature
                return {"choices": [{"message": {"content": "O"}}]}

            def close(self):
                self.closed = True

            def reset(self):
                self.reset_called = True

        config = LlamaRuntimeConfig(
            model_filename="model.gguf",
            n_gpu_layers=24,
            gpu_mode="cuda_auto",
        )

        with patch("core.model_runtime.Llama", FakeLlama, create=True), patch(
            "core.model_runtime._isolated_gpu_smoke_passes",
            return_value={"ok": False, "reason": "cuda smoke failed"},
        ), patch.dict("sys.modules", {"llama_cpp": type("M", (), {"Llama": FakeLlama})}):
            result = load_llama_with_gpu_fallback("model.gguf", config, n_ctx=1024)

        self.assertEqual(result.effective_config.n_gpu_layers, 0)
        self.assertEqual(result.effective_config.gpu_mode, "cpu_fallback")
        self.assertIn("cuda smoke failed", result.fallback_error)
        self.assertEqual(FakeLlama.instances[-1].n_gpu_layers, 0)


if __name__ == "__main__":
    unittest.main()
