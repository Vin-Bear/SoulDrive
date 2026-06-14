import unittest
from unittest.mock import patch

from core.gpu_runtime import (
    GpuDevice,
    clear_gpu_detection_cache,
    detect_gpu_devices,
    llama_gpu_strategy,
)


class GpuRuntimeTests(unittest.TestCase):
    def tearDown(self):
        clear_gpu_detection_cache()

    def test_detect_gpu_devices_reads_nvidia_smi_without_network(self):
        def fake_run(command, timeout_seconds=2.0):
            if command[0] == "nvidia-smi":
                return "NVIDIA GeForce RTX 5060 Laptop GPU, 8151"
            return None

        with patch("core.gpu_runtime._run_command", side_effect=fake_run):
            clear_gpu_detection_cache()
            devices = detect_gpu_devices()

        self.assertEqual(len(devices), 1)
        self.assertEqual(devices[0].vendor, "NVIDIA")
        self.assertEqual(devices[0].memory_mb, 8151)

    def test_llama_gpu_strategy_disables_known_incompatible_rtx_50_series(self):
        device = GpuDevice(
            name="NVIDIA GeForce RTX 5060 Laptop GPU",
            vendor="NVIDIA",
            memory_mb=8151,
            source="nvidia-smi",
        )

        with patch("core.gpu_runtime.detect_gpu_devices", return_value=[device]), patch(
            "core.gpu_runtime.llama_cuda_backend_available",
            return_value=True,
        ):
            strategy = llama_gpu_strategy("qwen2.5-3b-instruct-q4_k_m.gguf")

        self.assertEqual(strategy.mode, "cpu")
        self.assertEqual(strategy.n_gpu_layers, 0)
        self.assertIn("RTX 50-series", strategy.reason)
        self.assertIn("RTX 5060", strategy.device_name)

    def test_llama_gpu_strategy_uses_supported_nvidia_cuda_when_available(self):
        device = GpuDevice(
            name="NVIDIA GeForce RTX 4060 Laptop GPU",
            vendor="NVIDIA",
            memory_mb=8151,
            source="nvidia-smi",
        )

        with patch("core.gpu_runtime.detect_gpu_devices", return_value=[device]), patch(
            "core.gpu_runtime.llama_cuda_backend_available",
            return_value=True,
        ):
            strategy = llama_gpu_strategy("qwen2.5-3b-instruct-q4_k_m.gguf")

        self.assertEqual(strategy.mode, "cuda_auto")
        self.assertEqual(strategy.n_gpu_layers, 24)

    def test_llama_gpu_strategy_keeps_cpu_for_unsupported_gpu(self):
        device = GpuDevice(
            name="Intel(R) Iris Xe Graphics",
            vendor="Intel",
            memory_mb=4096,
            source="windows-video-controller",
        )

        with patch("core.gpu_runtime.detect_gpu_devices", return_value=[device]), patch(
            "core.gpu_runtime.llama_cuda_backend_available",
            return_value=True,
        ):
            strategy = llama_gpu_strategy("qwen2.5-3b-instruct-q4_k_m.gguf")

        self.assertEqual(strategy.mode, "cpu")
        self.assertEqual(strategy.n_gpu_layers, 0)


if __name__ == "__main__":
    unittest.main()
