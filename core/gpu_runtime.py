import json
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


_RUNTIME_GPU_FAILURE: str | None = None


@dataclass(frozen=True)
class GpuDevice:
    name: str
    vendor: str
    memory_mb: int | None = None
    source: str = "unknown"

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "vendor": self.vendor,
            "memory_mb": self.memory_mb,
            "source": self.source,
        }


@dataclass(frozen=True)
class GpuAccelerationStrategy:
    n_gpu_layers: int
    mode: str
    device_name: str | None
    vendor: str | None
    memory_mb: int | None
    reason: str
    detection_source: str | None
    backend_available: bool

    def public_dict(self) -> dict[str, Any]:
        return {
            "n_gpu_layers": self.n_gpu_layers,
            "mode": self.mode,
            "device_name": self.device_name,
            "vendor": self.vendor,
            "memory_mb": self.memory_mb,
            "reason": self.reason,
            "detection_source": self.detection_source,
            "backend_available": self.backend_available,
        }


def clear_gpu_detection_cache():
    global _RUNTIME_GPU_FAILURE
    _RUNTIME_GPU_FAILURE = None
    detect_gpu_devices.cache_clear()


def record_gpu_runtime_failure(reason: str):
    global _RUNTIME_GPU_FAILURE
    _RUNTIME_GPU_FAILURE = str(reason or "GPU runtime smoke test failed")[:240]


@lru_cache(maxsize=1)
def detect_gpu_devices() -> tuple[GpuDevice, ...]:
    devices: list[GpuDevice] = []
    devices.extend(_detect_nvidia_smi())
    devices.extend(_detect_windows_video_controllers())
    return tuple(_dedupe_devices(devices))


def llama_gpu_strategy(model_filename: str) -> GpuAccelerationStrategy:
    devices = detect_gpu_devices()
    cuda_backend = llama_cuda_backend_available()
    nvidia_device = _best_nvidia_device(devices)
    if nvidia_device is None:
        if devices:
            device = devices[0]
            return GpuAccelerationStrategy(
                n_gpu_layers=0,
                mode="cpu",
                device_name=device.name,
                vendor=device.vendor,
                memory_mb=device.memory_mb,
                reason="detected GPU is not supported by the packaged llama.cpp CUDA backend",
                detection_source=device.source,
                backend_available=cuda_backend,
            )
        return GpuAccelerationStrategy(
            n_gpu_layers=0,
            mode="cpu",
            device_name=None,
            vendor=None,
            memory_mb=None,
            reason="no local GPU detected",
            detection_source=None,
            backend_available=cuda_backend,
        )

    if _is_known_cuda_incompatible(nvidia_device.name):
        return GpuAccelerationStrategy(
            n_gpu_layers=0,
            mode="cpu",
            device_name=nvidia_device.name,
            vendor=nvidia_device.vendor,
            memory_mb=nvidia_device.memory_mb,
            reason="detected RTX 50-series GPU is not enabled for the packaged llama.cpp CUDA backend; using CPU runtime",
            detection_source=nvidia_device.source,
            backend_available=cuda_backend,
        )

    if not cuda_backend:
        return GpuAccelerationStrategy(
            n_gpu_layers=0,
            mode="cpu",
            device_name=nvidia_device.name,
            vendor=nvidia_device.vendor,
            memory_mb=nvidia_device.memory_mb,
            reason="NVIDIA GPU detected, but packaged llama.cpp CUDA backend is unavailable",
            detection_source=nvidia_device.source,
            backend_available=False,
        )

    if _RUNTIME_GPU_FAILURE:
        return GpuAccelerationStrategy(
            n_gpu_layers=0,
            mode="cpu_fallback",
            device_name=nvidia_device.name,
            vendor=nvidia_device.vendor,
            memory_mb=nvidia_device.memory_mb,
            reason=f"GPU runtime disabled after local smoke test failure: {_RUNTIME_GPU_FAILURE}",
            detection_source=nvidia_device.source,
            backend_available=True,
        )

    layers = _recommended_gpu_layers(nvidia_device.memory_mb, model_filename)
    if layers <= 0:
        return GpuAccelerationStrategy(
            n_gpu_layers=0,
            mode="cpu",
            device_name=nvidia_device.name,
            vendor=nvidia_device.vendor,
            memory_mb=nvidia_device.memory_mb,
            reason="GPU memory is below the conservative auto-offload threshold",
            detection_source=nvidia_device.source,
            backend_available=True,
        )

    return GpuAccelerationStrategy(
        n_gpu_layers=layers,
        mode="cuda_auto",
        device_name=nvidia_device.name,
        vendor=nvidia_device.vendor,
        memory_mb=nvidia_device.memory_mb,
        reason="NVIDIA CUDA backend detected; using conservative llama.cpp layer offload",
        detection_source=nvidia_device.source,
        backend_available=True,
    )


def llama_cuda_backend_available() -> bool:
    try:
        import llama_cpp

        package_dir = Path(llama_cpp.__file__).resolve().parent
    except Exception:
        return False

    candidate_names = (
        "ggml-cuda.dll",
        "libggml-cuda.so",
        "libggml-cuda.dylib",
    )
    search_dirs = [package_dir / "lib", package_dir]
    return any((directory / name).exists() for directory in search_dirs for name in candidate_names)


def _detect_nvidia_smi() -> list[GpuDevice]:
    output = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout_seconds=2.0,
    )
    if not output:
        return []

    devices: list[GpuDevice] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or "," not in line:
            continue
        name, memory = line.rsplit(",", 1)
        devices.append(
            GpuDevice(
                name=name.strip(),
                vendor="NVIDIA",
                memory_mb=_parse_int(memory),
                source="nvidia-smi",
            )
        )
    return devices


def _detect_windows_video_controllers() -> list[GpuDevice]:
    if sys.platform != "win32":
        return []

    output = _run_command(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_VideoController | "
            "Select-Object -Property Name,AdapterRAM | ConvertTo-Json -Compress",
        ],
        timeout_seconds=3.0,
    )
    if not output:
        return []

    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return []

    rows = payload if isinstance(payload, list) else [payload]
    devices = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        devices.append(
            GpuDevice(
                name=name,
                vendor=_vendor_from_name(name),
                memory_mb=_adapter_ram_to_mb(row.get("AdapterRAM")),
                source="windows-video-controller",
            )
        )
    return devices


def _run_command(command: list[str], timeout_seconds: float = 2.0) -> str | None:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _dedupe_devices(devices: list[GpuDevice]) -> list[GpuDevice]:
    deduped: dict[str, GpuDevice] = {}
    for device in devices:
        key = _normalize_device_name(device.name)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = device
            continue
        existing_memory = existing.memory_mb or 0
        device_memory = device.memory_mb or 0
        if device_memory > existing_memory or existing.source != "nvidia-smi":
            deduped[key] = device
    return list(deduped.values())


def _best_nvidia_device(devices: tuple[GpuDevice, ...]) -> GpuDevice | None:
    nvidia_devices = [device for device in devices if device.vendor == "NVIDIA"]
    if not nvidia_devices:
        return None
    return max(nvidia_devices, key=lambda device: device.memory_mb or 0)


def _recommended_gpu_layers(memory_mb: int | None, model_filename: str) -> int:
    memory = memory_mb or 0
    model = (model_filename or "").lower()
    is_seven_b = "7b" in model
    if memory < 6144:
        return 0
    if memory >= 12000:
        return 24 if is_seven_b else 32
    if memory >= 8000:
        return 16 if is_seven_b else 24
    return 12 if is_seven_b else 16


def _is_known_cuda_incompatible(name: str) -> bool:
    normalized = _normalize_device_name(name)
    return any(marker in normalized for marker in ("rtx 50", "5060", "5070", "5080", "5090"))


def _vendor_from_name(name: str) -> str:
    normalized = name.lower()
    if "nvidia" in normalized or "geforce" in normalized or "quadro" in normalized or "rtx" in normalized:
        return "NVIDIA"
    if "amd" in normalized or "radeon" in normalized:
        return "AMD"
    if "intel" in normalized or "iris" in normalized or "uhd" in normalized:
        return "Intel"
    return "Unknown"


def _adapter_ram_to_mb(value: Any) -> int | None:
    parsed = _parse_int(value)
    if parsed is None or parsed <= 0:
        return None
    if parsed > 1024 * 1024:
        return max(1, int(parsed / (1024 * 1024)))
    return parsed


def _parse_int(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _normalize_device_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip().lower())
