import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from core.gpu_runtime import GpuAccelerationStrategy, llama_gpu_strategy, record_gpu_runtime_failure
from core.paths import model_search_dirs, resolve_model_path


DEFAULT_CHAT_MODEL = "qwen2.5-3b-instruct-q4_k_m.gguf"
PREFERRED_CHAT_MODELS = (
    "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    DEFAULT_CHAT_MODEL,
)
DEFAULT_EMBEDDING_MODEL = "bge-small-zh-v1.5"


@dataclass(frozen=True)
class LlamaRuntimeConfig:
    model_filename: str = DEFAULT_CHAT_MODEL
    n_gpu_layers: int = 0
    gpu_mode: str = "cpu"
    gpu_device_name: str | None = None
    gpu_vendor: str | None = None
    gpu_memory_mb: int | None = None
    gpu_reason: str = "CPU runtime"
    gpu_detection_source: str | None = None
    gpu_backend_available: bool = False
    chat_n_ctx: int = 8192
    graph_n_ctx: int = 4096
    temperature: float = 0.01
    top_p: float = 0.85
    repeat_penalty: float = 1.12
    max_tokens: int = 900
    graph_temperature: float = 0.1
    graph_max_tokens: int = 2048
    graph_repeat_penalty: float = 1.1

    def public_dict(self) -> dict[str, Any]:
        return {
            "model_filename": self.model_filename,
            "n_gpu_layers": self.n_gpu_layers,
            "gpu_mode": self.gpu_mode,
            "gpu_device_name": self.gpu_device_name,
            "gpu_vendor": self.gpu_vendor,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_reason": self.gpu_reason,
            "gpu_detection_source": self.gpu_detection_source,
            "gpu_backend_available": self.gpu_backend_available,
            "chat_n_ctx": self.chat_n_ctx,
            "graph_n_ctx": self.graph_n_ctx,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "max_tokens": self.max_tokens,
            "graph_temperature": self.graph_temperature,
            "graph_max_tokens": self.graph_max_tokens,
            "graph_repeat_penalty": self.graph_repeat_penalty,
        }


def llama_runtime_config(workspace_path: str | None = None) -> LlamaRuntimeConfig:
    model_filename = preferred_chat_model_filename(workspace_path)
    gpu_strategy = _gpu_strategy_from_env(model_filename)
    return LlamaRuntimeConfig(
        model_filename=model_filename,
        n_gpu_layers=gpu_strategy.n_gpu_layers,
        gpu_mode=gpu_strategy.mode,
        gpu_device_name=gpu_strategy.device_name,
        gpu_vendor=gpu_strategy.vendor,
        gpu_memory_mb=gpu_strategy.memory_mb,
        gpu_reason=gpu_strategy.reason,
        gpu_detection_source=gpu_strategy.detection_source,
        gpu_backend_available=gpu_strategy.backend_available,
        chat_n_ctx=_bounded_int("SOULDRIVE_LLAMA_CHAT_CTX", 8192, 1024, 32768),
        graph_n_ctx=_bounded_int("SOULDRIVE_LLAMA_GRAPH_CTX", 8192, 1024, 16384),
        temperature=_bounded_float("SOULDRIVE_LLAMA_TEMPERATURE", 0.01, 0.0, 1.0),
        top_p=_bounded_float("SOULDRIVE_LLAMA_TOP_P", 0.85, 0.05, 1.0),
        repeat_penalty=_bounded_float("SOULDRIVE_LLAMA_REPEAT_PENALTY", 1.12, 1.0, 2.0),
        max_tokens=_bounded_int("SOULDRIVE_LLAMA_MAX_TOKENS", 900, 128, 4096),
        graph_temperature=_bounded_float("SOULDRIVE_GRAPH_TEMPERATURE", 0.1, 0.0, 1.0),
        graph_max_tokens=_bounded_int("SOULDRIVE_GRAPH_MAX_TOKENS", 2048, 256, 4096),
        graph_repeat_penalty=_bounded_float("SOULDRIVE_GRAPH_REPEAT_PENALTY", 1.1, 1.0, 2.0),
    )


@dataclass(frozen=True)
class LlamaLoadResult:
    model: Any
    load_ms: int
    effective_config: LlamaRuntimeConfig
    fallback_error: str | None = None


def resolve_chat_model_path(workspace_path: str | None = None, config: LlamaRuntimeConfig | None = None) -> str:
    runtime_config = config or llama_runtime_config(workspace_path)
    return resolve_model_path(runtime_config.model_filename, workspace_path)


def preferred_chat_model_filename(workspace_path: str | None = None) -> str:
    configured = os.environ.get("SOULDRIVE_CHAT_MODEL")
    if configured:
        return configured
    for directory in model_search_dirs(workspace_path):
        for model_filename in PREFERRED_CHAT_MODELS:
            if (directory / model_filename).exists():
                return model_filename
    return PREFERRED_CHAT_MODELS[0]


def load_llama_with_gpu_fallback(
    model_path: str,
    config: LlamaRuntimeConfig,
    n_ctx: int,
) -> LlamaLoadResult:
    from llama_cpp import Llama

    def load_with_layers(layers: int):
        return Llama(
            model_path=model_path,
            n_gpu_layers=layers,
            n_ctx=n_ctx,
            verbose=False,
        )

    if config.n_gpu_layers > 0:
        smoke = _isolated_gpu_smoke_passes(model_path, config.n_gpu_layers, n_ctx)
        if not smoke["ok"]:
            reason = _summarize_gpu_failure(smoke["reason"])
            record_gpu_runtime_failure(reason)
            cpu_config = replace(
                config,
                n_gpu_layers=0,
                gpu_mode="cpu_fallback",
                gpu_reason=f"GPU isolated smoke test failed; fell back to CPU: {reason[:180]}",
            )
            model, load_ms = timed_model_load(lambda: load_with_layers(0))
            return LlamaLoadResult(
                model=model,
                load_ms=load_ms,
                effective_config=cpu_config,
                fallback_error=reason,
            )

    try:
        model, load_ms = timed_model_load(lambda: load_with_layers(config.n_gpu_layers))
        return LlamaLoadResult(model=model, load_ms=load_ms, effective_config=config)
    except Exception as exc:
        try:
            close_fn = getattr(model, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception:
            pass
        if config.n_gpu_layers <= 0:
            raise
        reason = _summarize_gpu_failure(str(exc))
        cpu_config = replace(
            config,
            n_gpu_layers=0,
            gpu_mode="cpu_fallback",
            gpu_reason=f"GPU initialization failed; fell back to CPU: {reason[:180]}",
        )
        record_gpu_runtime_failure(reason)
        model, load_ms = timed_model_load(lambda: load_with_layers(0))
        return LlamaLoadResult(
            model=model,
            load_ms=load_ms,
            effective_config=cpu_config,
            fallback_error=reason,
        )


def _isolated_gpu_smoke_passes(model_path: str, n_gpu_layers: int, n_ctx: int) -> dict[str, Any]:
    command = _gpu_smoke_command(model_path, n_gpu_layers, n_ctx)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:
        return {"ok": False, "reason": str(exc)}
    if completed.returncode == 0:
        return {"ok": True, "reason": "GPU smoke passed"}
    stderr = (completed.stderr or completed.stdout or "").strip()
    return {
        "ok": False,
        "reason": _summarize_gpu_failure(stderr) if stderr else f"gpu smoke exited {completed.returncode}",
    }


def _gpu_smoke_command(model_path: str, n_gpu_layers: int, n_ctx: int) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "gpu-smoke", model_path, str(n_gpu_layers), str(n_ctx)]
    return [
        sys.executable,
        "-m",
        "core.gpu_smoke",
        model_path,
        str(n_gpu_layers),
        str(n_ctx),
    ]


def model_runtime_diagnostics(workspace_path: str | None = None) -> dict[str, Any]:
    config = llama_runtime_config(workspace_path)
    chat_path = Path(resolve_chat_model_path(workspace_path, config))
    embedding_path = Path(resolve_model_path(DEFAULT_EMBEDDING_MODEL, workspace_path))
    search_roots = model_search_dirs(workspace_path)
    return {
        "ready": chat_path.exists() and embedding_path.exists(),
        "config": config.public_dict(),
        "chat_model": _path_status(chat_path),
        "embedding_model": _path_status(embedding_path),
        "search_roots": [_redact_path(path) for path in search_roots],
    }


def timed_model_load(load_fn):
    started_at = time.time()
    result = load_fn()
    return result, int((time.time() - started_at) * 1000)


def _path_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "ready": False,
            "name": path.name,
            "size": 0,
        }
    if path.is_file():
        size = path.stat().st_size
    else:
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return {
        "ready": True,
        "name": path.name,
        "size": size,
    }


def _redact_path(path: Path) -> str:
    return path.name if path.name else str(path)


def _summarize_gpu_failure(reason: str) -> str:
    normalized = " ".join((reason or "").split())
    lowered = normalized.lower()
    if "cuda error" in lowered or "ggml-cuda" in lowered:
        return "llama.cpp CUDA backend failed during isolated smoke test; CPU fallback is active"
    if "out of memory" in lowered or "cuda_malloc" in lowered:
        return "GPU memory was insufficient during isolated smoke test; CPU fallback is active"
    capacity_warning = "n_ctx_per_seq"
    if capacity_warning in normalized and "full capacity of the model will not be utilized" in normalized:
        normalized = normalized.split("full capacity of the model will not be utilized", 1)[-1].strip(" -:;")
    return normalized[-240:] if normalized else "GPU smoke test failed; CPU fallback is active"


def _gpu_strategy_from_env(model_filename: str) -> GpuAccelerationStrategy:
    raw_value = os.environ.get("SOULDRIVE_LLAMA_GPU_LAYERS", "auto")
    value = str(raw_value).strip().lower()
    if value in ("", "auto"):
        return llama_gpu_strategy(model_filename)
    if value in ("0", "off", "false", "cpu", "none"):
        return GpuAccelerationStrategy(
            n_gpu_layers=0,
            mode="manual_cpu",
            device_name=None,
            vendor=None,
            memory_mb=None,
            reason="SOULDRIVE_LLAMA_GPU_LAYERS disables GPU offload",
            detection_source=None,
            backend_available=False,
        )
    layers = _bounded_int("SOULDRIVE_LLAMA_GPU_LAYERS", 0, 0, 128)
    return GpuAccelerationStrategy(
        n_gpu_layers=layers,
        mode="manual",
        device_name=None,
        vendor=None,
        memory_mb=None,
        reason="SOULDRIVE_LLAMA_GPU_LAYERS manual override",
        detection_source=None,
        backend_available=layers > 0,
    )


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _bounded_float(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))
