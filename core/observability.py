import threading
import time
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RuntimeMetrics:
    started_at: float = field(default_factory=time.time)
    total_requests: int = 0
    unauthorized_requests: int = 0
    rejected_remote_requests: int = 0
    rate_limited_requests: int = 0
    rejected_body_requests: int = 0
    failed_requests: int = 0
    chat_requests: int = 0
    chat_failures: int = 0
    retrieval_failures: int = 0
    generation_failures: int = 0
    model_load_failures: int = 0
    last_model_load_ms: int = 0
    total_latency_ms: int = 0
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_request(self, elapsed_ms: int, status_code: int):
        with self._lock:
            self.total_requests += 1
            self.total_latency_ms += max(0, elapsed_ms)
            if status_code >= 500:
                self.failed_requests += 1

    def increment(self, name: str, amount: int = 1):
        with self._lock:
            current = getattr(self, name)
            setattr(self, name, current + amount)

    def record_error(self, message: str):
        with self._lock:
            self.last_error = _sanitize_error_message(message)[:500]

    def record_model_load(self, elapsed_ms: int):
        with self._lock:
            self.last_model_load_ms = max(0, elapsed_ms)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            average_latency = self.total_latency_ms / self.total_requests if self.total_requests else 0
            return {
                "uptime_seconds": int(time.time() - self.started_at),
                "total_requests": self.total_requests,
                "unauthorized_requests": self.unauthorized_requests,
                "rejected_remote_requests": self.rejected_remote_requests,
                "rate_limited_requests": self.rate_limited_requests,
                "rejected_body_requests": self.rejected_body_requests,
                "failed_requests": self.failed_requests,
                "chat_requests": self.chat_requests,
                "chat_failures": self.chat_failures,
                "retrieval_failures": self.retrieval_failures,
                "generation_failures": self.generation_failures,
                "model_load_failures": self.model_load_failures,
                "last_model_load_ms": self.last_model_load_ms,
                "average_latency_ms": round(average_latency, 2),
                "last_error": self.last_error,
            }


runtime_metrics = RuntimeMetrics()


WINDOWS_PATH_PATTERN = re.compile(r"(?i)\b[A-Z]:\\[^\s\r\n\t\"']+")
POSIX_PATH_PATTERN = re.compile(r"(?<!\w)/(?:[^\s\r\n\t\"']+/)+[^\s\r\n\t\"']+")
SENSITIVE_FIELD_PATTERN = re.compile(
    r"(?i)\b(token|secret|password|api[_-]?key|hardware[_-]?sn|serial)\b\s*[:=]\s*[^\s\r\n\t\"']+"
)


def _sanitize_error_message(message: str) -> str:
    text = str(message or "")
    text = WINDOWS_PATH_PATTERN.sub("[local path]", text)
    text = POSIX_PATH_PATTERN.sub("[local path]", text)
    return SENSITIVE_FIELD_PATTERN.sub(lambda match: f"{match.group(1)}=[redacted]", text)
