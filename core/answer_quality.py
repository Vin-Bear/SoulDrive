import re
from dataclasses import dataclass
from typing import Any


CITATION_PATTERN = re.compile(r"\[E(\d+)\]")
DEFAULT_MIN_EVIDENCE_SCORE = 0.015


@dataclass(frozen=True)
class EvidenceGateResult:
    allowed: bool
    reason: str
    evidence_count: int
    max_score: float

    def public_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "evidence_count": self.evidence_count,
            "max_score": self.max_score,
        }


def evaluate_evidence_gate(
    evidence: list[dict[str, Any]],
    min_evidence_count: int = 1,
    min_score: float = DEFAULT_MIN_EVIDENCE_SCORE,
) -> EvidenceGateResult:
    if len(evidence) < min_evidence_count:
        return EvidenceGateResult(False, "no sufficient evidence", len(evidence), 0.0)

    scores = [_safe_float(item.get("score")) for item in evidence]
    max_score = max(scores or [0.0])
    if max_score < min_score:
        return EvidenceGateResult(False, "evidence confidence below threshold", len(evidence), max_score)

    return EvidenceGateResult(True, "evidence accepted", len(evidence), max_score)


def refusal_answer(query: str, gate: EvidenceGateResult) -> str:
    _ = query
    return (
        "根据本地存储的知识库，未找到足够可靠的相关证据，"
        f"因此不会生成可能误导的回答。原因：{gate.reason}。"
    )


def citation_coverage(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    cited_ids = {f"E{match}" for match in CITATION_PATTERN.findall(answer or "")}
    available_ids = {str(item.get("id")) for item in evidence if item.get("id")}
    used = sorted(cited_ids & available_ids)
    unsupported = sorted(cited_ids - available_ids)
    return {
        "available_count": len(available_ids),
        "cited_count": len(used),
        "coverage": round(len(used) / len(available_ids), 6) if available_ids else 0.0,
        "used": used,
        "unsupported": unsupported,
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
