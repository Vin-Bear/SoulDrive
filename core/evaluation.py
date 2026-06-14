import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.answer_quality import evaluate_evidence_gate
from core.retrieval import SearchCandidate, rank_hybrid_candidates


@dataclass
class RetrievalEvalCase:
    query: str
    expected_ids: list[str]
    candidates: list[SearchCandidate]


def load_retrieval_eval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    cases: list[RetrievalEvalCase] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            cases.append(RetrievalEvalCase(
                query=payload["query"],
                expected_ids=list(payload["expected_ids"]),
                candidates=[
                    SearchCandidate(
                        id=item["id"],
                        content=item["content"],
                        metadata=item.get("metadata") or {},
                        dense_rank=item.get("dense_rank"),
                        graph_score=item.get("graph_score", 0.0),
                        vector_distance=item.get("vector_distance"),
                    )
                    for item in payload["candidates"]
                ],
            ))
    return cases


def evaluate_retrieval_cases(cases: list[RetrievalEvalCase], top_k: int = 3) -> dict[str, Any]:
    if not cases:
        return {"case_count": 0, "hit_rate_at_k": 0.0, "mean_reciprocal_rank": 0.0}

    hits = 0
    reciprocal_ranks = []
    accepted_cases = 0
    average_top_score = 0.0
    low_evidence_cases = []

    for case in cases:
        ranked = rank_hybrid_candidates(case.query, case.candidates, top_k=top_k)
        ranked_ids = [candidate.id for candidate in ranked]
        expected = set(case.expected_ids)
        hit_positions = [index for index, candidate_id in enumerate(ranked_ids, start=1) if candidate_id in expected]
        evidence = [
            {"id": f"E{index}", "score": candidate.final_score}
            for index, candidate in enumerate(ranked, start=1)
        ]
        gate = evaluate_evidence_gate(evidence)
        if gate.allowed:
            accepted_cases += 1
        else:
            low_evidence_cases.append(case.query)
        if ranked:
            average_top_score += ranked[0].final_score

        if hit_positions:
            hits += 1
            reciprocal_ranks.append(1 / hit_positions[0])
        else:
            reciprocal_ranks.append(0.0)

    return {
        "case_count": len(cases),
        "hit_rate_at_k": round(hits / len(cases), 6),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(cases), 6),
        "evidence_acceptance_rate": round(accepted_cases / len(cases), 6),
        "average_top_score": round(average_top_score / len(cases), 6),
        "low_evidence_cases": low_evidence_cases,
    }
