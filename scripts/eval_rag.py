import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime_state import get_runtime_state
from core.security_context import WORKSPACE_DATA_KEY_ENV, get_workspace_keys, restore_workspace_keys, set_workspace_keys
from core.workspace import WORKSPACE_MANIFEST, SoulDriveWorkspace
from core.workspace_crypto import unlock_keystore

CITATION_PATTERN = re.compile(r"\[E(\d+)\]")
REFUSAL_MARKERS = (
    "未找到足够可靠",
    "没有足够",
    "无法回答",
    "不能回答",
    "不会生成可能误导",
    "无相关证据",
)
SOURCE_ALIASES = {
    "大模型知识管理系统": ["大模型知识管理系统"],
    "大模型基准测试体系研究报告": ["大模型基准测试体系研究报告"],
    "数据分类分级规则": ["数据分类分级规则", "数据安全技术数据分类分级规则"],
    "生成式人工智能服务安全基本要求": ["生成式人工智能服务安全基本要求"],
    "知识库问答场景中大语言模型私有化研究与应用实践": ["知识库问答场景中大语言模型私有化研究与应用实践"],
    "大模型私有部署与基础应用落地": ["大模型私有部署与基础应用落地"],
    "城市+AI应用场景清单": ["城市+AI应用场景清单", "城市AI应用场景清单"],
    "综合交通运输大模型智能体创新应用典型案例": [
        "综合交通运输大模型智能体创新应用典型案例",
        "综合交通运输大模型智能体创新应用",
    ],
    "人工智能安全治理蓝皮书": ["人工智能安全治理蓝皮书"],
    "数据要素发展报告": ["数据要素发展报告"],
    "基于大模型和RAG的知识库问答系统": ["基于大模型和RAG的知识库问答系统", "MaxKB"],
    "企业级RAG方案": ["企业级RAG方案", "LazyLLM企业级RAG方案", "LazyLLM"],
    "MaxKB": ["MaxKB", "基于大模型和RAG的知识库问答系统"],
    "LazyLLM": ["LazyLLM", "LazyLLM企业级RAG方案", "企业级RAG方案"],
}


@dataclass(frozen=True)
class EvalRecord:
    id: str
    question: str
    expected_sources: list[str]
    type: str
    rubric: list[str]
    expected_keywords: list[str]
    should_refuse: bool
    requires_multi_doc: bool
    graph_relevant: bool


@dataclass(frozen=True)
class EvalResult:
    id: str
    question: str
    expected_sources: list[str]
    answer: str
    evidence: list[dict[str, Any]]
    elapsed_ms: int | None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SoulDrive retrieval and citation quality.")
    parser.add_argument("--dataset", default="eval/enterprise_zh_v0.2.jsonl", help="JSONL evaluation dataset path.")
    parser.add_argument("--workspace", default=None, help="SoulDrive workspace root. Defaults to current runtime workspace.")
    parser.add_argument("--top-k", type=int, default=3, help="Evidence count to retrieve.")
    parser.add_argument("--from-results", default=None, help="Read prior JSONL results instead of running retrieval.")
    parser.add_argument("--save-results", default=None, help="Write JSONL retrieval results for later comparison.")
    parser.add_argument("--passphrase", default=None, help="Workspace passphrase for encrypted local indexes.")
    args = parser.parse_args(argv)

    records = load_dataset(Path(args.dataset))
    if not records:
        print("评测样本数: 0")
        print("没有可评测的数据。")
        return 1

    if args.from_results:
        results = load_results(Path(args.from_results), records)
    else:
        workspace = resolve_workspace_arg(args.workspace)
        if not workspace:
            parser.error(
                "live retrieval requires a SoulDrive workspace. "
                "Pass --workspace <path>, set SOULDRIVE_WORKSPACE, or unlock the workspace in the app first."
            )
        prepare_workspace_keys(workspace, args.passphrase)
        if workspace_requires_key(workspace) and get_workspace_keys(workspace) is None:
            parser.error(
                "encrypted workspace requires unlock credentials. "
                "Pass --passphrase <workspace password> or set SOULDRIVE_WORKSPACE_PASSPHRASE."
            )
        print(f"评测工作区: {workspace}")
        results = run_retrieval(records, workspace=workspace, top_k=args.top_k)

    if args.save_results:
        save_results(Path(args.save_results), results)

    report = summarize(records, results, top_k=args.top_k)
    print_report(report)
    return 0


def load_dataset(path: Path) -> list[EvalRecord]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        records.append(
            EvalRecord(
                id=str(payload["id"]),
                question=str(payload["question"]),
                expected_sources=[str(item) for item in payload.get("expected_sources", [])],
                type=str(payload["type"]),
                rubric=[str(item) for item in payload.get("rubric", [])],
                expected_keywords=[str(item) for item in payload.get("expected_keywords", [])],
                should_refuse=bool(payload.get("should_refuse", False)),
                requires_multi_doc=bool(payload.get("requires_multi_doc", False)),
                graph_relevant=bool(payload.get("graph_relevant", False)),
            )
        )
    return records


def load_results(path: Path, records: list[EvalRecord]) -> list[EvalResult]:
    by_id: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            by_id[str(payload["id"])] = payload

    results = []
    for record in records:
        payload = by_id.get(record.id, {})
        results.append(
            EvalResult(
                id=record.id,
                question=record.question,
                expected_sources=record.expected_sources,
                answer=str(payload.get("answer", "")),
                evidence=list(payload.get("evidence", [])),
                elapsed_ms=_optional_int(payload.get("elapsed_ms")),
            )
        )
    return results


def run_retrieval(records: list[EvalRecord], workspace: str | None, top_k: int) -> list[EvalResult]:
    from core.knowledge_base import LocalKnowledgeBase

    if not workspace:
        raise ValueError("workspace is required for live retrieval")
    kb = LocalKnowledgeBase(workspace_path=workspace)
    results = []
    try:
        for record in records:
            started_at = time.time()
            retrieval = kb.search_with_evidence(record.question, top_k=top_k)
            elapsed_ms = int((time.time() - started_at) * 1000)
            results.append(
                EvalResult(
                    id=record.id,
                    question=record.question,
                    expected_sources=record.expected_sources,
                    answer="",
                    evidence=list(retrieval.get("evidence", [])),
                    elapsed_ms=elapsed_ms,
                )
            )
    finally:
        kb.close()
    return results


def resolve_workspace_arg(workspace: str | None) -> str | None:
    runtime_workspace = str(get_runtime_state().get("workspace_path") or "")
    candidates = [
        workspace,
        os.environ.get("SOULDRIVE_WORKSPACE"),
        runtime_workspace,
        find_unlocked_drive_workspace(),
        str(PROJECT_ROOT / "souldrive_db"),
    ]
    for candidate in candidates:
        if candidate and is_workspace_path(candidate):
            return str(Path(candidate))
    return None


def is_workspace_path(path: str) -> bool:
    if not path:
        return False
    root = Path(path)
    return (root / "config" / WORKSPACE_MANIFEST).exists()


def prepare_workspace_keys(workspace_path: str, passphrase: str | None) -> bool:
    if restore_workspace_keys(workspace_path, os.environ.get(WORKSPACE_DATA_KEY_ENV)):
        return True

    passphrase = passphrase or os.environ.get("SOULDRIVE_WORKSPACE_PASSPHRASE")
    if not passphrase:
        return False

    workspace = SoulDriveWorkspace(workspace_path).ensure()
    keys = unlock_keystore(workspace, passphrase)
    set_workspace_keys(workspace.root_path, keys)
    return True


def workspace_requires_key(workspace_path: str) -> bool:
    return (Path(workspace_path) / "index" / "secure_vectors.sqlite").exists()


def find_unlocked_drive_workspace() -> str | None:
    for drive in iter_filesystem_drives():
        workspace = Path(drive.mountpoint) / "SoulDrive"
        if not is_workspace_path(str(workspace)):
            continue

        state_path = workspace / "runtime" / "runtime_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("locked") is False:
            state_workspace = str(state.get("workspace_path") or workspace)
            if is_workspace_path(state_workspace):
                return str(Path(state_workspace))
    return None


def iter_filesystem_drives():
    try:
        import psutil

        return psutil.disk_partitions(all=False)
    except Exception:
        if os.name != "nt":
            return []
        drives = []
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            root = f"{letter}:\\"
            if Path(root).exists():
                drives.append(type("Drive", (), {"mountpoint": root})())
        return drives


def save_results(path: Path, results: list[EvalResult]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for result in results:
        lines.append(
            json.dumps(
                {
                    "id": result.id,
                    "question": result.question,
                    "expected_sources": result.expected_sources,
                    "answer": result.answer,
                    "evidence": result.evidence,
                    "elapsed_ms": result.elapsed_ms,
                },
                ensure_ascii=False,
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(records: list[EvalRecord], results: list[EvalResult], top_k: int) -> dict[str, Any]:
    result_by_id = {result.id: result for result in results}
    rows = []
    for record in records:
        result = result_by_id[record.id]
        rows.append(score_record(record, result, top_k=top_k))

    positive_rows = [row for row in rows if row["expected_sources"]]
    answer_rows = [row for row in rows if row["has_answer"]]
    refusal_rows = [row for row in rows if row["should_refuse"] and row["has_answer"]]
    keyword_rows = [row for row in rows if row["keyword_sample"]]
    multi_doc_rows = [row for row in rows if row["requires_multi_doc"]]
    graph_rows = [row for row in rows if row["graph_relevant"]]
    latencies = [row["elapsed_ms"] for row in rows if row["elapsed_ms"] is not None]

    return {
        "sample_count": len(rows),
        "top_k": top_k,
        "retrieval_all_hit_rate": _ratio(sum(row["source_all_hit"] for row in positive_rows), len(positive_rows)),
        "retrieval_any_hit_rate": _ratio(sum(row["source_any_hit"] for row in positive_rows), len(positive_rows)),
        "first_hit_rate": _ratio(sum(row["first_hit"] for row in positive_rows), len(positive_rows)),
        "mean_reciprocal_rank": _ratio(
            sum(row["reciprocal_rank"] for row in positive_rows),
            len(positive_rows),
        ),
        "average_source_coverage": _ratio(
            sum(row["source_coverage"] for row in positive_rows),
            len(positive_rows),
        ),
        "multi_doc_all_hit_rate": _ratio(
            sum(row["source_all_hit"] for row in multi_doc_rows),
            len(multi_doc_rows),
        ),
        "graph_context_hit_rate": _ratio(
            sum(row["source_any_hit"] for row in graph_rows),
            len(graph_rows),
        ),
        "citation_valid_rate": _ratio(sum(row["citation_valid"] for row in answer_rows), len(answer_rows)),
        "refusal_accuracy": _ratio(sum(row["refusal_correct"] for row in refusal_rows), len(refusal_rows)),
        "answer_keyword_coverage": _ratio(sum(row["keyword_coverage"] for row in keyword_rows), len(keyword_rows)),
        "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
        "rows": rows,
    }


def score_record(record: EvalRecord, result: EvalResult, top_k: int) -> dict[str, Any]:
    evidence = result.evidence[:top_k]
    answer = result.answer or ""
    expected_sources = record.expected_sources
    source_hits = [
        any(source_matches(item.get("source_filename"), expected) for item in evidence)
        for expected in expected_sources
    ]
    first_evidence = evidence[0] if evidence else {}
    first_hit = bool(expected_sources) and any(
        source_matches(first_evidence.get("source_filename"), expected)
        for expected in expected_sources
    )
    first_match_rank = first_matching_rank(evidence, expected_sources)
    source_coverage = (sum(source_hits) / len(expected_sources)) if expected_sources else 0.0
    citation_valid = citations_are_valid(answer, evidence)
    refusal_correct = record.should_refuse and answer_refuses(answer)
    keyword_coverage = keyword_coverage_ratio(answer, record.expected_keywords)
    return {
        "id": record.id,
        "question": record.question,
        "expected_sources": expected_sources,
        "top_sources": [str(item.get("source_filename") or "未知来源") for item in evidence],
        "has_answer": bool(answer.strip()),
        "source_all_hit": bool(expected_sources) and all(source_hits),
        "source_any_hit": bool(expected_sources) and any(source_hits),
        "first_hit": first_hit,
        "first_match_rank": first_match_rank,
        "reciprocal_rank": (1.0 / first_match_rank) if first_match_rank else 0.0,
        "source_coverage": source_coverage,
        "citation_valid": citation_valid,
        "refusal_correct": refusal_correct,
        "should_refuse": record.should_refuse,
        "requires_multi_doc": record.requires_multi_doc,
        "graph_relevant": record.graph_relevant,
        "keyword_sample": bool(answer.strip() and record.expected_keywords and not record.should_refuse),
        "keyword_coverage": keyword_coverage,
        "elapsed_ms": result.elapsed_ms,
    }


def source_matches(source_filename: Any, expected: str) -> bool:
    source = normalize_source_name(str(source_filename or ""))
    for alias in source_aliases(expected):
        normalized_alias = normalize_source_name(alias)
        if normalized_alias and normalized_alias in source:
            return True
    return False


def source_aliases(expected: str) -> list[str]:
    aliases = SOURCE_ALIASES.get(expected, [])
    return [expected, *aliases]


def normalize_source_name(value: str) -> str:
    lowered = value.lower()
    for suffix in (".pdf", ".docx", ".md", ".html", ".txt"):
        if lowered.endswith(suffix):
            lowered = lowered[: -len(suffix)]
            break
    return re.sub(r"[\s_\-—–·,，。\.《》<>“”\"'‘’()（）【】\[\]：:；;]+", "", lowered)


def first_matching_rank(evidence: list[dict[str, Any]], expected_sources: list[str]) -> int | None:
    if not expected_sources:
        return None
    for rank, item in enumerate(evidence, start=1):
        if any(source_matches(item.get("source_filename"), expected) for expected in expected_sources):
            return rank
    return None


def keyword_coverage_ratio(answer: str, expected_keywords: list[str]) -> float:
    keywords = [keyword for keyword in expected_keywords if keyword]
    if not answer.strip() or not keywords:
        return 0.0
    normalized_answer = answer.lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in normalized_answer)
    return hits / len(keywords)


def citations_are_valid(answer: str, evidence: list[dict[str, Any]]) -> bool:
    cited_ids = {f"E{match}" for match in CITATION_PATTERN.findall(answer or "")}
    if not answer:
        return True
    if not cited_ids:
        return answer_refuses(answer)
    available_ids = {str(item.get("id")) for item in evidence if item.get("id")}
    return cited_ids <= available_ids


def answer_refuses(answer: str) -> bool:
    return any(marker in (answer or "") for marker in REFUSAL_MARKERS)


def print_report(report: dict[str, Any]):
    print(f"评测样本数: {report['sample_count']}")
    print(f"检索全命中率@{report['top_k']}: {_percent(report['retrieval_all_hit_rate'])}")
    print(f"检索任一命中率@{report['top_k']}: {_percent(report['retrieval_any_hit_rate'])}")
    print(f"首条命中率@1: {_percent(report['first_hit_rate'])}")
    print(f"平均倒数排名(MRR): {_decimal(report['mean_reciprocal_rank'])}")
    print(f"平均来源覆盖率@{report['top_k']}: {_percent(report['average_source_coverage'])}")
    print(f"跨文档全命中率@{report['top_k']}: {_percent(report['multi_doc_all_hit_rate'])}")
    print(f"图谱相关题命中率@{report['top_k']}: {_percent(report['graph_context_hit_rate'])}")
    print(f"引用合法率: {_percent(report['citation_valid_rate'])}")
    print(f"拒答正确率: {_percent(report['refusal_accuracy'])}")
    print(f"答案关键词覆盖率: {_percent(report['answer_keyword_coverage'])}")
    if report["avg_latency_ms"] is not None:
        print(f"平均检索耗时: {report['avg_latency_ms']} ms")

    misses = [row for row in report["rows"] if row["expected_sources"] and not row["source_all_hit"]]
    if misses:
        print("\n未命中样本:")
        for row in misses[:10]:
            expected = " / ".join(row["expected_sources"])
            actual = " / ".join(row["top_sources"][:3]) or "无证据"
            print(f"- {row['id']}: 期望 {expected}; 实际 {actual}")


def _percent(value: float | None) -> str:
    if value is None:
        return "无样本"
    return f"{value * 100:.1f}%"


def _decimal(value: float | None) -> str:
    if value is None:
        return "无样本"
    return f"{value:.3f}"


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
