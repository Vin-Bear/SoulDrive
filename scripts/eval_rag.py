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


@dataclass(frozen=True)
class EvalRecord:
    id: str
    question: str
    expected_sources: list[str]
    type: str
    rubric: list[str]


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
    parser.add_argument("--dataset", default="eval/enterprise_zh_v0.1.jsonl", help="JSONL evaluation dataset path.")
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
    negative_rows = [row for row in rows if not row["expected_sources"]]
    answer_rows = [row for row in rows if row["has_answer"]]
    answer_negative_rows = [row for row in negative_rows if row["has_answer"]]
    latencies = [row["elapsed_ms"] for row in rows if row["elapsed_ms"] is not None]

    return {
        "sample_count": len(rows),
        "top_k": top_k,
        "retrieval_hit_rate": _ratio(sum(row["source_hit"] for row in positive_rows), len(positive_rows)),
        "citation_valid_rate": _ratio(sum(row["citation_valid"] for row in answer_rows), len(answer_rows)),
        "refusal_accuracy": _ratio(
            sum(row["refusal_correct"] for row in answer_negative_rows),
            len(answer_negative_rows),
        ),
        "avg_latency_ms": int(sum(latencies) / len(latencies)) if latencies else None,
        "rows": rows,
    }


def score_record(record: EvalRecord, result: EvalResult, top_k: int) -> dict[str, Any]:
    evidence = result.evidence[:top_k]
    answer = result.answer or ""
    expected_sources = record.expected_sources
    source_hit = bool(expected_sources) and all(
        any(source_matches(item.get("source_filename"), expected) for item in evidence)
        for expected in expected_sources
    )
    citation_valid = citations_are_valid(answer, evidence)
    refusal_correct = (not expected_sources) and answer_refuses(answer)
    return {
        "id": record.id,
        "question": record.question,
        "expected_sources": expected_sources,
        "top_sources": [str(item.get("source_filename") or "未知来源") for item in evidence],
        "has_answer": bool(answer.strip()),
        "source_hit": source_hit,
        "citation_valid": citation_valid,
        "refusal_correct": refusal_correct,
        "elapsed_ms": result.elapsed_ms,
    }


def source_matches(source_filename: Any, expected: str) -> bool:
    source = str(source_filename or "").lower()
    normalized_expected = expected.lower()
    return normalized_expected in source


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
    print(f"检索命中率@{report['top_k']}: {_percent(report['retrieval_hit_rate'])}")
    print(f"引用合法率: {_percent(report['citation_valid_rate'])}")
    print(f"拒答正确率: {_percent(report['refusal_accuracy'])}")
    if report["avg_latency_ms"] is not None:
        print(f"平均检索耗时: {report['avg_latency_ms']} ms")

    misses = [row for row in report["rows"] if row["expected_sources"] and not row["source_hit"]]
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
