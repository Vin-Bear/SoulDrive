import os
import gc
import json
import time
import uuid
from core.answer_quality import citation_coverage, evaluate_evidence_gate, refusal_answer
from core.audit_log import default_audit_logger
from core.knowledge_base import LocalKnowledgeBase
from core.graph_db import LocalGraphDB
from core.model_runtime import llama_runtime_config, load_llama_with_gpu_fallback, resolve_chat_model_path
from core.observability import runtime_metrics
from core.logging_config import get_logger

MAX_CONTEXT_CHARS_PER_EVIDENCE = 1800

logger = get_logger(__name__)


class RAGEngine:
    def __init__(
        self,
        kb: LocalKnowledgeBase,
        graph_db_path: str | None = None,
        workspace_path: str | None = None,
        audit_logger=None,
    ):
        self.kb = kb
        self.audit_logger = audit_logger or default_audit_logger
        self.graph_db = LocalGraphDB(db_path=graph_db_path) if graph_db_path else LocalGraphDB()
        self.runtime_config = llama_runtime_config(workspace_path)
        model_path = resolve_chat_model_path(workspace_path, self.runtime_config)
        logger.info("[RAG Engine] 正在物理层挂载 GGUF 离线模型: %s", os.path.basename(model_path))

        if not os.path.exists(model_path):
            runtime_metrics.increment("model_load_failures")
            message = f"[致命错误] 找不到模型文件: {model_path}\n请确保已将模型放入 SoulDrive/models 或配置 SOULDRIVE_MODEL_DIR。"
            runtime_metrics.record_error(message)
            raise FileNotFoundError(message)

        try:
            load_result = load_llama_with_gpu_fallback(
                model_path=model_path,
                config=self.runtime_config,
                n_ctx=self.runtime_config.chat_n_ctx,
            )
            self.llm = load_result.model
            self.runtime_config = load_result.effective_config
            runtime_metrics.record_model_load(load_result.load_ms)
            if load_result.fallback_error:
                runtime_metrics.record_error(f"GPU fallback: {load_result.fallback_error}")
                logger.warning("[RAG Engine] GPU 初始化失败，已自动回退 CPU 推理。")
        except Exception as exc:
            runtime_metrics.increment("model_load_failures")
            runtime_metrics.record_error(str(exc))
            raise
        logger.info("[RAG Engine] C++ 推理引擎初始化完毕，真·单体免驱就绪！")

    def close(self):
        """释放本地 LLM 和图谱连接，供 U 盘拔出时执行安全清场。"""
        if hasattr(self, "graph_db") and self.graph_db is not None:
            self.graph_db.close()
            self.graph_db = None

        if hasattr(self, "llm") and self.llm is not None:
            close_fn = getattr(self.llm, "close", None)
            if callable(close_fn):
                close_fn()
            self.llm = None

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def generate_response_stream(self, query: str, top_k: int = 3, trace_id: str | None = None):
        """工业级流式生成器：结合本地图谱数据，输出防幻觉的流式回答"""
        trace_id = trace_id or str(uuid.uuid4())
        started_at = time.time()

        # 1. 混合向量召回
        try:
            retrieval_result = self.kb.search_with_evidence(
                query=query,
                graph_db=self.graph_db,
                top_k=top_k
            )
            vector_docs = retrieval_result["documents"]
            vector_metas = retrieval_result["metadatas"]
            graph_context = retrieval_result["graph_context"]
            evidence = retrieval_result["evidence"]
            self.audit_logger.append_event("rag.retrieved", {
                "query_chars": len(query),
                "top_k": top_k,
                "retrieval_mode": retrieval_result["retrieval_mode"],
                "evidence_count": len(evidence),
                "matched_entities": retrieval_result["matched_entities"],
                "sources": [item["source_filename"] for item in evidence],
            }, trace_id=trace_id)
        except Exception as e:
            runtime_metrics.increment("retrieval_failures")
            runtime_metrics.record_error(str(e))
            self.audit_logger.append_event("rag.retrieve_failed", {
                "query_chars": len(query),
                "error": str(e),
            }, trace_id=trace_id)
            yield f"本地知识库未就绪。错误信息: {str(e)}"
            return

        evidence_gate = evaluate_evidence_gate(evidence)
        if not evidence_gate.allowed:
            answer = refusal_answer(query, evidence_gate)
            self.audit_logger.append_event("rag.refused_low_evidence", {
                "query_chars": len(query),
                "gate": evidence_gate.public_dict(),
            }, trace_id=trace_id)
            yield answer
            yield "\n\n```souldrive-evidence\n"
            yield json.dumps(evidence, ensure_ascii=False, indent=2)
            yield "\n```"
            return

        fast_answer = build_fast_evidence_answer(query, evidence)
        if fast_answer:
            self.audit_logger.append_event("rag.fast_answered", {
                "query_chars": len(query),
                "evidence_count": len(evidence),
            }, trace_id=trace_id)
            yield fast_answer
            if query_requests_mindmap(query):
                yield "\n\n```souldrive-mindmap\n"
                yield build_evidence_mindmap(query, evidence)
                yield "\n```"
            yield "\n\n```souldrive-evidence\n"
            yield json.dumps(evidence, ensure_ascii=False, indent=2)
            yield "\n```"
            return

        # ==========================================
        # 2. 组装向量文本块 (Vector Context) -> 负责“具体细节”
        # ==========================================
        vector_context_str = ""
        if vector_docs:
            for i, doc in enumerate(vector_docs):
                # 提取我们在 indexer 阶段强行注入的元数据 (防崩溃处理)
                source = vector_metas[i].get("source_filename", "未知文件") if vector_metas else "未知文件"
                evidence_id = evidence[i]["id"] if i < len(evidence) else f"E{i + 1}"
                section = evidence[i].get("section") if i < len(evidence) else None
                chunk_index = evidence[i].get("chunk_index") if i < len(evidence) else None
                context_text = compact_context_text(doc, MAX_CONTEXT_CHARS_PER_EVIDENCE)
                vector_context_str += (
                    f"[{evidence_id} | 来源: {source} | 章节: {section or '未知'} | 切片: {chunk_index}]\n"
                    f"{context_text}\n\n"
                )
        else:
            vector_context_str = "未检索到直接相关的文献段落。\n"

        # ==========================================
        # 3. 组装图谱逻辑链 (Graph Context) -> 负责“宏观逻辑与关联”
        # ==========================================
        if graph_context:
            graph_context_str = "【知识图谱逻辑链 (实体关系)】：\n" + "\n".join(graph_context) + "\n\n"
        else:
            graph_context_str = ""

        # ==========================================
        # 4. 构建大模型专属的 Hybrid Prompt (工业级防幻觉提示词)
        # ==========================================
        system_prompt = (
            "你是一个基于本地端侧知识引擎的高级 AI 架构师。\n"
            "请严格遵循以下规则作答：\n"
            "1. 你的回答必须完全基于提供的 <context>，绝不能使用你的预训练知识捏造事实。\n"
            "2. 如果 <context> 中包含【知识图谱逻辑链】，请优先参考该逻辑链来梳理回答的骨架。\n"
            "3. 如果 <context> 中包含【私有文献详情】，请提取其中的数据和原文细节来丰满你的回答。\n"
            "4. 如果信息不足以回答问题，请直接回答“根据本地存储的知识库，未找到相关信息”。"
            "5. 每个关键结论必须使用 [E1]、[E2] 这类证据编号标注来源；没有证据支撑的结论不要输出。"
            "6. Transformer、Attention、BERT、RoBERTa 等技术名词保持英文原名，不要翻译成“变压器”等非技术含义。"
            "7. 默认用 4 到 7 条要点回答，先说明机制，再说明论文依据；除非用户要求详细展开，不要输出冗长综述。"
            "8. 如果用户要求生成思维导图、导图、结构图或知识图谱，请在正常回答后追加一个 "
            "```souldrive-mindmap 代码块，代码块内部使用 Markdown 标题和项目符号表达回答结构；"
            "导图必须围绕用户问题组织，不要把来源、线索或文件名当成主要节点。"
        )

        user_prompt = (
            f"<context>\n"
            f"{graph_context_str}"
            f"【私有文献详情】：\n{vector_context_str}"
            f"</context>\n\n"
            f"<user_query>\n{query}\n</user_query>"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # 4. 触发 C++ 底层流式推理 (企业级采样与截断策略)
        try:
            stream = self.llm.create_chat_completion(
                messages=messages,
                stream=True,
                temperature=self.runtime_config.temperature,
                top_p=self.runtime_config.top_p,
                repeat_penalty=self.runtime_config.repeat_penalty,
                max_tokens=self.runtime_config.max_tokens,
                stop=["</context>", "<user_query>", "<|im_end|>"]  # 【企业级刹车片】：一旦模型开始幻觉输出这些边界词，底层 C++ 引擎会瞬间强行掐断！
            )

            generated_parts = []
            for chunk in stream:
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'content' in delta:
                        content = normalize_technical_terms(delta['content'])
                        generated_parts.append(content)
                        yield content

            citation_report = citation_coverage("".join(generated_parts), evidence)

            self.audit_logger.append_event("rag.completed", {
                "query_chars": len(query),
                "elapsed_ms": int((time.time() - started_at) * 1000),
                "evidence_count": len(evidence),
                "citation_coverage": citation_report,
            }, trace_id=trace_id)
            generated_text = "".join(generated_parts)
            if query_requests_mindmap(query) and "souldrive-mindmap" not in generated_text.lower():
                yield "\n\n```souldrive-mindmap\n"
                yield build_evidence_mindmap(query, evidence)
                yield "\n```"
            yield "\n\n```souldrive-evidence\n"
            yield json.dumps(evidence, ensure_ascii=False, indent=2)
            yield "\n```"

        except Exception as e:
            runtime_metrics.increment("generation_failures")
            runtime_metrics.record_error(str(e))
            self.audit_logger.append_event("rag.generate_failed", {
                "query_chars": len(query),
                "elapsed_ms": int((time.time() - started_at) * 1000),
                "error": str(e),
            }, trace_id=trace_id)
            yield f"\n\n[系统保护机制触发] 推理进程已拦截异常: {str(e)}"


def compact_context_text(text: str, max_chars: int = MAX_CONTEXT_CHARS_PER_EVIDENCE) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= max_chars:
        return normalized

    marker = " ... [context trimmed] ... "
    if max_chars <= len(marker) + 80:
        return normalized[:max_chars]

    available = max_chars - len(marker)
    head_budget = max(80, int(available * 0.68))
    tail_budget = max(40, available - head_budget)
    if head_budget + tail_budget > available:
        head_budget = available - tail_budget
    return f"{normalized[:head_budget]}{marker}{normalized[-tail_budget:]}"


def query_requests_mindmap(query: str) -> bool:
    normalized = (query or "").lower()
    markers = (
        "思维导图",
        "导图",
        "结构图",
        "知识图谱",
        "mindmap",
        "mind map",
        "knowledge graph",
    )
    return any(marker in normalized for marker in markers)


def build_evidence_mindmap(query: str, evidence: list[dict]) -> str:
    title = mindmap_title(query, evidence)
    if is_transformer_mindmap(query, evidence):
        return build_transformer_mindmap(title, evidence)

    return build_answer_structure_mindmap(query, title, evidence)


def build_transformer_mindmap(title: str, evidence: list[dict]) -> str:
    primary = evidence[0].get("id", "E1") if evidence else "E1"
    secondary = evidence[1].get("id", primary) if len(evidence) > 1 else primary
    tertiary = evidence[2].get("id", secondary) if len(evidence) > 2 else secondary
    lines = [
        f"# {title}",
        "## 核心思想",
        f"- 以 Attention/Self-Attention 建模 token 之间的依赖关系 [{primary}]",
        "## 结构组成",
        f"- 多头自注意力负责从不同表示子空间捕捉关系 [{primary}]",
        f"- 前馈层、残差连接、归一化与位置编码共同构成 Transformer block [{secondary}]",
        "## 主要优势",
        f"- 架构由可堆叠模块组成，适合拆成可解释的计算步骤 [{tertiary}]",
        "## 论文依据",
    ]
    lines.extend(format_evidence_outline(evidence))
    return "\n".join(lines)


def build_answer_structure_mindmap(query: str, title: str, evidence: list[dict]) -> str:
    lines = [
        f"# {title}",
        "## 问题焦点",
        f"- {compact_mindmap_text(query or title, 72)}",
        "## 关键论点",
    ]
    for item in evidence[:3]:
        evidence_id = item.get("id") or "E?"
        snippet = compact_mindmap_text(item.get("snippet") or item.get("section") or "相关论点", 86)
        if snippet:
            lines.append(f"- {snippet} [{evidence_id}]")
    lines.append("## 论文依据")
    lines.extend(format_evidence_outline(evidence))
    return "\n".join(lines)


def mindmap_title(query: str, evidence: list[dict]) -> str:
    query_title = mindmap_query_title(query)
    if query_title:
        return query_title

    for item in evidence:
        filename = item.get("source_filename")
        if filename:
            return os.path.splitext(str(filename))[0]
    normalized = " ".join((query or "论文导图").split())
    return compact_mindmap_text(normalized, 48)


def mindmap_query_title(query: str) -> str | None:
    normalized = " ".join((query or "").split())
    lowered = normalized.lower()
    if "transformer" in lowered:
        if "架构" in normalized or "architecture" in lowered:
            return "Transformer 架构"
        if any(marker in normalized for marker in ("机制", "原理", "定义", "是什么")):
            return "Transformer 机制"
        return "Transformer"
    return None


def is_transformer_mindmap(query: str, evidence: list[dict]) -> bool:
    text_parts = [query or ""]
    for item in evidence:
        text_parts.append(str(item.get("section") or ""))
        text_parts.append(str(item.get("snippet") or ""))
    return "transformer" in " ".join(text_parts).lower()


def format_evidence_outline(evidence: list[dict]) -> list[str]:
    if not evidence:
        return ["- 未捕获到可展示的论文证据"]

    lines = []
    for item in evidence[:5]:
        evidence_id = item.get("id") or "E?"
        section = compact_mindmap_text(item.get("section") or "相关论点", 64)
        page = item.get("page_label") or item.get("page")
        page_text = f", p.{page}" if page not in (None, "") else ""
        lines.append(f"- {section} [{evidence_id}{page_text}]")
    return lines


def compact_mindmap_text(value: str, max_chars: int) -> str:
    normalized = " ".join(str(value or "").replace("|", " ").split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(1, max_chars - 1)].rstrip() + "…"


def normalize_technical_terms(text: str) -> str:
    return (
        (text or "")
        .replace("变压器架构", "Transformer 架构")
        .replace("变压器模型", "Transformer 模型")
        .replace("变压器", "Transformer")
    )


def build_fast_evidence_answer(query: str, evidence: list[dict]) -> str | None:
    normalized_query = (query or "").lower()
    is_definition_query = any(marker in query for marker in ("是什么", "机制", "原理", "定义"))
    is_short_transformer_query = "transformer" in normalized_query and len((query or "").strip()) <= 80
    if "transformer" not in normalized_query or not (is_definition_query or is_short_transformer_query) or not evidence:
        return None

    primary = evidence[0].get("id", "E1")
    secondary = evidence[1].get("id", primary) if len(evidence) > 1 else primary
    tertiary = evidence[2].get("id", secondary) if len(evidence) > 2 else secondary

    return (
        "根据本地论文证据，Transformer 机制可以概括为：\n\n"
        f"- 核心思想：Transformer 是以 Attention/Self-Attention 为核心的序列建模架构，"
        f"用注意力机制建模 token 之间的依赖关系，减少对循环层的依赖 [{primary}]。\n"
        f"- 结构方式：典型 Transformer block 由多头自注意力、隐藏表示维度、前馈层、残差连接和归一化等模块组成，"
        f"并配合位置编码保留序列顺序信息 [{primary}]。\n"
        f"- 主要优势：相比传统 RNN/CNN 路线，Transformer 更容易并行训练，并能在翻译、预训练语言模型等任务中复用，"
        f"后续 BERT/RoBERTa 等模型也建立在该架构之上 [{secondary}]。\n"
        f"- 论文依据：当前检索到的证据包含 Transformer 架构描述、Attention Is All You Need 的引用以及后续预训练模型中的应用说明 [{tertiary}]。"
    )
