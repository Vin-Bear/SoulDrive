import os
from pathlib import Path
from core.graph_store import GraphStore
from core.keyword_index import KeywordIndex
from core.paths import resolve_model_path
from core.parent_child_index import split_parent_document
from core.parent_doc_store import ParentDocumentStore
from core.retrieval import (
    SearchCandidate,
    build_citations,
    rank_hybrid_candidates,
)
from core.logging_config import get_logger
from core.observability import runtime_metrics
from core.workspace import SoulDriveWorkspace

logger = get_logger(__name__)

# 动态获取项目根目录的绝对路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "souldrive_db")


def _persistent_client(path: str):
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )


def _sentence_transformer(model_path: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_path, local_files_only=True, device="cpu")


class LocalKnowledgeBase:
    def __init__(
        self,
        db_path: str | None = None,
        parent_doc_path: str | None = None,
        keyword_index_path: str | None = None,
        workspace_path: str | None = None,
    ):
        workspace = SoulDriveWorkspace.default().ensure() if db_path is None else None
        db_path = db_path or workspace.chroma_path
        index_root = Path(db_path).parent
        workspace_path = workspace_path or str(index_root.parent)
        parent_doc_path = parent_doc_path or str(index_root / "parent_docs.sqlite")
        keyword_index_path = keyword_index_path or str(index_root / "keyword_index.sqlite")

        # 1. 初始化本地 ChromaDB，锁定绝对路径
        logger.info("[VectorDB] 正在挂载离线知识图谱，锁定绝对路径: %s", db_path)
        self.client = _persistent_client(path=db_path)
        self.collection = self.client.get_or_create_collection(name="souldrive_papers")
        self.parent_store = ParentDocumentStore(parent_doc_path)
        self.keyword_index = KeywordIndex(keyword_index_path)

        # 2. 加载轻量级中文嵌入模型
        logger.info("[VectorDB] 正在加载 BGE-Small 嵌入模型...")
        BGE_MODEL_PATH = resolve_model_path("bge-small-zh-v1.5", workspace_path)

        if not os.path.exists(BGE_MODEL_PATH):
            raise FileNotFoundError(f"找不到本地嵌入模型，请检查路径: {BGE_MODEL_PATH}")

        # 强制从本地路径加载，杜绝一切网络请求
        self.embedding_model = _sentence_transformer(BGE_MODEL_PATH)
        logger.info("BGE-Small 嵌入模型加载成功!")

    def ingest_chunks(self, document_id: str, chunks: list):
        if not chunks:
            return

        texts = []
        metadatas = []
        ids = []

        for parent_index, chunk in enumerate(chunks):
            parent_id = f"{document_id}_parent_{parent_index}"
            parent_content = chunk.page_content
            parent_metadata = {
                **chunk.metadata,
                "parent_id": parent_id,
                "parent_index": parent_index,
            }
            self.parent_store.upsert_parent(parent_id, parent_content, parent_metadata)

            children = split_parent_document(parent_id, parent_content)
            for child in children:
                child_metadata = {
                    **parent_metadata,
                    "parent_id": parent_id,
                    "child_id": child.child_id,
                    "child_index": child.child_index,
                    "child_start_char": child.start_char,
                    "child_end_char": child.end_char,
                    "child_excerpt": child.content[:420],
                }
                texts.append(child.content)
                metadatas.append(child_metadata)
                ids.append(child.child_id)
                self.keyword_index.upsert_document(child.child_id, child.content, child_metadata)

        if not texts:
            logger.info("[VectorDB] %s 未生成可索引子块，已跳过向量入库。", document_id)
            return

        logger.info("[VectorDB] 正在计算 %s 个文本块的向量...", len(texts))
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True).tolist()

        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        logger.info("[VectorDB] %s 成功入库！", document_id)

    def search_with_evidence(self, query: str, graph_db: GraphStore | None = None, top_k: int = 3):
        """
        企业级混合检索：Dense 向量召回 + BM25 关键词召回 + 图谱增强 + 轻量重排。
        """
        query_vector = self.embedding_model.encode([query], normalize_embeddings=True).tolist()
        dense_pool_size = max(top_k * 4, 12)

        dense_candidates = self._dense_candidates(query_vector, dense_pool_size)
        keyword_candidates = self._keyword_candidates(query, dense_pool_size)
        candidates = self._merge_candidates(dense_candidates, keyword_candidates)

        graph_context, matched_entities = self._build_graph_context(
            query=query,
            docs=[candidate.content for candidate in dense_candidates[:2]],
            graph_db=graph_db,
        )

        if matched_entities:
            for candidate in candidates:
                candidate.graph_score = sum(
                    1 for entity in matched_entities if entity.lower() in candidate.content.lower()
                )

        reranked_candidates = rank_hybrid_candidates(query, candidates, top_k=max(top_k * 2, top_k))
        parent_candidates = self._expand_parent_context(reranked_candidates, top_k=top_k)

        return {
            "documents": [candidate.content for candidate in parent_candidates],
            "metadatas": [candidate.metadata for candidate in parent_candidates],
            "graph_context": graph_context,
            "matched_entities": matched_entities,
            "evidence": build_citations(parent_candidates),
            "retrieval_mode": "parent_child_hybrid_dense_fts_graph_rrf",
        }

    def search(self, query: str, graph_db: GraphStore | None = None, top_k: int = 3):
        """
        兼容旧调用方：返回文本片段、元数据和图谱上下文。
        """
        result = self.search_with_evidence(query=query, graph_db=graph_db, top_k=top_k)
        return result["documents"], result["metadatas"], result["graph_context"]

    def _dense_candidates(self, query_vector: list[list[float]], n_results: int):
        try:
            vector_results = self.collection.query(
                query_embeddings=query_vector,
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            runtime_metrics.increment("retrieval_failures")
            runtime_metrics.record_error(f"dense retrieval failed: {exc}")
            logger.warning("[VectorDB] Dense 检索降级: %s", exc)
            return []

        documents = vector_results.get("documents", [[]])[0] or []
        metadatas = vector_results.get("metadatas", [[]])[0] or []
        distances = vector_results.get("distances", [[]])[0] or []
        ids = vector_results.get("ids", [[]])[0] or []

        candidates = []
        for index, document in enumerate(documents):
            candidates.append(SearchCandidate(
                id=ids[index] if index < len(ids) else f"dense_{index}",
                content=document,
                metadata=metadatas[index] if index < len(metadatas) and metadatas[index] else {},
                dense_rank=index + 1,
                vector_distance=distances[index] if index < len(distances) else None,
            ))
        return candidates

    def _keyword_candidates(self, query: str, n_results: int):
        results = self.keyword_index.search(query, limit=n_results)
        return [
            SearchCandidate(
                id=result.doc_id,
                content=result.content,
                metadata=result.metadata,
                sparse_rank=index + 1,
                keyword_score=result.score,
            )
            for index, result in enumerate(results)
        ]

    def _merge_candidates(self, *candidate_groups: list[SearchCandidate]):
        merged: dict[str, SearchCandidate] = {}
        for group in candidate_groups:
            for candidate in group:
                if candidate.id not in merged:
                    merged[candidate.id] = candidate
                    continue

                existing = merged[candidate.id]
                if existing.dense_rank is None:
                    existing.dense_rank = candidate.dense_rank
                if existing.sparse_rank is None:
                    existing.sparse_rank = candidate.sparse_rank
                existing.keyword_score = max(existing.keyword_score, candidate.keyword_score)
                if existing.vector_distance is None:
                    existing.vector_distance = candidate.vector_distance

        return list(merged.values())

    def _expand_parent_context(self, candidates: list[SearchCandidate], top_k: int):
        expanded = []
        seen_parent_ids = set()

        for candidate in candidates:
            parent_id = candidate.metadata.get("parent_id")
            if not parent_id or parent_id in seen_parent_ids:
                continue

            parent = self.parent_store.get_parent(parent_id)
            if not parent:
                expanded.append(candidate)
                seen_parent_ids.add(parent_id)
                continue

            parent_candidate = SearchCandidate(
                id=candidate.id,
                content=parent["content"],
                metadata={
                    **parent["metadata"],
                    **candidate.metadata,
                },
                dense_rank=candidate.dense_rank,
                sparse_rank=candidate.sparse_rank,
                graph_score=candidate.graph_score,
                vector_distance=candidate.vector_distance,
                keyword_score=candidate.keyword_score,
                rrf_score=candidate.rrf_score,
                rerank_score=candidate.rerank_score,
                final_score=candidate.final_score,
                score_breakdown=candidate.score_breakdown,
            )
            expanded.append(parent_candidate)
            seen_parent_ids.add(parent_id)

            if len(expanded) >= top_k:
                break

        return expanded[:top_k]

    def _build_graph_context(self, query: str, docs: list[str], graph_db: GraphStore | None = None):
        if not graph_db:
            return [], []

        try:
            # ==========================================
            # 步骤 1：构建匹配空间 (Match Space)
            # ==========================================
            # 【核心技巧】反向子串匹配：从图库中拉取所有实体名，在 query 与高相关文档片段中找命中实体。
            # 这样 0 显存损耗（无需跑额外的 NER 模型），也能覆盖用户没有直接点名实体、但向量召回已命中文档主题的场景。

            # 将用户的提问 (query) 和向量检索召回的前 2 篇最相关文档 (docs[:2]) 拼接在一起。
            # 统一转换为小写 (.lower())，以实现大小写不敏感的匹配，提升命中率。
            graph_match_space = "\n".join([query, *docs[:2]]).lower()

            # 从图谱数据库 (graph_db) 中全量拉取所有的实体列表
            all_entities = graph_db.get_all_entities()

            # ==========================================
            # 步骤 2：反向匹配实体
            # ==========================================
            matched_entities = []
            for entity in all_entities:
                # 遍历所有图谱实体，检查实体名（转小写后）是否作为子串出现在我们的匹配空间中
                if entity and entity.lower() in graph_match_space:
                    matched_entities.append(entity)

                # 【截断策略】为了防止匹配到过多的泛实体导致后续查询爆炸，这里限制最多只取前 8 个命中的实体
                if len(matched_entities) >= 8:
                    break

            # ==========================================
            # 步骤 3：查询图谱拓扑关系并构建上下文
            # ==========================================
            # 使用集合 (set) 来记录已经添加过的关系，防止不同实体查出重复的边（例如 A-B 和 B-A 查出同一条边）
            seen_relations = set()
            graph_context = []

            # 遍历刚才匹配到的实体列表
            for entity in matched_entities:
                # 针对每个实体，向图谱数据库（如 SQLite/Neo4j）查询深度为 1 (depth=1) 的子图
                # 即只查询与该实体直接相连的一阶邻居节点
                relations = graph_db.get_subgraph(entity, depth=1)

                # 遍历该实体返回的所有关系三元组 (头实体, 关系, 尾实体)
                for source, relation, target in relations:
                    # 将三元组打包成元组作为去重 key
                    relation_key = (source, relation, target)

                    # 如果这个关系已经被处理过，则跳过
                    if relation_key in seen_relations:
                        continue

                    # 将新关系加入去重集合
                    seen_relations.add(relation_key)

                    # 将结构化的三元组转换为大模型容易理解的自然语言格式，并存入 graph_context
                    graph_context.append(f"已知逻辑关系: [{source}] --({relation})--> [{target}]")

                    # 【截断策略】限制图谱上下文的总条数不超过 24 条
                    # 防止注入的 prompt 过长，超出大模型上下文窗口或稀释核心注意力
                    if len(graph_context) >= 24:
                        break

                # 外层循环也要判断，一旦达到 24 条关系，立刻停止查询其他实体的子图
                if len(graph_context) >= 24:
                    break

        except Exception as e:
            runtime_metrics.record_error(f"graph retrieval failed: {e}")
            # 容错处理：图谱查询如果发生异常（如数据库断连、查询超时），不会导致整个程序崩溃
            # 而是打印警告日志，并进行功能降级（即 graph_context 返回空列表，仅依赖常规文本检索）
            logger.warning("[Warning] 图谱检索降级，原因: %s", e)
            return [], []

        return graph_context, matched_entities

    def delete_document_indexes(self, document_hash: str):
        self.parent_store.delete_by_document_hash(document_hash)
        self.keyword_index.delete_by_document_hash(document_hash)

    def close(self):
        self.parent_store.close()
        self.keyword_index.close()
