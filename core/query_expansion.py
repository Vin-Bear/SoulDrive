TERM_ALIASES = (
    ("异常检测", "anomaly detection"),
    ("工业控制系统", "industrial control systems ICS"),
    ("时序", "time series"),
    ("时间序列", "time series"),
    ("生成对抗网络", "GAN generative adversarial network"),
    ("知识管理", "knowledge management"),
    ("知识库", "knowledge base"),
    ("检索增强", "RAG retrieval augmented generation"),
    ("分类分级", "data classification grading"),
    ("个人信息", "personal information"),
)


def expand_query_variants(query: str, limit: int = 3) -> list[str]:
    base = (query or "").strip()
    if not base:
        return []

    variants = [base]
    lowered = base.lower()

    if any(token in base for token in ("怎么", "如何")):
        variants.append(base.replace("怎么", "实现方式").replace("如何", "实现方式"))

    if "隐私" in base:
        variants.append(base.replace("隐私", "数据安全与隐私保护"))
    elif "安全" in base:
        variants.append(base.replace("安全", "风险控制与安全机制"))

    if "是什么" in base:
        variants.append(base.replace("是什么", "定义与核心机制"))

    if "graphrag" in lowered and "local search" in lowered:
        variants.append(base.replace("怎么工作", "核心机制").replace("如何工作", "核心机制"))

    alias_query = build_alias_query(base)
    if alias_query:
        variants.append(alias_query)

    deduped = []
    seen = set()
    for item in variants:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
        if len(deduped) >= limit:
            break
    return deduped


def build_alias_query(query: str) -> str | None:
    aliases = []
    for term, alias in TERM_ALIASES:
        if term in query:
            aliases.append(alias)
    if not aliases:
        return None
    return " ".join([query, *aliases])
