from dataclasses import dataclass


@dataclass(frozen=True)
class ChildDocument:
    parent_id: str
    child_id: str
    content: str
    start_char: int
    end_char: int
    child_index: int


def split_parent_document(
    parent_id: str,
    text: str,
    child_size: int = 900,
    child_overlap: int = 120,
) -> list[ChildDocument]:
    clean_text = (text or "").strip()
    if not clean_text:
        return []
    if child_size <= 0:
        raise ValueError("child_size must be positive")
    if child_overlap < 0 or child_overlap >= child_size:
        raise ValueError("child_overlap must be smaller than child_size")

    children = []
    start = 0
    child_index = 0
    while start < len(clean_text):
        end = min(start + child_size, len(clean_text))
        content = clean_text[start:end].strip()
        if content:
            children.append(ChildDocument(
                parent_id=parent_id,
                child_id=f"{parent_id}_child_{child_index}",
                content=content,
                start_char=start,
                end_char=end,
                child_index=child_index,
            ))
            child_index += 1
        if end >= len(clean_text):
            break
        start = max(0, end - child_overlap)

    return children
