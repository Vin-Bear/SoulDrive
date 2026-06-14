import os
from collections import defaultdict
from typing import Any

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

from core.indexing_errors import classify_indexing_error
from core.logging_config import get_logger

logger = get_logger(__name__)


class AdvancedDocumentParser:
    def __init__(self):
        logger.info("[System] 正在挂载工业级版面分析引擎 (Docling) ...")
        # Docling 默认调用轻量级 ONNX 模型运行在 CPU 上
        # 彻底解放显存，同时具备极强的双栏、表格、公式结构化能力
        self.converter = DocumentConverter()

        # 保留基于语义的 Markdown 切片器，防止段落被打碎
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False
        )
        self.last_error: dict[str, str] | None = None

    def parse_and_chunk(self, pdf_path: str):
        """主入口：基于版面感知 (Layout-Aware) 的极速解析管道"""
        self.last_error = None
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"找不到指定的 PDF 文件: {pdf_path}")

        logger.info("[Parser] 开始执行版面感知解析: %s", os.path.basename(pdf_path))
        logger.info("[Parser] 引擎正在识别双栏、表格、公式并重构 Markdown (100% CPU 运算，保全 VRAM)...")

        try:
            # 这一行代码完成了所有工作：读取PDF -> 目标检测(分栏/表格) -> OCR -> Markdown对齐
            result = self.converter.convert(pdf_path)
            page_markdown = self._export_markdown_by_page(result.document)

            logger.info("[Parser] Markdown 重构完毕，正在执行带页码的语义切片...")
            chunks = self._split_page_markdown(page_markdown)
            if not chunks:
                full_markdown = result.document.export_to_markdown()
                chunks = self.markdown_splitter.split_text(full_markdown)
            logger.info("[Parser] 解析完成！共生成 %s 个高质量语义块。", len(chunks))

            return chunks

        except Exception as e:
            reason = str(e)
            self.last_error = {
                "source_filename": os.path.basename(pdf_path),
                "reason": reason,
                **classify_indexing_error(reason),
            }
            logger.error("[Parser 致命错误] Docling 解析失败: %s", e)
            return []

    def _export_markdown_by_page(self, document) -> dict[int, str]:
        page_parts: dict[int, list[str]] = defaultdict(list)
        last_page: int | None = None

        for raw_entry in document.iterate_items():
            item = raw_entry[0] if isinstance(raw_entry, tuple) else raw_entry
            page = self._page_number(item) or last_page
            markdown = self._item_markdown(document, item)
            if page is None or not markdown:
                continue
            page_parts[page].append(markdown)
            last_page = page

        return {
            page: "\n\n".join(part for part in parts if part.strip()).strip()
            for page, parts in sorted(page_parts.items())
            if any(part.strip() for part in parts)
        }

    def _split_page_markdown(self, page_markdown: dict[int, str]):
        chunks = []
        for page, markdown in page_markdown.items():
            page_chunks = self.markdown_splitter.split_text(markdown)
            for chunk in page_chunks:
                if not chunk.page_content.strip():
                    continue
                chunk.metadata["page"] = page
                chunk.metadata["page_number"] = page
                chunk.metadata["page_label"] = str(page)
                chunks.append(chunk)
        return chunks

    def _item_markdown(self, document, item) -> str:
        label = self._item_label(item)
        if "table" in label and hasattr(item, "export_to_markdown"):
            try:
                return str(item.export_to_markdown(document)).strip()
            except TypeError:
                return str(item.export_to_markdown()).strip()
            except Exception:
                return ""

        text = str(getattr(item, "text", None) or getattr(item, "orig", "") or "").strip()
        if not text:
            return ""

        if "section_header" in label or item.__class__.__name__.lower().endswith("sectionheaderitem"):
            level = self._bounded_heading_level(getattr(item, "level", 1))
            return f"{'#' * level} {text}"
        if "list_item" in label:
            return f"- {text}"
        return text

    def _page_number(self, item) -> int | None:
        for provenance in getattr(item, "prov", []) or []:
            page_no = getattr(provenance, "page_no", None)
            page = self._positive_int(page_no)
            if page is not None:
                return page
        return None

    def _item_label(self, item) -> str:
        label = getattr(item, "label", "")
        return str(getattr(label, "value", label)).lower()

    def _bounded_heading_level(self, value: Any) -> int:
        level = self._positive_int(value) or 1
        return max(1, min(level, 6))

    def _positive_int(self, value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None
