import json
import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Any


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class KeywordSearchResult:
    doc_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class KeywordIndex:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS keyword_documents
            USING fts5(doc_id UNINDEXED, content, metadata_json UNINDEXED)
        """)
        self.conn.commit()

    def upsert_document(self, doc_id: str, content: str, metadata: dict[str, Any]):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM keyword_documents WHERE doc_id = ?", (doc_id,))
        cursor.execute("""
            INSERT INTO keyword_documents (doc_id, content, metadata_json)
            VALUES (?, ?, ?)
        """, (doc_id, content, json.dumps(metadata, ensure_ascii=False)))
        self.conn.commit()

    def delete_by_document_hash(self, document_hash: str):
        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM keyword_documents
            WHERE metadata_json LIKE ?
        """, (f'%"{document_hash}"%',))
        self.conn.commit()

    def search(self, query: str, limit: int = 12) -> list[KeywordSearchResult]:
        match_query = self._build_match_query(query)
        if not match_query:
            return []

        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                SELECT doc_id, content, metadata_json, bm25(keyword_documents) AS score
                FROM keyword_documents
                WHERE keyword_documents MATCH ?
                ORDER BY score ASC
                LIMIT ?
            """, (match_query, limit))
        except sqlite3.OperationalError:
            return []

        return [
            KeywordSearchResult(
                doc_id=row[0],
                content=row[1],
                metadata=json.loads(row[2]),
                score=abs(float(row[3])),
            )
            for row in cursor.fetchall()
        ]

    def close(self):
        self.conn.close()

    def _build_match_query(self, query: str):
        tokens = TOKEN_PATTERN.findall((query or "").lower())
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens[:12])
