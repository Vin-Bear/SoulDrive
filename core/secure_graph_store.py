import base64
import hashlib
import hmac
import json
import os
import sqlite3
from dataclasses import dataclass

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


ENTITY_AAD = b"SoulDrive secure graph entity v1"
RELATIONSHIP_AAD = b"SoulDrive secure graph relationship v1"


@dataclass(frozen=True)
class _EntityRecord:
    name: str
    entity_type: str
    description: str


@dataclass(frozen=True)
class _RelationshipRecord:
    source: str
    relation: str
    target: str


class SecureGraphStore:
    def __init__(self, db_path: str, keys):
        self.db_path = db_path
        self.keys = keys
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY,
                payload_nonce TEXT NOT NULL,
                payload_ciphertext TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY,
                payload_nonce TEXT NOT NULL,
                payload_ciphertext TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def add_entity(self, name: str, entity_type: str, description: str):
        entity_id = self._entity_id(name)
        payload = {
            "name": name,
            "type": entity_type,
            "description": description,
        }
        nonce, ciphertext = self._encrypt_json(payload, ENTITY_AAD + entity_id.encode("ascii"))
        self.conn.execute(
            """
            INSERT OR REPLACE INTO entities (entity_id, payload_nonce, payload_ciphertext)
            VALUES (?, ?, ?)
            """,
            (entity_id, nonce, ciphertext),
        )
        self.conn.commit()

    def add_relationship(self, source: str, target: str, relation: str):
        relationship_id = self._relationship_id(source, relation, target)
        payload = {
            "source": source,
            "target": target,
            "relation": relation,
        }
        nonce, ciphertext = self._encrypt_json(
            payload,
            RELATIONSHIP_AAD + relationship_id.encode("ascii"),
        )
        self.conn.execute(
            """
            INSERT OR REPLACE INTO relationships (relationship_id, payload_nonce, payload_ciphertext)
            VALUES (?, ?, ?)
            """,
            (relationship_id, nonce, ciphertext),
        )
        self.conn.commit()

    def get_all_entities(self) -> list[str]:
        entities = [record.name for record in self._all_entities()]
        return sorted(entities, key=lambda value: (-len(value), value))

    def get_related_entities(self, entity_name: str, limit: int = 20) -> list[tuple[str, str, str]]:
        matches = [
            (record.source, record.relation, record.target)
            for record in self._all_relationships()
            if record.source == entity_name or record.target == entity_name
        ]
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return matches[:limit]

    def get_subgraph(self, entity_name: str, depth: int = 1) -> list[tuple[str, str, str]]:
        if depth < 1:
            return []

        max_edges = 50
        visited_entities = {entity_name}
        visited_edges = set()
        results = []
        frontier = [(entity_name, 0)]

        while frontier and len(results) < max_edges:
            current_entity, current_depth = frontier.pop(0)
            if current_depth >= depth:
                continue

            for source, relation, target in self.get_related_entities(current_entity):
                edge_key = (source, relation, target)
                if edge_key not in visited_edges:
                    visited_edges.add(edge_key)
                    results.append(edge_key)

                for next_entity in (source, target):
                    if next_entity not in visited_entities:
                        visited_entities.add(next_entity)
                        frontier.append((next_entity, current_depth + 1))

                if len(results) >= max_edges:
                    break

        return results

    def search_context(self, query: str, depth: int = 1, max_entities: int = 8, max_edges: int = 24):
        match_space = (query or "").lower()
        matched_entities = []
        for entity in self.get_all_entities():
            if entity and entity.lower() in match_space:
                matched_entities.append(entity)
            if len(matched_entities) >= max_entities:
                break

        graph_context = []
        seen_relations = set()
        for entity in matched_entities:
            for source, relation, target in self.get_subgraph(entity, depth=depth):
                relation_key = (source, relation, target)
                if relation_key in seen_relations:
                    continue
                seen_relations.add(relation_key)
                graph_context.append(f"宸茬煡閫昏緫鍏崇郴: [{source}] --({relation})--> [{target}]")
                if len(graph_context) >= max_edges:
                    return graph_context

        return graph_context

    def close(self):
        self.conn.close()

    def _all_entities(self) -> list[_EntityRecord]:
        rows = self.conn.execute(
            """
            SELECT entity_id, payload_nonce, payload_ciphertext
            FROM entities
            ORDER BY entity_id ASC
            """
        ).fetchall()
        return [
            _EntityRecord(
                name=str(payload.get("name") or ""),
                entity_type=str(payload.get("type") or ""),
                description=str(payload.get("description") or ""),
            )
            for payload in (
                self._decrypt_json(row[1], row[2], ENTITY_AAD + row[0].encode("ascii"))
                for row in rows
            )
        ]

    def _all_relationships(self) -> list[_RelationshipRecord]:
        rows = self.conn.execute(
            """
            SELECT relationship_id, payload_nonce, payload_ciphertext
            FROM relationships
            ORDER BY relationship_id ASC
            """
        ).fetchall()
        return [
            _RelationshipRecord(
                source=str(payload.get("source") or ""),
                relation=str(payload.get("relation") or ""),
                target=str(payload.get("target") or ""),
            )
            for payload in (
                self._decrypt_json(row[1], row[2], RELATIONSHIP_AAD + row[0].encode("ascii"))
                for row in rows
            )
        ]

    def _entity_id(self, name: str) -> str:
        return self._digest("entity:" + name)

    def _relationship_id(self, source: str, relation: str, target: str) -> str:
        return self._digest("rel:" + source + "\0" + relation + "\0" + target)

    def _digest(self, value: str) -> str:
        return hmac.new(self.keys.graph_key, value.encode("utf-8"), hashlib.sha256).hexdigest()

    def _encrypt_json(self, payload: dict, aad: bytes) -> tuple[str, str]:
        nonce = os.urandom(12)
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ciphertext = AESGCM(self.keys.graph_key).encrypt(nonce, encoded, aad)
        return _b64(nonce), _b64(ciphertext)

    def _decrypt_json(self, nonce: str, ciphertext: str, aad: bytes) -> dict:
        try:
            payload = AESGCM(self.keys.graph_key).decrypt(_unb64(nonce), _unb64(ciphertext), aad)
        except InvalidTag as exc:
            raise ValueError("secure graph store authentication failed") from exc
        return json.loads(payload.decode("utf-8"))


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))
