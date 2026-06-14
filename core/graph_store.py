from typing import Protocol


class GraphStore(Protocol):
    def add_entity(self, name: str, entity_type: str, description: str):
        ...

    def add_relationship(self, source: str, target: str, relation: str):
        ...

    def get_all_entities(self) -> list[str]:
        ...

    def get_related_entities(self, entity_name: str, limit: int = 20) -> list[tuple[str, str, str]]:
        ...

    def get_subgraph(self, entity_name: str, depth: int = 1) -> list[tuple[str, str, str]]:
        ...

    def close(self):
        ...
