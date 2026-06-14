from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]

    def public_spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        self._tools[spec.name] = spec

    def list_tools(self) -> list[dict[str, Any]]:
        return [spec.public_spec() for spec in sorted(self._tools.values(), key=lambda item: item.name)]

    def call(self, name: str, arguments: dict[str, Any] | None = None):
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name].handler(arguments or {})


def build_default_tool_registry(
    runtime_status_handler: Callable[[dict[str, Any]], Any],
    audit_recent_handler: Callable[[dict[str, Any]], Any],
    workspace_diagnostics_handler: Callable[[dict[str, Any]], Any],
    product_diagnostics_handler: Callable[[dict[str, Any]], Any] | None = None,
    audit_verify_handler: Callable[[dict[str, Any]], Any] | None = None,
):
    registry = ToolRegistry()
    registry.register(ToolSpec(
        name="souldrive.runtime_status",
        description="Return the current redacted SoulDrive runtime state.",
        input_schema={"type": "object", "properties": {}},
        handler=runtime_status_handler,
    ))
    registry.register(ToolSpec(
        name="souldrive.audit_recent",
        description="Return recent hash-chain audit events.",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
        },
        handler=audit_recent_handler,
    ))
    if audit_verify_handler is not None:
        registry.register(ToolSpec(
            name="souldrive.audit_verify",
            description="Verify the local hash-chain audit log and return tamper-evidence status.",
            input_schema={
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 10000}},
            },
            handler=audit_verify_handler,
        ))
    registry.register(ToolSpec(
        name="souldrive.workspace_diagnostics",
        description="Return local workspace layout diagnostics.",
        input_schema={"type": "object", "properties": {}},
        handler=workspace_diagnostics_handler,
    ))
    if product_diagnostics_handler is not None:
        registry.register(ToolSpec(
            name="souldrive.product_diagnostics",
            description="Return local runtime diagnostics for SoulDrive.",
            input_schema={"type": "object", "properties": {}},
            handler=product_diagnostics_handler,
        ))
    return registry
