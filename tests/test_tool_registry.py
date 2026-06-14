import unittest

from core.tool_registry import build_default_tool_registry


class ToolRegistryTests(unittest.TestCase):
    def test_default_registry_can_include_audit_verify_tool(self):
        registry = build_default_tool_registry(
            runtime_status_handler=lambda _arguments: {"locked": False},
            audit_recent_handler=lambda _arguments: {"events": []},
            workspace_diagnostics_handler=lambda _arguments: {"ready": True},
            product_diagnostics_handler=lambda _arguments: {"ready": True},
            audit_verify_handler=lambda _arguments: {"ready": True},
        )

        names = {tool["name"] for tool in registry.list_tools()}
        result = registry.call("souldrive.audit_verify", {"limit": 10})

        self.assertEqual(names, {
            "souldrive.audit_recent",
            "souldrive.audit_verify",
            "souldrive.product_diagnostics",
            "souldrive.runtime_status",
            "souldrive.workspace_diagnostics",
        })
        self.assertEqual(result, {"ready": True})


if __name__ == "__main__":
    unittest.main()
