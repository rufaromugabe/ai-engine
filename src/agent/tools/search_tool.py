from .base_tool import BaseTool, ToolResult
from typing import Dict, Any, Optional

class SearchTool(BaseTool):
    def __init__(self, organization_id: str, workspace_id: Optional[str] = None):
        super().__init__(
            name="search_tool",
            description="Simulate a web search and return canned results for testing.",
            organization_id=organization_id,
            workspace_id=workspace_id
        )

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def execute(self, query: str) -> ToolResult:
        if not self._initialized:
            return ToolResult(success=False, data=None, error="Search tool not initialized")
        # Simulate search results
        results = [
            {"title": f"Result 1 for {query}", "snippet": f"This is a snippet for {query} (1)."},
            {"title": f"Result 2 for {query}", "snippet": f"This is a snippet for {query} (2)."},
            {"title": f"Result 3 for {query}", "snippet": f"This is a snippet for {query} (3)."}
        ]
        return ToolResult(success=True, data={"results": results}, error=None)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        } 