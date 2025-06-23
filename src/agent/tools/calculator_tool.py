from .base_tool import BaseTool, ToolResult
from typing import Dict, Any, Optional

class CalculatorTool(BaseTool):
    def __init__(self, organization_id: str, workspace_id: Optional[str] = None):
        super().__init__(
            name="calculator_tool",
            description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
            organization_id=organization_id,
            workspace_id=workspace_id
        )

    async def initialize(self) -> bool:
        self._initialized = True
        return True

    async def execute(self, operation: str, a: float, b: float) -> ToolResult:
        if not self._initialized:
            return ToolResult(success=False, data=None, error="Calculator tool not initialized")
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return ToolResult(success=False, data=None, error="Division by zero")
                result = a / b
            else:
                return ToolResult(success=False, data=None, error=f"Unknown operation: {operation}")
            return ToolResult(success=True, data={"result": result}, error=None)
        except Exception as e:
            return ToolResult(success=False, data=None, error=str(e))

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {
                            "type": "string",
                            "enum": ["add", "subtract", "multiply", "divide"],
                            "description": "The arithmetic operation to perform"
                        },
                        "a": {"type": "number", "description": "First operand"},
                        "b": {"type": "number", "description": "Second operand"}
                    },
                    "required": ["operation", "a", "b"]
                }
            }
        } 