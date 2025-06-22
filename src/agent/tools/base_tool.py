"""Base tool interface for the AI agent system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseTool(ABC):
    """Base class for all tools in the system."""
    
    def __init__(
        self, 
        name: str, 
        description: str, 
        organization_id: str,
        workspace_id: Optional[str] = None
    ):
        self.name = name
        self.description = description
        self.organization_id = organization_id
        self.workspace_id = workspace_id
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the tool. Return True if successful."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema for LLM function calling."""
        pass
    
    def is_initialized(self) -> bool:
        """Check if the tool is initialized."""
        return self._initialized
    
    async def cleanup(self) -> None:
        """Cleanup resources when tool is no longer needed."""
        pass
    
    def get_context_key(self) -> str:
        """Get unique context key for this tool instance."""
        if self.workspace_id:
            return f"{self.organization_id}_{self.workspace_id}_{self.name}"
        return f"{self.organization_id}_{self.name}"
