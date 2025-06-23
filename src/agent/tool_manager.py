"""Tool manager for dynamic tool loading and management."""

import logging
from typing import Dict, List, Optional, Any, Type, Tuple
from dataclasses import dataclass

from .tools.base_tool import BaseTool, ToolResult
from .tools.rag_tool import RAGTool
from .config import config_manager, ToolType
from .tools.calculator_tool import CalculatorTool
from .tools.search_tool import SearchTool

logger = logging.getLogger(__name__)

@dataclass
class ToolExecutionResult:
    """Result from tool execution with additional context."""
    tool_name: str
    success: bool
    result: ToolResult
    execution_time: float
    organization_id: str

class ToolManager:
    """Manages tools for the AI agent system. Supports dynamic registration and unregistration."""
    
    def __init__(self):
        self._tool_registry: Dict[ToolType, Type[BaseTool]] = {}
        self._active_tools: Dict[str, Dict[str, BaseTool]] = {}  # {org_id_workspace_id: {tool_name: tool_instance}}
        self._organization_tools: Dict[str, Dict[str, BaseTool]] = {}  # {org_id: {tool_name: tool_instance}}
        self._tool_schema_cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}  # (org_id, ws_id) -> schemas
        self._register_default_tools()
    
    def _get_tool_key(self, organization_id: str, workspace_id: Optional[str] = None) -> str:
        """Generate a unique key for tool storage."""
        if workspace_id:
            return f"{organization_id}_{workspace_id}"
        return organization_id
    
    def _get_cache_key(self, organization_id: str, workspace_id: Optional[str]) -> Tuple[str, str]:
        return (organization_id, workspace_id or "__none__")
    
    def _register_default_tools(self) -> None:
        """Register default tools."""
        self._tool_registry[ToolType.RAG] = RAGTool
        self._tool_registry[ToolType.CALCULATOR] = CalculatorTool
        self._tool_registry[ToolType.SEARCH] = SearchTool
        
        # Additional tools will be registered here as they are implemented
    
    def register_tool(self, tool_type: ToolType, tool_class: Type[BaseTool]) -> None:
        """Dynamically register a new tool type."""
        self._tool_registry[tool_type] = tool_class
        self._tool_schema_cache.clear()
        logger.info(f"Registered tool type: {tool_type.value}")
    
    def unregister_tool(self, tool_type: ToolType) -> None:
        """Dynamically unregister a tool type."""
        if tool_type in self._tool_registry:
            del self._tool_registry[tool_type]
            self._tool_schema_cache.clear()
            logger.info(f"Unregistered tool type: {tool_type.value}")
    
    def reload_tools(self) -> None:
        """Reload tools from config or API (stub for future extension)."""
        self._register_default_tools()
        self._tool_schema_cache.clear()
        # In the future, load additional tools from config or external API
    
    async def initialize_tools_for_organization(
        self, 
        organization_id: str, 
        workspace_id: Optional[str] = None,
        tools: Optional[List[ToolType]] = None
    ) -> bool:
        """Initialize tools for an organization or workspace."""
        try:
            tool_key = self._get_tool_key(organization_id, workspace_id)
            
            if tool_key not in self._active_tools:
                self._active_tools[tool_key] = {}
            
            # Get configuration
            config = config_manager.get_organization_config(organization_id)
            
            # Determine which tools to initialize
            if tools:
                tools_to_init = tools
            else:
                tools_to_init = config.enabled_tools
            
            # Initialize enabled tools
            initialization_results = []
            for tool_type in tools_to_init:
                if tool_type in self._tool_registry:
                    tool_class = self._tool_registry[tool_type]
                    tool_instance = tool_class(
                        organization_id=organization_id,
                        workspace_id=workspace_id
                    )
                    
                    success = await tool_instance.initialize()
                    if success:
                        self._active_tools[tool_key][tool_type.value] = tool_instance
                        
                        # Also store in organization tools if no workspace specified
                        if not workspace_id:
                            if organization_id not in self._organization_tools:
                                self._organization_tools[organization_id] = {}
                            self._organization_tools[organization_id][tool_type.value] = tool_instance
                        
                        logger.info(f"Initialized {tool_type.value} for {tool_key}")
                    else:
                        logger.error(f"Failed to initialize {tool_type.value} for {tool_key}")
                    
                    initialization_results.append(success)
                else:
                    logger.warning(f"Tool type {tool_type} not found in registry")
                    initialization_results.append(False)
            
            self._tool_schema_cache.pop(self._get_cache_key(organization_id, workspace_id), None)
            return all(initialization_results)
            
        except Exception as e:
            logger.error(f"Error initializing tools for {tool_key}: {str(e)}")
            return False
    
    async def execute_tool(
        self, 
        organization_id: str, 
        tool_name: str, 
        workspace_id: Optional[str] = None,
        **kwargs
    ) -> ToolExecutionResult:
        """Execute a specific tool for an organization/workspace."""
        import time
        
        start_time = time.time()
        tool_key = self._get_tool_key(organization_id, workspace_id)
        
        try:
            # Check if tool is available for the organization/workspace
            if (tool_key not in self._active_tools or 
                tool_name not in self._active_tools[tool_key]):
                
                # Try fallback to organization-level tools if workspace tool not found
                if workspace_id and organization_id in self._organization_tools:
                    if tool_name in self._organization_tools[organization_id]:
                        tool = self._organization_tools[organization_id][tool_name]
                        result = await tool.execute(**kwargs)
                        return ToolExecutionResult(
                            tool_name=tool_name,
                            success=result.success,
                            result=result,
                            execution_time=time.time() - start_time,
                            organization_id=organization_id
                        )
                
                result = ToolResult(
                    success=False,
                    data=None,
                    error=f"Tool {tool_name} not available for {tool_key}"
                )
                
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=result,
                    execution_time=time.time() - start_time,
                    organization_id=organization_id
                )
            
            # Execute the tool
            tool = self._active_tools[tool_key][tool_name]
            result = await tool.execute(**kwargs)
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=result.success,
                result=result,
                execution_time=time.time() - start_time,
                organization_id=organization_id
            )
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name} for {tool_key}: {str(e)}")
            
            result = ToolResult(
                success=False,
                data=None,
                error=f"Tool execution error: {str(e)}"
            )
            
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=result,
                execution_time=time.time() - start_time,
                organization_id=organization_id            )
    
    def get_available_tools(self, organization_id: str, workspace_id: Optional[str] = None) -> List[str]:
        """Get list of available tools for an organization/workspace."""
        tool_key = self._get_tool_key(organization_id, workspace_id)
        
        if tool_key in self._active_tools:
            return list(self._active_tools[tool_key].keys())
        
        # Fallback to organization-level tools
        if workspace_id and organization_id in self._organization_tools:
            return list(self._organization_tools[organization_id].keys())
        
        return []
    
    def get_tool_schemas(self, organization_id: str, workspace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get schemas for all available tools for an organization/workspace, with caching."""
        cache_key = self._get_cache_key(organization_id, workspace_id)
        if cache_key in self._tool_schema_cache:
            return self._tool_schema_cache[cache_key]
        tool_key = self._get_tool_key(organization_id, workspace_id)
        schemas = []
        
        if tool_key in self._active_tools:
            for tool in self._active_tools[tool_key].values():
                schemas.append(tool.get_schema())
        elif workspace_id and organization_id in self._organization_tools:
            # Fallback to organization-level tools
            for tool in self._organization_tools[organization_id].values():
                schemas.append(tool.get_schema())
        
        self._tool_schema_cache[cache_key] = schemas
        return schemas
    
    def is_tool_available(self, organization_id: str, tool_name: str, workspace_id: Optional[str] = None) -> bool:
        """Check if a tool is available for an organization/workspace."""
        tool_key = self._get_tool_key(organization_id, workspace_id)
        
        available = (tool_key in self._active_tools and 
                    tool_name in self._active_tools[tool_key])
        
        # Check organization-level tools as fallback
        if not available and workspace_id:
            available = (organization_id in self._organization_tools and 
                        tool_name in self._organization_tools[organization_id])
        
        return available
    
    async def cleanup_organization_tools(self, organization_id: str, workspace_id: Optional[str] = None) -> None:
        """Cleanup tools for an organization/workspace."""
        tool_key = self._get_tool_key(organization_id, workspace_id)
        
        if tool_key in self._active_tools:
            for tool in self._active_tools[tool_key].values():
                await tool.cleanup()
            del self._active_tools[tool_key]
            logger.info(f"Cleaned up tools for {tool_key}")
        
        # If cleaning up organization, also clean organization tools
        if not workspace_id and organization_id in self._organization_tools:
            for tool in self._organization_tools[organization_id].values():
                await tool.cleanup()
            del self._organization_tools[organization_id]
    
    async def reload_organization_tools(self, organization_id: str, workspace_id: Optional[str] = None) -> bool:
        """Reload tools for an organization/workspace (useful when configuration changes)."""
        await self.cleanup_organization_tools(organization_id, workspace_id)
        return await self.initialize_tools_for_organization(organization_id, workspace_id)
    
    def get_tool_instance(self, organization_id: str, tool_name: str, workspace_id: Optional[str] = None) -> Optional[BaseTool]:
        """Get a specific tool instance for an organization/workspace."""
        tool_key = self._get_tool_key(organization_id, workspace_id)
        
        if (tool_key in self._active_tools and 
            tool_name in self._active_tools[tool_key]):
            return self._active_tools[tool_key][tool_name]
        
        # Fallback to organization-level tools
        if workspace_id and organization_id in self._organization_tools:
            if tool_name in self._organization_tools[organization_id]:
                return self._organization_tools[organization_id][tool_name]
        
        return None

# Global tool manager instance
tool_manager = ToolManager()
