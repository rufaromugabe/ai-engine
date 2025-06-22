"""LangGraph entry point for development and deployment."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, add_messages

# Add the src directory to Python path for imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from .enhanced_graph import (
        AgentState,
        WorkspaceAgentConfiguration,
        enhanced_graph,
        # Import the new functions
        intelligent_router,
        execute_tools,
        reflection_node,
        generate_response,
        ensure_dev_organization_exists
    )
except ImportError:
    # Fallback for when relative imports don't work
    from agent.enhanced_graph import (
        AgentState,
        WorkspaceAgentConfiguration,
        enhanced_graph,
        # Import the new functions
        intelligent_router,
        execute_tools,
        reflection_node,
        generate_response,
        ensure_dev_organization_exists
    )

logger = logging.getLogger(__name__)

# Export the main graph for LangGraph development
graph = enhanced_graph

# Additional configuration for development
def create_dev_config() -> Dict[str, Any]:
    """Create development configuration with sensible defaults."""
    return {
        "configurable": {
            "organization_id": os.getenv("DEFAULT_ORG_ID", "dev-org"),
            "workspace_id": os.getenv("DEFAULT_WORKSPACE_ID", "dev-workspace"),
            "agent_id": os.getenv("DEFAULT_AGENT_ID", "dev-agent")
        }
    }

def get_safe_config(config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Get configuration with safe fallbacks for LangGraph Studio."""
    if config and config.get("configurable"):
        return config["configurable"]
    
    # Fallback to environment variables if no config provided
    return {
        "organization_id": os.getenv("DEFAULT_ORG_ID", "dev-org"),
        "workspace_id": os.getenv("DEFAULT_WORKSPACE_ID", "dev-workspace"),
        "agent_id": os.getenv("DEFAULT_AGENT_ID", "dev-agent")
    }

async def test_query(query: str = "What is LangGraph?") -> Dict[str, Any]:
    """Test function for development."""
    config = create_dev_config()
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "organization_id": config["configurable"]["organization_id"],
        "workspace_id": config["configurable"]["workspace_id"],
        "agent_id": config["configurable"]["agent_id"],
        "current_tool_calls": [],
        "tool_results": [],
        "user_query": query,
        "context": {},
        "iteration_count": 0,
        "reflection_history": [],
        "routing_history": []
    }
    
    try:
        result = await graph.ainvoke(initial_state, config)
        return {
            "success": True,
            "messages": [msg.content for msg in result["messages"]],
            "context": result.get("context", {})
        }
    except Exception as e:
        logger.error(f"Test query failed: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

# Configuration schema for LangGraph Studio
from langchain_core.runnables import RunnableConfig
from typing import Optional

def get_configuration_schema() -> Dict[str, Any]:
    """Get the configuration schema for LangGraph Studio."""
    return {
        "type": "object",
        "properties": {
            "organization_id": {
                "type": "string",
                "description": "Organization identifier",
                "default": os.getenv("DEFAULT_ORG_ID", "dev-org")
            },
            "workspace_id": {
                "type": "string",
                "description": "Workspace identifier (optional)",
                "default": os.getenv("DEFAULT_WORKSPACE_ID", "dev-workspace")
            },
            "agent_id": {
                "type": "string", 
                "description": "Agent identifier (optional)",
                "default": os.getenv("DEFAULT_AGENT_ID", "dev-agent")
            },
            "custom_instructions": {
                "type": "string",
                "description": "Custom instructions for the agent",
                "default": ""
            }
        },
        "required": ["organization_id"]
    }

async def initialize_with_config(config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """Initialize the graph with proper configuration for LangGraph Studio."""
    safe_config = get_safe_config(config)
    
    # Create initial state with safe defaults
    initial_state = {
        "messages": [],
        "organization_id": safe_config["organization_id"],
        "workspace_id": safe_config.get("workspace_id"),
        "agent_id": safe_config.get("agent_id"),
        "current_tool_calls": [],
        "tool_results": [],
        "user_query": "",
        "context": {},
        "iteration_count": 0,
        "reflection_history": [],
        "routing_history": []
    }
    
    return initial_state

# Export the configuration schema
configuration_schema = get_configuration_schema()

# Additional utility functions for development
def get_graph_info() -> Dict[str, Any]:
    """Get information about the graph structure."""
    return {
        "nodes": list(graph.nodes.keys()) if hasattr(graph, 'nodes') else [],
        "edges": list(graph.edges.keys()) if hasattr(graph, 'edges') else [],
        "config_schema": WorkspaceAgentConfiguration.__annotations__
    }

async def test_studio_config(query: str = "What is LangGraph?") -> Dict[str, Any]:
    """Test function specifically for LangGraph Studio with explicit config."""
    # Force the development configuration
    config = {
        "configurable": {
            "organization_id": "dev-org",
            "workspace_id": "dev-workspace",
            "agent_id": "dev-agent"
        }
    }
    
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "organization_id": "dev-org",
        "workspace_id": "dev-workspace",
        "agent_id": "dev-agent",
        "current_tool_calls": [],
        "tool_results": [],
        "user_query": query,
        "context": {},
        "iteration_count": 0,
        "reflection_history": [],
        "routing_history": []
    }
    
    try:
        logger.info(f"Testing with config: {config}")
        result = await graph.ainvoke(initial_state, config)
        return {
            "success": True,
            "messages": [msg.content for msg in result["messages"]],
            "context": result.get("context", {}),
            "config_used": config
        }
    except Exception as e:
        logger.error(f"Studio test query failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "config_used": config
        }

# Alias for studio compatibility
studio_test = test_studio_config

# Export for LangGraph Studio and development
__all__ = [
    "graph", 
    "test_query", 
    "test_studio_config",
    "studio_test",
    "get_graph_info", 
    "create_dev_config", 
    "get_safe_config",
    "configuration_schema",
    "initialize_with_config",
    "get_configuration_schema"
]
