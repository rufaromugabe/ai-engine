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
        generate_response
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
        generate_response
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

# Additional utility functions for development
def get_graph_info() -> Dict[str, Any]:
    """Get information about the graph structure."""
    return {
        "nodes": list(graph.nodes.keys()) if hasattr(graph, 'nodes') else [],
        "edges": list(graph.edges.keys()) if hasattr(graph, 'edges') else [],
        "config_schema": WorkspaceAgentConfiguration.__annotations__
    }

# Export for LangGraph Studio and development
__all__ = ["graph", "test_query", "get_graph_info", "create_dev_config"]
