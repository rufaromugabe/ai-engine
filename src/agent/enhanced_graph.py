"""Enhanced agent graph system for multi-workspace support."""

from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, add_messages

from .config import config_manager
from .tool_manager import tool_manager
from .workspace_manager import workspace_manager, AgentConfig

logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    """Enhanced state for workspace-aware AI agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    organization_id: str
    workspace_id: Optional[str]
    agent_id: Optional[str]
    current_tool: Optional[str]
    tool_results: List[Dict[str, Any]]
    user_query: str
    context: Dict[str, Any]

@dataclass
class WorkspaceAgentConfiguration:
    """Configuration for workspace agent execution."""
    organization_id: str
    workspace_id: Optional[str] = None
    agent_id: Optional[str] = None
    custom_instructions: str = ""
    
async def route_query(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Route the user query to determine which tools to use."""
    # Initialize variables to prevent UnboundLocalError
    org_id = None
    workspace_id = None
    agent_id = None
    user_query = ""
    
    try:
        configuration = config.get("configurable", {})
        org_id = configuration.get("organization_id")
        if not org_id:
            raise ValueError("organization_id is required")
        workspace_id = configuration.get("workspace_id")
        agent_id = configuration.get("agent_id")
        
        # Get the latest user message
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                user_query = message.content
                break
        
        # Get agent configuration if specified
        agent_config = None
        if workspace_id and agent_id:
            agent_config = workspace_manager.get_agent(workspace_id, agent_id)
        
        # Initialize tools for organization/workspace
        if workspace_id:
            await tool_manager.initialize_tools_for_organization(org_id, workspace_id)
        else:
            await tool_manager.initialize_tools_for_organization(org_id)
        
        # Get available tools
        available_tools = tool_manager.get_available_tools(org_id, workspace_id)
        
        # Enhanced routing logic - consider agent configuration
        selected_tool = None
        
        if agent_config:
            # Use agent-specific tool selection logic
            enabled_tool_names = [tool.value for tool in agent_config.enabled_tools]
            
            # Simple routing based on query content and agent's enabled tools
            needs_rag = any(keyword in user_query.lower() for keyword in [
                "search", "find", "information", "document", "knowledge", 
                "tell me about", "what is", "explain", "help"
            ])
            
            if needs_rag and "rag" in enabled_tool_names and "rag" in available_tools:
                selected_tool = "rag"
            # Add more sophisticated routing logic here for other tools
        else:
            # Default routing logic
            needs_rag = any(keyword in user_query.lower() for keyword in [
                "search", "find", "information", "document", "knowledge", 
                "tell me about", "what is", "explain", "help"
            ])
            
            if needs_rag and "rag" in available_tools:
                selected_tool = "rag"
        
        return {
            "organization_id": org_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_tool": selected_tool,
            "user_query": user_query,
            "context": {
                "available_tools": available_tools,
                "routing_decision": selected_tool,
                "agent_config": agent_config.__dict__ if agent_config else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in route_query: {str(e)}")
        return {
            "organization_id": org_id or "unknown",
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_tool": None,
            "user_query": user_query,
            "context": {"error": str(e)}
        }

async def execute_rag_tool(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the RAG tool with workspace context."""
    try:
        org_id = state["organization_id"]
        workspace_id = state.get("workspace_id")
        user_query = state["user_query"]
        
        # Execute RAG tool with workspace context
        execution_result = await tool_manager.execute_tool(
            organization_id=org_id,
            tool_name="rag",
            workspace_id=workspace_id,
            query=user_query
        )
        
        # Format results for the LLM
        if execution_result.success and execution_result.result.data:
            rag_results = execution_result.result.data["results"]
            context_text = "\n\n".join([
                f"[Score: {result['score']:.3f}] {result['content']}" 
                for result in rag_results[:3]  # Top 3 results
            ])
            
            tool_message = f"Retrieved relevant information:\n{context_text}"
        else:
            tool_message = f"RAG search failed: {execution_result.result.error or 'Unknown error'}"
        
        # Add tool result to messages
        messages = [AIMessage(content=tool_message)]
        
        return {
            "messages": messages,
            "tool_results": [execution_result.result.__dict__]
        }
        
    except Exception as e:
        logger.error(f"Error executing RAG tool: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Error executing RAG tool: {str(e)}")],
            "tool_results": []
        }

async def generate_response(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the final response using the LLM with agent-specific configuration."""
    try:
        configuration = config.get("configurable", {})
        org_id = state["organization_id"]
        workspace_id = state.get("workspace_id")
        agent_id = state.get("agent_id")
        
        # Get agent configuration
        agent_config = None
        if workspace_id and agent_id:
            agent_config = workspace_manager.get_agent(workspace_id, agent_id)
        
        # Get LLM configuration (workspace-aware)
        llm_config = config_manager.get_llm_config(org_id, workspace_id)
        
        # Apply agent-specific model settings if available
        model_settings = {}
        if agent_config and agent_config.model_settings:
            model_settings = agent_config.model_settings
        
        # Initialize LLM with configuration
        llm = ChatOpenAI(
            model=model_settings.get("model", llm_config.model),
            temperature=model_settings.get("temperature", llm_config.temperature),
            max_tokens=model_settings.get("max_tokens", llm_config.max_tokens),
            api_key=llm_config.api_key
        )
        
        # Build system prompt
        system_prompt = f"""You are an AI assistant for organization {org_id}."""
        
        if workspace_id:
            system_prompt += f"\nWorkspace: {workspace_id}"
        
        if agent_config:
            system_prompt += f"\nAgent: {agent_config.name} - {agent_config.description}"
            
            # Add custom system prompt if specified
            if agent_config.system_prompt:
                system_prompt += f"\n\n{agent_config.system_prompt}"
            
            # Add custom instructions
            if agent_config.custom_instructions:
                system_prompt += f"\n\nCustom Instructions: {agent_config.custom_instructions}"
        
        # Add general instructions
        system_prompt += f"""

You have access to various tools and a knowledge base through RAG (Retrieval-Augmented Generation).

When responding:
1. Use the retrieved information to provide accurate, contextual answers
2. If no relevant information was found, be honest about it
3. Be helpful, concise, and professional
4. Cite information when appropriate

Available tools: {tool_manager.get_available_tools(org_id, workspace_id)}
"""
        
        # Add custom instructions from configuration if available
        custom_instructions = configuration.get("custom_instructions", "")
        if custom_instructions:
            system_prompt += f"\n\nAdditional instructions: {custom_instructions}"
        
        # Prepare messages for LLM
        messages_for_llm = [SystemMessage(content=system_prompt)]
        messages_for_llm.extend(state["messages"])
        
        # Generate response
        response = await llm.ainvoke(messages_for_llm)
        
        return {
            "messages": [response]
        }
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        error_message = AIMessage(content=f"I apologize, but I encountered an error while processing your request: {str(e)}")
        return {
            "messages": [error_message]
        }

def should_use_rag(state: AgentState) -> str:
    """Determine if we should use RAG tool."""
    current_tool = state.get("current_tool")
    if current_tool == "rag":
        return "rag_tool"
    else:
        return "generate_response"

# Create the enhanced graph
workflow = StateGraph(AgentState, config_schema=WorkspaceAgentConfiguration)

# Add nodes
workflow.add_node("route_query", route_query)
workflow.add_node("rag_tool", execute_rag_tool)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("__start__", "route_query")
workflow.add_conditional_edges(
    "route_query",
    should_use_rag,
    {
        "rag_tool": "rag_tool",
        "generate_response": "generate_response"
    }
)
workflow.add_edge("rag_tool", "generate_response")
workflow.add_edge("generate_response", "__end__")

# Compile the enhanced graph
enhanced_graph = workflow.compile(name="Multi-Workspace AI Agent")

# Convenience functions for the enhanced system

async def create_workspace_agent(
    organization_id: str,
    workspace_id: str,
    agent_id: str,
    config_overrides: Optional[Dict] = None
) -> Any:
    """Create and configure an agent for a specific workspace."""
    # Initialize tools for the workspace
    success = await tool_manager.initialize_tools_for_organization(organization_id, workspace_id)
    if not success:
        logger.warning(f"Some tools failed to initialize for workspace {workspace_id}")
    
    # Return configured graph with workspace context
    return enhanced_graph

async def execute_agent_query(
    organization_id: str,
    query: str,
    workspace_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    config_overrides: Optional[Dict] = None
) -> Dict[str, Any]:
    """Execute a query using the enhanced agent system."""
    try:
        # Prepare configuration
        config = {
            "configurable": {
                "organization_id": organization_id,
                "workspace_id": workspace_id,
                "agent_id": agent_id
            }
        }
        
        if config_overrides:
            config["configurable"].update(config_overrides)
        
        # Prepare initial state
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_tool": None,
            "tool_results": [],
            "user_query": query,
            "context": {}
        }
        
        # Execute the graph
        result = await enhanced_graph.ainvoke(initial_state, config)
        
        # Extract the final response
        final_message = None
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                final_message = message.content
                break
        
        return {
            "success": True,
            "response": final_message or "No response generated",
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "tool_results": result.get("tool_results", [])
        }
        
    except Exception as e:
        logger.error(f"Error executing agent query: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id
        }

# API Helper Functions

async def ingest_document_for_organization(
    organization_id: str, 
    content: str, 
    metadata: Optional[Dict] = None,
    document_id: Optional[str] = None,
    workspace_id: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest a document into an organization's or workspace's knowledge base."""
    # Initialize tools if needed
    await tool_manager.initialize_tools_for_organization(organization_id, workspace_id)
    
    # Get RAG tool instance
    rag_tool = tool_manager.get_tool_instance(organization_id, "rag", workspace_id)
    if not rag_tool:
        return {"success": False, "error": "RAG tool not available"}
    
    # Ingest document
    result = await rag_tool.ingest_document(content, metadata, document_id)
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error
    }

async def get_organization_knowledge_base_info(
    organization_id: str,
    workspace_id: Optional[str] = None
) -> Dict[str, Any]:
    """Get information about an organization's or workspace's knowledge base."""
    # Initialize tools if needed
    await tool_manager.initialize_tools_for_organization(organization_id, workspace_id)
      # Get RAG tool instance
    rag_tool = tool_manager.get_tool_instance(organization_id, "rag", workspace_id)
    if not rag_tool:
        return {"success": False, "error": "RAG tool not available"}
    
    # Get collection info
    result = await rag_tool.get_collection_info()
    return {
        "success": result.success,
        "data": result.data,
        "error": result.error
    }
