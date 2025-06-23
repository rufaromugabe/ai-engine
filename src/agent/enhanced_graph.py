"""Enhanced agent graph system with intelligent routing and self-correction."""

from __future__ import annotations

import logging
import json
import time
import os
from typing import Any, Dict, List, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, add_messages

# Handle both relative and absolute imports
try:
    from .config import config_manager
    from .tool_manager import tool_manager
    from .workspace_manager import workspace_manager, AgentConfig
except ImportError:
    # Fallback for when relative imports don't work
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    
    from config import config_manager
    from tool_manager import tool_manager
    from workspace_manager import workspace_manager, AgentConfig

logger = logging.getLogger(__name__)

# Enhanced routing models for LLM-based decision making
class ToolCall(BaseModel):
    """Represents a tool call decision."""
    name: str = Field(description="The name of the tool to call")
    args: dict = Field(description="The arguments to pass to the tool")
    reasoning: str = Field(description="Why this tool was selected")

class RouterDecision(BaseModel):
    """Router decision with tool calls and strategy."""
    tool_calls: List[ToolCall] = Field(description="List of tools to call in sequence")
    strategy: Literal["single", "sequential", "parallel"] = Field(
        description="How to execute the tools: single (one tool), sequential (one after another), parallel (simultaneously)"
    )
    confidence: float = Field(description="Confidence in this routing decision (0-1)")

class ReflectionResult(BaseModel):
    """Result of reflection on tool execution."""
    quality_score: float = Field(description="Quality of the tool results (0-1)")
    should_retry: bool = Field(description="Whether to retry with different approach")
    retry_strategy: Optional[str] = Field(description="How to retry if needed")
    feedback: str = Field(description="Feedback on the results")

class AgentState(TypedDict):
    """Enhanced state for workspace-aware AI agent with reflection."""
    messages: Annotated[List[BaseMessage], add_messages]
    organization_id: str
    workspace_id: Optional[str]
    agent_id: Optional[str]
    current_tool_calls: List[ToolCall]
    tool_results: List[Dict[str, Any]]
    user_query: str
    context: Dict[str, Any]
    iteration_count: int
    reflection_history: List[ReflectionResult]
    routing_history: List[RouterDecision]

@dataclass
class WorkspaceAgentConfiguration:
    """Configuration for workspace agent execution."""
    organization_id: str
    workspace_id: Optional[str] = None
    agent_id: Optional[str] = None
    custom_instructions: str = ""
    
async def intelligent_router(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """LLM-based intelligent router that decides which tools to use."""
    # Initialize variables to prevent UnboundLocalError
    org_id = None
    workspace_id = None
    agent_id = None
    user_query = ""
    
    try:
        # Get configuration with safe fallbacks
        configuration = config.get("configurable", {}) if config else {}
        
        # Use environment variables as fallbacks if config not provided
        org_id = configuration.get("organization_id") or os.getenv("DEFAULT_ORG_ID", "dev-org")
        workspace_id = configuration.get("workspace_id") or os.getenv("DEFAULT_WORKSPACE_ID", "dev-workspace")
        agent_id = configuration.get("agent_id") or os.getenv("DEFAULT_AGENT_ID", "dev-agent")
        
        if not org_id:
            raise ValueError("organization_id is required")
        
        # Ensure the development organization exists
        if org_id in ["dev-org", os.getenv("DEFAULT_ORG_ID", "dev-org")]:
            await ensure_dev_organization_exists()
        
        logger.info(f"Router using org_id: {org_id}, workspace_id: {workspace_id}, agent_id: {agent_id}")
        
        # Get the latest user message
        user_query = ""
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                user_query = message.content
                break
        
        # Initialize tools for organization/workspace
        if workspace_id:
            await tool_manager.initialize_tools_for_organization(org_id, workspace_id)
        else:
            await tool_manager.initialize_tools_for_organization(org_id)
        
        # Get available tools and their schemas
        available_tools = tool_manager.get_available_tools(org_id, workspace_id)
        tool_schemas = tool_manager.get_tool_schemas(org_id, workspace_id)
        
        # Get agent configuration if specified
        agent_config = None
        if workspace_id and agent_id:
            agent_config = workspace_manager.get_agent(workspace_id, agent_id)
        
        # Get LLM configuration for routing
        llm_config = config_manager.get_llm_config(org_id, workspace_id)
        llm = ChatOpenAI(
            model=llm_config.model,
            temperature=0.1,  # Low temperature for consistent routing decisions
            api_key=llm_config.api_key
        )
        structured_llm = llm.with_structured_output(RouterDecision)
        
        # Build context for routing decision
        routing_context = ""
        if state.get("reflection_history"):
            recent_reflections = state["reflection_history"][-3:]  # Last 3 reflections
            routing_context += "\n\nPrevious attempt feedback:\n"
            for i, reflection in enumerate(recent_reflections):
                routing_context += f"{i+1}. Quality: {reflection.quality_score:.2f}, Feedback: {reflection.feedback}\n"
        
        if state.get("routing_history"):
            routing_context += f"\n\nPrevious routing attempts: {len(state['routing_history'])}"
          # Create system prompt for intelligent routing
        system_prompt = f"""You are an intelligent tool router for an AI assistant. Analyze the user's query and decide whether tools are needed and which ones to use.

Available tools and their capabilities:
{json.dumps(tool_schemas, indent=2)}

User Query: {user_query}

Current iteration: {state.get('iteration_count', 0)}
{routing_context}

Instructions:
1. First, determine if the query requires any tools at all
2. For simple greetings, conversational responses, or basic questions that don't need external data, return an empty tool_calls list
3. Only use tools when the query specifically requires information retrieval, analysis, or external processing
4. If tools are needed, select the most appropriate tool(s) to fulfill the request
5. Consider the previous feedback if this is a retry
6. Provide clear reasoning for your tool selection (or lack thereof)
7. Set confidence based on how well your decision matches the query intent

Tool usage guidelines:
- Simple greetings ("hi", "hello", "how are you") → NO TOOLS (empty tool_calls list)
- Basic conversational questions → NO TOOLS (empty tool_calls list)
- General questions that can be answered with common knowledge → NO TOOLS (empty tool_calls list)
- Knowledge/information requests about specific topics → RAG tool
- Document search or retrieval needs → RAG tool
- Multiple related information questions → Sequential tool calls
- Complex analysis requiring external data → Consider multiple tools

IMPORTANT: Return empty tool_calls list for conversational queries that don't need external information.
Be strategic: if previous attempts failed, try a different approach or tool combination."""

        # Get routing decision from LLM
        router_result = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt)
        ])
          # Filter tool calls to only include available tools
        valid_tool_calls = []
        for tool_call in router_result.tool_calls:
            if tool_call.name in available_tools:
                valid_tool_calls.append(tool_call)
            else:
                logger.warning(f"Router selected unavailable tool: {tool_call.name}")
        
        # Only apply default RAG if the query seems to require information retrieval
        # and no valid tools were selected (but the router confidence is low)
        if not valid_tool_calls and router_result.confidence < 0.3:
            # Check if this might be an information-seeking query by looking for keywords
            info_keywords = [
                "what", "how", "when", "where", "why", "who", "tell me", "explain", 
                "describe", "define", "information", "details", "about", "help with",
                "find", "search", "lookup", "know", "learn"
            ]
            query_lower = user_query.lower()
            
            # If the query contains information-seeking keywords, default to RAG
            if any(keyword in query_lower for keyword in info_keywords) and "rag" in available_tools:
                valid_tool_calls = [ToolCall(
                    name="rag",
                    args={"query": user_query},
                    reasoning="Query appears to seek information but router confidence was low - defaulting to RAG"
                )]
                logger.info(f"Applied RAG fallback for information-seeking query: {user_query}")
        
        # Log the routing decision for debugging
        logger.info(f"Router decision for '{user_query}': {len(valid_tool_calls)} tools selected, confidence: {router_result.confidence}")
        if valid_tool_calls:
            tool_names = [tc.name for tc in valid_tool_calls]
            logger.info(f"Selected tools: {tool_names}")
        else:
            logger.info("No tools selected - will generate direct response")
          # Update routing history
        routing_history = state.get("routing_history", [])
        routing_history.append(RouterDecision(
            tool_calls=valid_tool_calls,
            strategy=router_result.strategy,
            confidence=router_result.confidence
        ))
        
        return {
            "organization_id": org_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_tool_calls": valid_tool_calls,
            "user_query": user_query,
            "routing_history": routing_history,
            "context": {
                "available_tools": available_tools,
                "routing_decision": router_result.dict(),
                "agent_config": agent_config.__dict__ if agent_config else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in intelligent_router: {str(e)}")
        return {
            "organization_id": org_id or "unknown",
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_tool_calls": [],
            "user_query": user_query,
            "context": {"error": str(e)}
        }

async def execute_tools(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Execute the selected tools based on the routing decision."""
    try:
        org_id = state["organization_id"]
        workspace_id = state.get("workspace_id")
        user_query = state["user_query"]
        tool_calls = state.get("current_tool_calls", [])
        
        if not tool_calls:
            return {
                "messages": [AIMessage(content="No tools were selected for execution.")],
                "tool_results": []
            }
        
        messages = []
        tool_results = []
        
        # Execute tools based on strategy (for now, execute sequentially)
        for tool_call in tool_calls:
            start_time = time.time()
            
            try:
                # Execute the tool
                execution_result = await tool_manager.execute_tool(
                    organization_id=org_id,
                    tool_name=tool_call.name,
                    workspace_id=workspace_id,
                    **tool_call.args
                )
                
                execution_time = time.time() - start_time
                
                # Format results for the LLM
                if execution_result.success and execution_result.result.data:
                    if tool_call.name == "rag":
                        rag_results = execution_result.result.data.get("results", [])
                        if rag_results:
                            context_text = "\n\n".join([
                                f"[Relevance: {result.get('score', 0):.3f}] {result.get('content', '')}" 
                                for result in rag_results[:5]  # Top 5 results
                            ])
                            tool_message = f"Retrieved relevant information using {tool_call.reasoning}:\n{context_text}"
                        else:
                            tool_message = f"No relevant information found for the query: '{user_query}'"
                    else:
                        # Generic tool result formatting
                        tool_message = f"Tool {tool_call.name} executed successfully: {execution_result.result.data}"
                else:
                    tool_message = f"Tool {tool_call.name} execution failed: {execution_result.result.error or 'Unknown error'}"
                
                messages.append(AIMessage(content=tool_message))
                tool_results.append({
                    "tool_name": tool_call.name,
                    "success": execution_result.success,
                    "execution_time": execution_time,
                    "reasoning": tool_call.reasoning,
                    "result": execution_result.result.__dict__
                })
                
            except Exception as e:
                logger.error(f"Error executing tool {tool_call.name}: {str(e)}")
                messages.append(AIMessage(content=f"Error executing tool {tool_call.name}: {str(e)}"))
                tool_results.append({
                    "tool_name": tool_call.name,
                    "success": False,
                    "error": str(e),
                    "reasoning": tool_call.reasoning
                })
        
        return {
            "messages": messages,
            "tool_results": tool_results
        }
        
    except Exception as e:
        logger.error(f"Error executing tools: {str(e)}")
        return {
            "messages": [AIMessage(content=f"Error executing tools: {str(e)}")],
            "tool_results": []
        }

async def reflection_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Reflect on tool execution results and decide if retry is needed."""
    try:
        # Get configuration with safe fallbacks
        configuration = config.get("configurable", {}) if config else {}
        org_id = state["organization_id"] or os.getenv("DEFAULT_ORG_ID", "dev-org")
        workspace_id = state.get("workspace_id") or os.getenv("DEFAULT_WORKSPACE_ID", "dev-workspace")
        tool_results = state.get("tool_results", [])
        user_query = state["user_query"]
        iteration_count = state.get("iteration_count", 0)
        
        # Get LLM for reflection
        llm_config = config_manager.get_llm_config(org_id, workspace_id)
        llm = ChatOpenAI(
            model=llm_config.model,
            temperature=0.2,
            api_key=llm_config.api_key
        )
        structured_llm = llm.with_structured_output(ReflectionResult)
        
        # Analyze tool results
        results_summary = ""
        successful_tools = 0
        total_tools = len(tool_results)
        
        for result in tool_results:
            if result.get("success"):
                successful_tools += 1
                results_summary += f"✓ {result.get('tool_name')}: Success\n"
            else:
                results_summary += f"✗ {result.get('tool_name')}: Failed - {result.get('error', 'Unknown error')}\n"
        
        # Create reflection prompt
        reflection_prompt = f"""Analyze the quality and relevance of these tool execution results for the user's query.

User Query: {user_query}
Iteration: {iteration_count + 1}
Successful Tools: {successful_tools}/{total_tools}

Tool Results Summary:
{results_summary}

Detailed Results:
{json.dumps(tool_results, indent=2)}

Evaluate:
1. Quality of results (0-1 score)
2. Whether the results adequately address the user's query
3. If a retry with different approach would be beneficial
4. Specific feedback for improvement

Consider:
- Did we get relevant information?
- Are there gaps in the response?
- Would different tools or search terms help?
- Have we tried this approach before?

Max iterations allowed: 3 (current: {iteration_count + 1})"""

        # Get reflection decision
        reflection_result = await structured_llm.ainvoke([
            SystemMessage(content=reflection_prompt)
        ])
        
        # Update reflection history
        reflection_history = state.get("reflection_history", [])
        reflection_history.append(reflection_result)
        
        return {
            "reflection_history": reflection_history,
            "iteration_count": iteration_count + 1
        }
        
    except Exception as e:
        logger.error(f"Error in reflection_node: {str(e)}")
        # Default to not retry on reflection error
        return {
            "reflection_history": state.get("reflection_history", []) + [
                ReflectionResult(
                    quality_score=0.5,
                    should_retry=False,
                    feedback=f"Reflection error: {str(e)}"
                )            ],
            "iteration_count": state.get("iteration_count", 0) + 1
        }

async def generate_response(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate the final response using the LLM with agent-specific configuration."""
    try:
        # Get configuration with safe fallbacks
        configuration = config.get("configurable", {}) if config else {}
        org_id = state["organization_id"] or os.getenv("DEFAULT_ORG_ID", "dev-org")
        workspace_id = state.get("workspace_id") or os.getenv("DEFAULT_WORKSPACE_ID", "dev-workspace")
        agent_id = state.get("agent_id") or os.getenv("DEFAULT_AGENT_ID", "dev-agent")
        
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
        
        # Build enhanced system prompt with reflection context
        system_prompt = f"""You are an intelligent AI assistant for organization {org_id}."""
        
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
        # Add context from tool execution and reflection
        tool_results = state.get("tool_results", [])
        reflection_history = state.get("reflection_history", [])
        iteration_count = state.get("iteration_count", 0)
        
        system_prompt += f"""

You are an advanced AI assistant with access to various tools and knowledge bases.

Current Context:
- Iteration: {iteration_count}
- Tools executed: {len(tool_results)}
- Available tools: {tool_manager.get_available_tools(org_id, workspace_id)}

When responding:
1. If no tools were used, provide a natural conversational response using your built-in knowledge
2. For simple greetings and casual conversation, respond warmly and naturally without tool information
3. If tools were used, synthesize information from all tool results
4. Be transparent about the quality and limitations of available information when tools were used
5. If multiple attempts were made, acknowledge the iteration process
6. Provide helpful, accurate, and contextual responses
7. Cite sources when appropriate and tools were used

Response Guidelines:
- Simple greetings ("hi", "hello") → Provide a friendly, natural greeting
- Casual conversation → Respond conversationally without mentioning tools
- Information queries with tool results → Synthesize and cite tool information
- Information queries without tool results → Use general knowledge and explain limitations
- If tool results are limited, explain why and suggest alternatives
- If this is a retry iteration, build upon previous attempts
- Be honest about gaps in knowledge or failed tool executions
- Prioritize user satisfaction while maintaining accuracy
"""

        # Add reflection context if available
        if reflection_history:
            latest_reflection = reflection_history[-1]
            system_prompt += f"""

Latest Reflection Analysis:
- Quality Score: {latest_reflection.quality_score:.2f}
- Feedback: {latest_reflection.feedback}

Use this reflection to improve your response quality.
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

def should_execute_tools(state: AgentState) -> str:
    """Determine if we should execute tools."""
    current_tool_calls = state.get("current_tool_calls", [])
    if current_tool_calls:
        return "execute_tools"
    else:
        return "generate_response"

def should_retry_or_respond(state: AgentState) -> str:
    """Decide whether to retry with different approach or generate final response."""
    reflection_history = state.get("reflection_history", [])
    iteration_count = state.get("iteration_count", 0)
    
    # Check if we should retry
    if reflection_history:
        latest_reflection = reflection_history[-1]
        
        # Retry conditions:
        # 1. Latest reflection suggests retry
        # 2. We haven't exceeded max iterations (3)
        # 3. Quality score is below threshold (0.6)
        if (latest_reflection.should_retry and 
            iteration_count < 3 and 
            latest_reflection.quality_score < 0.6):
            return "intelligent_router"  # Go back to routing for different approach
    
    # Otherwise, generate final response
    return "generate_response"

# Create the enhanced graph with intelligent routing and self-correction
workflow = StateGraph(AgentState, config_schema=WorkspaceAgentConfiguration)

# Add nodes
workflow.add_node("intelligent_router", intelligent_router)
workflow.add_node("execute_tools", execute_tools)
workflow.add_node("reflection_node", reflection_node)
workflow.add_node("generate_response", generate_response)

# Add edges
workflow.add_edge("__start__", "intelligent_router")

# Conditional edges from router
workflow.add_conditional_edges(
    "intelligent_router",
    should_execute_tools,
    {
        "execute_tools": "execute_tools",
        "generate_response": "generate_response"
    }
)

# After tool execution, always reflect
workflow.add_edge("execute_tools", "reflection_node")

# Conditional edges from reflection - retry or respond
workflow.add_conditional_edges(
    "reflection_node",
    should_retry_or_respond,
    {
        "intelligent_router": "intelligent_router",  # Retry with different approach
        "generate_response": "generate_response"     # Generate final response
    }
)

# End after generating response
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
          # Prepare initial state with enhanced structure
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "organization_id": organization_id,
            "workspace_id": workspace_id,
            "agent_id": agent_id,
            "current_tool_calls": [],
            "tool_results": [],
            "user_query": query,
            "context": {},
            "iteration_count": 0,
            "reflection_history": [],
            "routing_history": []
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

async def ensure_dev_organization_exists():
    """Ensure the development organization exists for LangGraph Studio."""
    dev_org_id = os.getenv("DEFAULT_ORG_ID", "dev-org")
    
    try:
        # Try to get the organization config
        config_manager.get_organization_config(dev_org_id)
        logger.info(f"Development organization '{dev_org_id}' already exists")
    except ValueError:
        # Organization doesn't exist, create it
        logger.info(f"Creating development organization '{dev_org_id}'")
        
        # Handle both relative and absolute imports
        try:
            from .config import OrganizationConfig, ToolType
        except ImportError:
            from config import OrganizationConfig, ToolType
        
        org_config = OrganizationConfig(
            organization_id=dev_org_id,
            enabled_tools=[ToolType.RAG]  # Enable RAG tool by default
        )
        
        # Register the organization
        config_manager.register_organization(org_config)
        
        # Initialize tools for the organization
        await tool_manager.initialize_tools_for_organization(dev_org_id)
        
        logger.info(f"Development organization '{dev_org_id}' created successfully")
    
    return dev_org_id
