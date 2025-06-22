"""FastAPI server for the RAG-enabled AI agent system."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import asyncio
import logging
from contextlib import asynccontextmanager

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.enhanced_graph import enhanced_graph, execute_agent_query, ingest_document_for_organization, get_organization_knowledge_base_info
from agent.config import config_manager, OrganizationConfig, ToolType, WorkspaceSpecificConfig
from agent.tool_manager import tool_manager
from agent.workspace_manager import workspace_manager, AgentConfig, WorkspaceConfig
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class QueryRequest(BaseModel):
    query: str
    organization_id: str
    workspace_id: Optional[str] = None
    agent_id: Optional[str] = None

class QueryResponse(BaseModel):
    success: bool
    response: str
    organization_id: str
    workspace_id: Optional[str] = None
    agent_id: Optional[str] = None
    tool_results: List[Dict[str, Any]] = []
    error: Optional[str] = None

class DocumentRequest(BaseModel):
    content: str
    organization_id: str
    workspace_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    document_id: Optional[str] = None

class DocumentResponse(BaseModel):
    success: bool
    document_id: Optional[str] = None
    chunks_created: Optional[int] = None
    error: Optional[str] = None

# Workspace management models
class CreateWorkspaceRequest(BaseModel):
    organization_id: str
    name: str
    description: str = ""
    shared_tools: Optional[List[str]] = None
    shared_settings: Optional[Dict[str, Any]] = None

class WorkspaceResponse(BaseModel):
    success: bool
    workspace_id: Optional[str] = None
    error: Optional[str] = None

class CreateAgentRequest(BaseModel):
    workspace_id: str
    name: str
    description: str = ""
    enabled_tools: Optional[List[str]] = None
    custom_instructions: str = ""
    system_prompt: str = ""
    model_settings: Optional[Dict[str, Any]] = None
    tool_configurations: Optional[Dict[str, Dict[str, Any]]] = None

class AgentResponse(BaseModel):
    success: bool
    agent_id: Optional[str] = None
    error: Optional[str] = None

class UpdateAgentRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    enabled_tools: Optional[List[str]] = None
    custom_instructions: Optional[str] = None
    system_prompt: Optional[str] = None
    model_settings: Optional[Dict[str, Any]] = None
    tool_configurations: Optional[Dict[str, Dict[str, Any]]] = None
    is_active: Optional[bool] = None

class OrganizationSetupRequest(BaseModel):
    organization_id: str
    enabled_tools: List[str] = ["rag"]
    rag_settings: Optional[Dict[str, Any]] = None

class KnowledgeBaseInfo(BaseModel):
    success: bool
    collection_name: Optional[str] = None
    total_points: Optional[int] = None
    organization_documents: Optional[int] = None
    error: Optional[str] = None

# New Pydantic models for multitenant features
class GroupedSearchRequest(BaseModel):
    query: str
    organization_id: str
    workspace_id: Optional[str] = None
    group_by: str
    limit: int = 5
    group_size: int = 5
    filter_conditions: Optional[Dict[str, Any]] = None

class GroupedSearchResponse(BaseModel):
    success: bool
    query: str
    groups: Dict[str, List[Dict[str, Any]]]
    total_groups: int
    group_by: str
    organization_id: str
    workspace_id: Optional[str] = None
    error: Optional[str] = None

class TenantStatsRequest(BaseModel):
    organization_id: str
    workspace_id: Optional[str] = None

class TenantStatsResponse(BaseModel):
    success: bool
    organization_id: str
    workspace_id: str
    total_documents: int
    total_chunks: int
    collection_name: str
    avg_chunks_per_document: float
    error: Optional[str] = None

class BulkDeleteRequest(BaseModel):
    organization_id: str
    workspace_id: Optional[str] = None
    confirm_organization_id: str

class BulkDeleteResponse(BaseModel):
    success: bool
    organization_id: str
    workspace_id: str
    deleted_documents: int
    deleted_chunks: int
    collection_name: str
    error: Optional[str] = None

class MigrationRequest(BaseModel):
    organization_id: str
    source_workspace_id: Optional[str] = None
    target_workspace_id: str
    document_filter: Optional[Dict[str, Any]] = None

class MigrationResponse(BaseModel):
    success: bool
    source_organization_id: str
    source_workspace_id: str
    target_workspace_id: str
    migrated_chunks: int
    collection_name: str
    error: Optional[str] = None

# Global state to track initialized organizations
initialized_orgs = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Multi-Workspace AI Agent API")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Workspace AI Agent API")
    
    # Cleanup all organization tools
    for org_id in initialized_orgs:
        await tool_manager.cleanup_organization_tools(org_id)

app = FastAPI(
    title="Multi-Workspace AI Agent API",
    description="Multi-tenant AI agent system with workspace isolation using LangGraph, Qdrant, and OpenAI",
    version="2.0.0",
    lifespan=lifespan
)

@app.post("/api/v1/setup-organization", response_model=Dict[str, Any])
async def setup_organization(request: OrganizationSetupRequest):
    """Set up a new organization with custom configuration."""
    try:
        org_id = request.organization_id
        
        # Convert string tool names to ToolType enum
        enabled_tools = []
        for tool_name in request.enabled_tools:
            try:
                enabled_tools.append(ToolType(tool_name))
            except ValueError:
                logger.warning(f"Unknown tool type: {tool_name}")
        
        # Create organization configuration
        org_config = OrganizationConfig(
            organization_id=org_id,
            enabled_tools=enabled_tools
        )
        
        # Apply RAG settings if provided
        if request.rag_settings:
            for key, value in request.rag_settings.items():
                if hasattr(org_config.rag_config, key):
                    setattr(org_config.rag_config, key, value)
        
        # Register and initialize
        config_manager.register_organization(org_config)
        success = await tool_manager.initialize_tools_for_organization(org_id)
        
        if success:
            initialized_orgs.add(org_id)
            return {
                "success": True,
                "organization_id": org_id,
                "enabled_tools": request.enabled_tools,
                "message": f"Organization '{org_id}' set up successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to initialize tools for organization '{org_id}'")
            
    except Exception as e:
        logger.error(f"Error setting up organization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """Query the AI agent with enhanced workspace support."""
    try:
        # Use the enhanced execution system
        result = await execute_agent_query(
            organization_id=request.organization_id,
            query=request.query,
            workspace_id=request.workspace_id,
            agent_id=request.agent_id
        )
        
        return QueryResponse(
            success=result["success"],
            response=result.get("response", ""),
            organization_id=result["organization_id"],
            workspace_id=result.get("workspace_id"),
            agent_id=result.get("agent_id"),            tool_results=result.get("tool_results", []),
            error=result.get("error")
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            success=False,
            response="",
            organization_id=request.organization_id,
            workspace_id=request.workspace_id,
            agent_id=request.agent_id,
            error=str(e)
        )

@app.post("/api/v1/ingest", response_model=DocumentResponse)
async def ingest_document(request: DocumentRequest):
    """Ingest a document into the organization's or workspace's knowledge base."""
    try:
        org_id = request.organization_id
        workspace_id = request.workspace_id
        
        # Ensure organization is initialized
        if org_id not in initialized_orgs:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found. Please set it up first.")
        
        # Ingest the document with workspace context
        result = await ingest_document_for_organization(
            organization_id=org_id,
            content=request.content,
            metadata=request.metadata,
            document_id=request.document_id,
            workspace_id=workspace_id
        )
        
        if result["success"]:
            return DocumentResponse(
                success=True,
                document_id=result["data"]["document_id"],
                chunks_created=result["data"]["chunks_created"]
            )
        else:
            return DocumentResponse(
                success=False,
                error=result["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting document: {str(e)}")
        return DocumentResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/v1/knowledge-base/{organization_id}", response_model=KnowledgeBaseInfo)
async def get_knowledge_base_info(organization_id: str, workspace_id: Optional[str] = None):
    """Get information about an organization's or workspace's knowledge base."""
    try:
        # Ensure organization is initialized
        if organization_id not in initialized_orgs:
            raise HTTPException(status_code=404, detail=f"Organization '{organization_id}' not found.")
        
        # Get knowledge base info with workspace context
        info = await get_organization_knowledge_base_info(organization_id, workspace_id)
        
        if info["success"]:
            data = info["data"]
            return KnowledgeBaseInfo(
                success=True,
                collection_name=data.get("collection_name"),
                total_points=data.get("total_points"),
                organization_documents=data.get("organization_documents")
            )
        else:
            return KnowledgeBaseInfo(
                success=False,                error=info["error"]
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge base info: {str(e)}")
        return KnowledgeBaseInfo(
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "service": "multitenant-rag-api"}

@app.get("/api/v1/organizations")
async def list_organizations():
    """List all initialized organizations."""
    return {
        "organizations": list(initialized_orgs),
        "total": len(initialized_orgs)
    }

# Add CORS middleware for development
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace Management Endpoints

@app.post("/api/v1/workspaces", response_model=WorkspaceResponse)
async def create_workspace(request: CreateWorkspaceRequest):
    """Create a new workspace for an organization."""
    try:
        # Convert tool names to ToolType enum if provided
        shared_tools = None
        if request.shared_tools:
            shared_tools = []
            for tool_name in request.shared_tools:
                try:
                    shared_tools.append(ToolType(tool_name))
                except ValueError:
                    logger.warning(f"Unknown tool type: {tool_name}")
        
        # Create workspace
        workspace = workspace_manager.create_workspace(
            organization_id=request.organization_id,
            name=request.name,
            description=request.description,
            shared_tools=shared_tools,
            shared_settings=request.shared_settings
        )
        
        # Initialize workspace tools
        await workspace_manager.initialize_workspace_tools(workspace.workspace_id)

        return WorkspaceResponse(
            success=True,
            workspace_id=workspace.workspace_id
         )
        
    except Exception as e:
        logger.error(f"Error creating workspace: {str(e)}")
        return WorkspaceResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/v1/organizations/{organization_id}/workspaces")
async def list_organization_workspaces(organization_id: str):
    """List all workspaces for an organization."""
    try:
        workspaces = workspace_manager.list_organization_workspaces(organization_id)
        return {
            "success": True,
            "workspaces": [
                {
                    "workspace_id": ws.workspace_id,
                    "name": ws.name,
                    "description": ws.description,
                    "status": ws.status.value,
                    "agent_count": len(ws.agents),
                    "created_at": ws.created_at.isoformat()
                }
                for ws in workspaces
            ]
        }
    except Exception as e:
        logger.error(f"Error listing workspaces: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/workspaces/{workspace_id}")
async def get_workspace(workspace_id: str):
    """Get workspace details."""
    try:
        workspace = workspace_manager.get_workspace(workspace_id)
        if not workspace:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return {
            "success": True,
            "workspace": {
                "workspace_id": workspace.workspace_id,
                "organization_id": workspace.organization_id,
                "name": workspace.name,
                "description": workspace.description,
                "status": workspace.status.value,
                "shared_tools": [tool.value for tool in workspace.shared_tools],
                "agents": [
                    {
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "description": agent.description,
                        "is_active": agent.is_active,
                        "enabled_tools": [tool.value for tool in agent.enabled_tools]
                    }
                    for agent in workspace.agents.values()
                ],
                "created_at": workspace.created_at.isoformat(),
                "updated_at": workspace.updated_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace: {str(e)}")
        return {"success": False, "error": str(e)}

@app.delete("/api/v1/workspaces/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete a workspace."""
    try:
        success = workspace_manager.delete_workspace(workspace_id)
        if success:
            return {"success": True, "message": "Workspace deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Workspace not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting workspace: {str(e)}")
        return {"success": False, "error": str(e)}

# Agent Management Endpoints

@app.post("/api/v1/workspaces/{workspace_id}/agents", response_model=AgentResponse)
async def create_agent(workspace_id: str, request: CreateAgentRequest):
    """Create a new agent in a workspace."""
    try:
        # Convert tool names to ToolType enum if provided
        enabled_tools = None
        if request.enabled_tools:
            enabled_tools = []
            for tool_name in request.enabled_tools:
                try:
                    enabled_tools.append(ToolType(tool_name))
                except ValueError:
                    logger.warning(f"Unknown tool type: {tool_name}")
        
        # Create agent
        agent = workspace_manager.create_agent(
            workspace_id=workspace_id,
            name=request.name,
            description=request.description,
            enabled_tools=enabled_tools,
            custom_instructions=request.custom_instructions,
            system_prompt=request.system_prompt,
            model_settings=request.model_settings,
            tool_configurations=request.tool_configurations
        )
        
        if agent:
            return AgentResponse(
                success=True,
                agent_id=agent.agent_id
            )
        else:
            return AgentResponse(
                success=False,
                error="Failed to create agent"
            )
        
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        return AgentResponse(
            success=False,
            error=str(e)
        )

@app.get("/api/v1/workspaces/{workspace_id}/agents")
async def list_workspace_agents(workspace_id: str):
    """List all agents in a workspace."""
    try:
        agents = workspace_manager.list_agents_in_workspace(workspace_id)
        return {
            "success": True,
            "agents": [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "is_active": agent.is_active,
                    "enabled_tools": [tool.value for tool in agent.enabled_tools],
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat()
                }
                for agent in agents
            ]
        }
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/workspaces/{workspace_id}/agents/{agent_id}")
async def get_agent(workspace_id: str, agent_id: str):
    """Get agent details."""
    try:
        agent = workspace_manager.get_agent(workspace_id, agent_id)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return {
            "success": True,
            "agent": {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "enabled_tools": [tool.value for tool in agent.enabled_tools],
                "custom_instructions": agent.custom_instructions,
                "system_prompt": agent.system_prompt,
                "model_settings": agent.model_settings,
                "tool_configurations": agent.tool_configurations,
                "is_active": agent.is_active,
                "created_at": agent.created_at.isoformat(),
                "updated_at": agent.updated_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        return {"success": False, "error": str(e)}

@app.put("/api/v1/workspaces/{workspace_id}/agents/{agent_id}")
async def update_agent(workspace_id: str, agent_id: str, request: UpdateAgentRequest):
    """Update an agent configuration."""
    try:
        updates = {}
        
        # Build updates dictionary from non-None fields
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.custom_instructions is not None:
            updates["custom_instructions"] = request.custom_instructions
        if request.system_prompt is not None:
            updates["system_prompt"] = request.system_prompt
        if request.model_settings is not None:
            updates["model_settings"] = request.model_settings
        
        # Update the agent
        updated_agent = workspace_manager.update_agent(workspace_id, agent_id, updates)
        
        if updated_agent:
            return AgentResponse(success=True, agent_id=agent_id, message="Agent updated successfully")
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating agent: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/workspaces/{workspace_id}/agents/{agent_id}")
async def delete_agent(workspace_id: str, agent_id: str):
    """Delete an agent."""
    try:
        success = workspace_manager.delete_agent(workspace_id, agent_id)
        if success:
            return {"success": True, "message": "Agent deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting agent: {str(e)}")
        return {"success": False, "error": str(e)}

@app.get("/api/v1/workspaces/{workspace_id}/stats")
async def get_workspace_stats(workspace_id: str):
    """Get workspace statistics."""
    try:
        stats = workspace_manager.get_workspace_stats(workspace_id)
        if stats:
            return {"success": True, "stats": stats}
        else:
            raise HTTPException(status_code=404, detail="Workspace not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting workspace stats: {str(e)}")
        return {"success": False, "error": str(e)}

# Multitenant RAG Endpoints
@app.post("/api/v1/rag/grouped-search", response_model=GroupedSearchResponse)
async def grouped_search(request: GroupedSearchRequest):
    """Perform a grouped search across the knowledge base."""
    try:
        org_id = request.organization_id
        workspace_id = request.workspace_id
        
        # Ensure organization is initialized
        if org_id not in initialized_orgs:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found. Please set it up first.")
        
        # Initialize tools if needed
        await tool_manager.initialize_tools_for_organization(org_id, workspace_id)
        
        # Get RAG tool instance
        rag_tool = tool_manager.get_tool_instance(org_id, "rag", workspace_id)
        if not rag_tool:
            raise HTTPException(status_code=500, detail="RAG tool not available")
        
        # Execute grouped search
        result = await rag_tool.execute_grouped_search(
            query=request.query,
            group_by=request.group_by,
            limit=request.limit,
            group_size=request.group_size,
            filter_conditions=request.filter_conditions
        )
        
        if result.success:
            return GroupedSearchResponse(
                success=True,
                query=result.data["query"],
                groups=result.data["groups"],
                total_groups=result.data["total_groups"],
                group_by=result.data["group_by"],
                organization_id=org_id,
                workspace_id=workspace_id
            )
        else:
            return GroupedSearchResponse(
                success=False,
                query=request.query,
                groups={},
                total_groups=0,
                group_by=request.group_by,
                organization_id=org_id,
                workspace_id=workspace_id,
                error=result.error
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing grouped search: {str(e)}")
        return GroupedSearchResponse(
            success=False,
            query=request.query,
            groups={},
            total_groups=0,
            group_by=request.group_by,
            organization_id=request.organization_id,
            workspace_id=request.workspace_id,
            error=str(e)
        )

@app.post("/api/v1/rag/tenant-stats", response_model=TenantStatsResponse)
async def get_tenant_statistics(request: TenantStatsRequest):
    """Get statistics for a specific tenant (organization/workspace)."""
    try:
        org_id = request.organization_id
        workspace_id = request.workspace_id
        
        # Ensure organization is initialized
        if org_id not in initialized_orgs:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found. Please set it up first.")
        
        # Initialize tools if needed
        await tool_manager.initialize_tools_for_organization(org_id, workspace_id)
        
        # Get RAG tool instance
        rag_tool = tool_manager.get_tool_instance(org_id, "rag", workspace_id)
        if not rag_tool:
            raise HTTPException(status_code=500, detail="RAG tool not available")
          # Get tenant statistics with timeout
        try:
            import asyncio
            result = await asyncio.wait_for(
                rag_tool.get_tenant_statistics(),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Tenant statistics timeout for org '{org_id}', workspace '{workspace_id}'")
            return TenantStatsResponse(
                success=False,
                organization_id=org_id,
                workspace_id=workspace_id or "org_default",
                total_documents=0,
                total_chunks=0,
                collection_name="",
                avg_chunks_per_document=0.0,
                error="Statistics collection timeout - try again later"
            )
        
        if result.success:
            data = result.data
            return TenantStatsResponse(
                success=True,
                organization_id=data["organization_id"],
                workspace_id=data["workspace_id"],
                total_documents=data["total_documents"],
                total_chunks=data["total_chunks"],
                collection_name=data["collection_name"],
                avg_chunks_per_document=data["avg_chunks_per_document"]
            )
        else:
            return TenantStatsResponse(
                success=False,
                organization_id=org_id,
                workspace_id=workspace_id or "org_default",
                total_documents=0,
                total_chunks=0,
                collection_name="",
                avg_chunks_per_document=0.0,
                error=result.error
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting tenant statistics: {str(e)}")
        return TenantStatsResponse(
            success=False,
            organization_id=request.organization_id,
            workspace_id=request.workspace_id or "org_default",
            total_documents=0,
            total_chunks=0,
            collection_name="",
            avg_chunks_per_document=0.0,
            error=str(e)
        )

@app.post("/api/v1/rag/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_tenant_data(request: BulkDeleteRequest):
    """Bulk delete all data for a specific tenant."""
    try:
        org_id = request.organization_id
        workspace_id = request.workspace_id
        
        # Ensure organization is initialized
        if org_id not in initialized_orgs:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found. Please set it up first.")
        
        # Initialize tools if needed
        await tool_manager.initialize_tools_for_organization(org_id, workspace_id)
        
        # Get RAG tool instance
        rag_tool = tool_manager.get_tool_instance(org_id, "rag", workspace_id)
        if not rag_tool:
            raise HTTPException(status_code=500, detail="RAG tool not available")
        
        # Execute bulk delete with confirmation
        result = await rag_tool.bulk_delete_tenant_data(request.confirm_organization_id)
        
        if result.success:
            data = result.data
            return BulkDeleteResponse(
                success=True,
                organization_id=data["organization_id"],
                workspace_id=data["workspace_id"],
                deleted_documents=data["deleted_documents"],
                deleted_chunks=data["deleted_chunks"],
                collection_name=data["collection"]
            )
        else:
            return BulkDeleteResponse(
                success=False,
                organization_id=org_id,
                workspace_id=workspace_id or "org_default",
                deleted_documents=0,
                deleted_chunks=0,
                collection_name="",
                error=result.error
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during bulk delete: {str(e)}")
        return BulkDeleteResponse(
            success=False,
            organization_id=request.organization_id,
            workspace_id=request.workspace_id or "org_default",
            deleted_documents=0,
            deleted_chunks=0,
            collection_name="",
            error=str(e)
        )

@app.post("/api/v1/rag/migrate-data", response_model=MigrationResponse)
async def migrate_tenant_data(request: MigrationRequest):
    """Migrate data from one workspace to another within the same organization."""
    try:
        org_id = request.organization_id
        source_workspace_id = request.source_workspace_id
        target_workspace_id = request.target_workspace_id
        
        # Ensure organization is initialized
        if org_id not in initialized_orgs:
            raise HTTPException(status_code=404, detail=f"Organization '{org_id}' not found. Please set it up first.")
        
        # Initialize tools if needed
        await tool_manager.initialize_tools_for_organization(org_id, source_workspace_id)
        
        # Get RAG tool instance for source workspace
        rag_tool = tool_manager.get_tool_instance(org_id, "rag", source_workspace_id)
        if not rag_tool:
            raise HTTPException(status_code=500, detail="RAG tool not available")
        
        # Execute migration
        result = await rag_tool.migrate_tenant_data(
            target_workspace_id=target_workspace_id,
            document_filter=request.document_filter
        )
        
        if result.success:
            data = result.data
            return MigrationResponse(
                success=True,
                source_organization_id=data["source_organization_id"],
                source_workspace_id=data["source_workspace_id"],
                target_workspace_id=data["target_workspace_id"],
                migrated_chunks=data["migrated_chunks"],
                collection_name=data["collection"]
            )
        else:
            return MigrationResponse(
                success=False,
                source_organization_id=org_id,
                source_workspace_id=source_workspace_id or "org_default",
                target_workspace_id=target_workspace_id,
                migrated_chunks=0,
                collection_name="",
                error=result.error
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        return MigrationResponse(
            success=False,
            source_organization_id=request.organization_id,
            source_workspace_id=request.source_workspace_id or "org_default",
            target_workspace_id=request.target_workspace_id,
            migrated_chunks=0,
            collection_name="",
            error=str(e)
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
