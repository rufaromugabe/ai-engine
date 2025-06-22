"""Workspace management for multi-tenant AI agent system."""

from __future__ import annotations

import logging
import json
import os
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime

from .config import config_manager, ToolType, OrganizationConfig
from .tool_manager import tool_manager

logger = logging.getLogger(__name__)

class WorkspaceStatus(str, Enum):
    """Workspace status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    MAINTENANCE = "maintenance"

@dataclass
class AgentConfig:
    """Configuration for an individual agent within a workspace."""
    agent_id: str
    name: str
    description: str
    enabled_tools: List[ToolType]
    custom_instructions: str = ""
    system_prompt: str = ""
    model_settings: Dict[str, Any] = field(default_factory=dict)
    tool_configurations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

@dataclass
class WorkspaceConfig:
    """Configuration for a workspace containing multiple agents."""
    workspace_id: str
    organization_id: str
    name: str
    description: str
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    shared_tools: List[ToolType] = field(default_factory=list)
    shared_settings: Dict[str, Any] = field(default_factory=dict)
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def add_agent(self, agent_config: AgentConfig) -> None:
        """Add an agent to the workspace."""
        self.agents[agent_config.agent_id] = agent_config
        self.updated_at = datetime.now()
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the workspace."""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.updated_at = datetime.now()
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentConfig]:
        """Get an agent configuration."""
        return self.agents.get(agent_id)
    
    def list_active_agents(self) -> List[AgentConfig]:
        """List all active agents in the workspace."""
        return [agent for agent in self.agents.values() if agent.is_active]

class WorkspaceManager:
    """Manages workspaces and agents for organizations."""
    
    def __init__(self):
        self._workspaces: Dict[str, WorkspaceConfig] = {}
        self._organization_workspaces: Dict[str, Set[str]] = {}  # org_id -> workspace_ids
        self._agent_instances: Dict[str, Dict[str, Any]] = {}  # workspace_id -> {agent_id: graph_instance}
    
    def create_workspace(
        self,
        organization_id: str,
        name: str,
        description: str = "",
        workspace_id: Optional[str] = None,
        shared_tools: Optional[List[ToolType]] = None,
        shared_settings: Optional[Dict[str, Any]] = None
    ) -> WorkspaceConfig:
        """Create a new workspace for an organization."""
        if workspace_id is None:
            workspace_id = f"ws_{uuid.uuid4().hex[:8]}"
        
        if shared_tools is None:
            shared_tools = [ToolType.RAG]
        
        if shared_settings is None:
            shared_settings = {}
        
        workspace = WorkspaceConfig(
            workspace_id=workspace_id,
            organization_id=organization_id,
            name=name,
            description=description,
            shared_tools=shared_tools,
            shared_settings=shared_settings
        )
        
        self._workspaces[workspace_id] = workspace
        
        # Track workspace for organization
        if organization_id not in self._organization_workspaces:
            self._organization_workspaces[organization_id] = set()
        self._organization_workspaces[organization_id].add(workspace_id)
        
        logger.info(f"Created workspace {workspace_id} for organization {organization_id}")
        return workspace
    
    def get_workspace(self, workspace_id: str) -> Optional[WorkspaceConfig]:
        """Get a workspace configuration."""
        return self._workspaces.get(workspace_id)
    
    def list_organization_workspaces(self, organization_id: str) -> List[WorkspaceConfig]:
        """List all workspaces for an organization."""
        workspace_ids = self._organization_workspaces.get(organization_id, set())
        return [self._workspaces[ws_id] for ws_id in workspace_ids if ws_id in self._workspaces]
    
    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and cleanup associated resources."""
        if workspace_id not in self._workspaces:
            return False
        
        workspace = self._workspaces[workspace_id]
        organization_id = workspace.organization_id
        
        # Cleanup agent instances
        if workspace_id in self._agent_instances:
            del self._agent_instances[workspace_id]
        
        # Remove from organization tracking
        if organization_id in self._organization_workspaces:
            self._organization_workspaces[organization_id].discard(workspace_id)
            if not self._organization_workspaces[organization_id]:
                del self._organization_workspaces[organization_id]
        
        # Remove workspace
        del self._workspaces[workspace_id]
        
        logger.info(f"Deleted workspace {workspace_id}")
        return True
    
    def create_agent(
        self,
        workspace_id: str,
        name: str,
        description: str = "",
        enabled_tools: Optional[List[ToolType]] = None,
        custom_instructions: str = "",
        system_prompt: str = "",
        model_settings: Optional[Dict[str, Any]] = None,
        tool_configurations: Optional[Dict[str, Dict[str, Any]]] = None,
        agent_id: Optional[str] = None
    ) -> Optional[AgentConfig]:
        """Create a new agent in a workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            logger.error(f"Workspace {workspace_id} not found")
            return None
        
        if agent_id is None:
            agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        if enabled_tools is None:
            enabled_tools = workspace.shared_tools.copy()
        
        if model_settings is None:
            model_settings = {}
        
        if tool_configurations is None:
            tool_configurations = {}
        
        agent_config = AgentConfig(
            agent_id=agent_id,
            name=name,
            description=description,
            enabled_tools=enabled_tools,
            custom_instructions=custom_instructions,
            system_prompt=system_prompt,
            model_settings=model_settings,
            tool_configurations=tool_configurations
        )
        
        workspace.add_agent(agent_config)
        
        logger.info(f"Created agent {agent_id} in workspace {workspace_id}")
        return agent_config
    
    def get_agent(self, workspace_id: str, agent_id: str) -> Optional[AgentConfig]:
        """Get an agent configuration."""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return workspace.get_agent(agent_id)
        return None
    
    def update_agent(
        self,
        workspace_id: str,
        agent_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update an agent configuration."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        agent = workspace.get_agent(agent_id)
        if not agent:
            return False
        
        # Update agent fields
        for key, value in updates.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        
        agent.updated_at = datetime.now()
        workspace.updated_at = datetime.now()
        
        # Clear cached instance to force reinitialization
        if workspace_id in self._agent_instances and agent_id in self._agent_instances[workspace_id]:
            del self._agent_instances[workspace_id][agent_id]
        
        logger.info(f"Updated agent {agent_id} in workspace {workspace_id}")
        return True
    
    def delete_agent(self, workspace_id: str, agent_id: str) -> bool:
        """Delete an agent from a workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Remove from instances
        if workspace_id in self._agent_instances and agent_id in self._agent_instances[workspace_id]:
            del self._agent_instances[workspace_id][agent_id]
        
        # Remove from workspace
        return workspace.remove_agent(agent_id)
    
    async def initialize_workspace_tools(self, workspace_id: str) -> bool:
        """Initialize tools for all agents in a workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return False
        
        # Collect all unique tools needed for the workspace
        all_tools = set(workspace.shared_tools)
        for agent in workspace.agents.values():
            if agent.is_active:
                all_tools.update(agent.enabled_tools)
        
        # Initialize tools for the organization with workspace context
        success = await tool_manager.initialize_tools_for_organization(
            workspace.organization_id, 
            workspace_id=workspace_id,
            tools=list(all_tools)
        )
        
        return success
    
    def get_workspace_tools(self, workspace_id: str) -> List[ToolType]:
        """Get all tools available in a workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return []
        
        all_tools = set(workspace.shared_tools)
        for agent in workspace.agents.values():
            if agent.is_active:
                all_tools.update(agent.enabled_tools)
        
        return list(all_tools)
    
    def list_agents_in_workspace(self, workspace_id: str) -> List[AgentConfig]:
        """List all agents in a workspace."""
        workspace = self.get_workspace(workspace_id)
        if workspace:
            return list(workspace.agents.values())
        return []
    
    def get_workspace_stats(self, workspace_id: str) -> Dict[str, Any]:
        """Get statistics for a workspace."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return {}
        
        active_agents = len([a for a in workspace.agents.values() if a.is_active])
        total_agents = len(workspace.agents)
        tools_count = len(self.get_workspace_tools(workspace_id))
        
        return {
            "workspace_id": workspace_id,
            "organization_id": workspace.organization_id,
            "name": workspace.name,
            "status": workspace.status.value,
            "total_agents": total_agents,
            "active_agents": active_agents,
            "tools_count": tools_count,
            "created_at": workspace.created_at.isoformat(),
            "updated_at": workspace.updated_at.isoformat()
        }

# Global workspace manager instance
workspace_manager = WorkspaceManager()
