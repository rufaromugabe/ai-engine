from typing import List, Dict, Any, Optional
from agent.config import config_manager, OrganizationConfig, ToolType
from agent.workspace_manager import workspace_manager
from agent.enhanced_graph import get_organization_knowledge_base_info
import os

class DummyDataProvider:
    """Dummy data provider for organizations, workspaces, and agents."""
    DUMMY_ORGANIZATIONS = [
        {
            "organization_id": "dev-org",
            "name": "Development Org",
            "description": "A dummy organization for development/testing.",
            "enabled_tools": ["rag"]
        },
        {
            "organization_id": "test-org",
            "name": "Test Org",
            "description": "A dummy organization for testing purposes.",
            "enabled_tools": ["rag"]
        }
    ]

    def get_organizations(self) -> List[Dict[str, Any]]:
        """Return dummy organizations and ensure they exist in config_manager for Qdrant compatibility."""
        for org in self.DUMMY_ORGANIZATIONS:
            org_id = org["organization_id"]
            # Ensure org exists in config_manager
            try:
                config_manager.get_organization_config(org_id)
            except Exception:
                config_manager.register_organization(OrganizationConfig(
                    organization_id=org_id,
                    enabled_tools=[ToolType.RAG]
                ))
        return self.DUMMY_ORGANIZATIONS

    def get_workspaces(self, organization_id: str) -> List[Dict[str, Any]]:
        """Return dummy or real workspaces for an organization, and ensure dummy workspace exists in workspace_manager for Qdrant compatibility."""
        workspaces = workspace_manager.list_organization_workspaces(organization_id)
        if workspaces:
            return [
                {
                    "workspace_id": ws.workspace_id,
                    "name": ws.name,
                    "description": ws.description,
                    "organization_id": ws.organization_id,
                    "status": ws.status.value,
                    "created_at": ws.created_at.isoformat(),
                    "updated_at": ws.updated_at.isoformat(),
                }
                for ws in workspaces
            ]
        # Dummy fallback
        dummy_ws = {
            "workspace_id": "dev-workspace",
            "name": "Development Workspace",
            "description": "A dummy workspace for dev.",
            "organization_id": organization_id,
            "status": "active",
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
        # Ensure dummy workspace exists in workspace_manager
        ws_obj = workspace_manager.get_workspace(dummy_ws["workspace_id"])
        if not ws_obj:
            workspace_manager.create_workspace(
                organization_id=organization_id,
                name=dummy_ws["name"],
                description=dummy_ws["description"],
                workspace_id=dummy_ws["workspace_id"]
            )
        return [dummy_ws]

    def get_agents(self, workspace_id: str) -> List[Dict[str, Any]]:
        """Return dummy or real agents for a workspace."""
        ws = workspace_manager.get_workspace(workspace_id)
        if ws and ws.agents:
            return [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "enabled_tools": [tool.value for tool in agent.enabled_tools],
                    "is_active": agent.is_active,
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat(),
                }
                for agent in ws.agents.values()
            ]
        # Dummy fallback
        return [
            {
                "agent_id": "dev-agent",
                "name": "Development Agent",
                "description": "A dummy agent for dev.",
                "enabled_tools": ["rag"],
                "is_active": True,
                "created_at": "2024-01-01T00:00:00",
                "updated_at": "2024-01-01T00:00:00"
            }
        ]

    async def get_knowledge_base_info(self, organization_id: str, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        """Return real knowledge base info for a workspace (can be replaced with dummy)."""
        return await get_organization_knowledge_base_info(organization_id, workspace_id)

class RealDataProvider:
    """Real data provider for organizations, workspaces, and agents (stub for future extension)."""
    def get_organizations(self) -> List[Dict[str, Any]]:
        # Use config_manager to get real organizations
        return [
            {
                "organization_id": org.organization_id,
                "name": getattr(org, "name", org.organization_id),
                "description": getattr(org, "description", "")
            }
            for org in getattr(config_manager, "_organizations", {}).values()
        ]

    def get_workspaces(self, organization_id: str) -> List[Dict[str, Any]]:
        workspaces = workspace_manager.list_organization_workspaces(organization_id)
        return [
            {
                "workspace_id": ws.workspace_id,
                "name": ws.name,
                "description": ws.description,
                "organization_id": ws.organization_id,
                "status": ws.status.value,
                "created_at": ws.created_at.isoformat(),
                "updated_at": ws.updated_at.isoformat(),
            }
            for ws in workspaces
        ]

    def get_agents(self, workspace_id: str) -> List[Dict[str, Any]]:
        ws = workspace_manager.get_workspace(workspace_id)
        if ws and ws.agents:
            return [
                {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "enabled_tools": [tool.value for tool in agent.enabled_tools],
                    "is_active": agent.is_active,
                    "created_at": agent.created_at.isoformat(),
                    "updated_at": agent.updated_at.isoformat(),
                }
                for agent in ws.agents.values()
            ]
        return []

    async def get_knowledge_base_info(self, organization_id: str, workspace_id: Optional[str] = None) -> Dict[str, Any]:
        return await get_organization_knowledge_base_info(organization_id, workspace_id)

# Factory for switching data provider (dummy/real)
def get_data_provider() -> DummyDataProvider:
    """Return the current data provider (dummy or real) based on DUMMY_MODE env variable."""
    dummy_mode = os.getenv("DUMMY_MODE", "true").lower() == "true"
    if dummy_mode:
        return DummyDataProvider()
    else:
        return RealDataProvider()

