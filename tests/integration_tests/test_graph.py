import pytest
import os
import sys

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agent.enhanced_graph import execute_agent_query
from agent.workspace_manager import workspace_manager
from agent.config import config_manager, OrganizationConfig, ToolType

pytestmark = pytest.mark.anyio


@pytest.mark.langsmith
async def test_enhanced_agent_simple_query() -> None:
    """Test the enhanced agent system with a simple query."""
    
    # Set up test organization and workspace
    org_id = "test_org"
    workspace_name = "test_workspace"
    agent_name = "test_agent"
    
    # Create organization config
    org_config = OrganizationConfig(
        organization_id=org_id,
        enabled_tools=[ToolType.RAG]
    )
    config_manager.register_organization(org_config)
    
    # Create workspace
    workspace = workspace_manager.create_workspace(
        organization_id=org_id,
        name=workspace_name,
        description="Test workspace"
    )
    
    # Create agent
    agent = workspace_manager.create_agent(
        workspace_id=workspace.workspace_id,
        name=agent_name,
        description="Test agent"
    )
    
    # Execute query
    result = await execute_agent_query(
        organization_id=org_id,
        query="Hello, can you help me?",
        workspace_id=workspace.workspace_id,
        agent_id=agent.agent_id
    )
    
    assert result is not None
    assert result["success"] is True
    assert "response" in result
    assert result["organization_id"] == org_id
    assert result["workspace_id"] == workspace.workspace_id
    assert result["agent_id"] == agent.agent_id
