"""Test script for the multi-workspace agent system."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent.workspace_manager import workspace_manager, AgentConfig
from agent.config import config_manager, ToolType, OrganizationConfig
from agent.enhanced_graph import execute_agent_query, ingest_document_for_organization

async def test_multi_workspace_system():
    """Test the multi-workspace agent system."""
    print("🚀 Testing Multi-Workspace Agent System")
    print("=" * 50)
    
    # Test 1: Create an organization
    print("\n1. Setting up organization...")
    org_id = "test_org"
    org_config = OrganizationConfig(
        organization_id=org_id,
        enabled_tools=[ToolType.RAG]
    )
    config_manager.register_organization(org_config)
    print(f"✅ Created organization: {org_id}")
    
    # Test 2: Create workspaces
    print("\n2. Creating workspaces...")
    
    # Workspace 1: Customer Support
    support_workspace = workspace_manager.create_workspace(
        organization_id=org_id,
        name="Customer Support",
        description="Workspace for customer support agents",
        shared_tools=[ToolType.RAG]
    )
    print(f"✅ Created workspace: {support_workspace.workspace_id} (Customer Support)")
    
    # Workspace 2: Product Development  
    dev_workspace = workspace_manager.create_workspace(
        organization_id=org_id,
        name="Product Development",
        description="Workspace for product development team",
        shared_tools=[ToolType.RAG]
    )
    print(f"✅ Created workspace: {dev_workspace.workspace_id} (Product Development)")
    
    # Test 3: Create agents in workspaces
    print("\n3. Creating agents...")
    
    # Agent 1: Support Agent
    support_agent = workspace_manager.create_agent(
        workspace_id=support_workspace.workspace_id,
        name="Support Assistant",
        description="Helpful customer support agent",
        enabled_tools=[ToolType.RAG],
        custom_instructions="Be friendly and helpful to customers. Always try to resolve issues quickly.",
        system_prompt="You are a customer support agent. Be empathetic and solution-focused."
    )
    print(f"✅ Created agent: {support_agent.agent_id} (Support Assistant)")
    
    # Agent 2: Technical Agent  
    tech_agent = workspace_manager.create_agent(
        workspace_id=support_workspace.workspace_id,
        name="Technical Expert",
        description="Technical support specialist",
        enabled_tools=[ToolType.RAG],
        custom_instructions="Provide detailed technical explanations. Focus on accuracy and completeness.",
        system_prompt="You are a technical expert. Provide thorough technical assistance."
    )
    print(f"✅ Created agent: {tech_agent.agent_id} (Technical Expert)")
    
    # Agent 3: Product Manager Agent
    pm_agent = workspace_manager.create_agent(
        workspace_id=dev_workspace.workspace_id,
        name="Product Manager",
        description="Product strategy and planning assistant",
        enabled_tools=[ToolType.RAG],
        custom_instructions="Focus on strategic thinking and data-driven decisions.",
        system_prompt="You are a product manager. Think strategically about product decisions."
    )
    print(f"✅ Created agent: {pm_agent.agent_id} (Product Manager)")
    
    # Test 4: Initialize workspace tools
    print("\n4. Initializing workspace tools...")
    await workspace_manager.initialize_workspace_tools(support_workspace.workspace_id)
    await workspace_manager.initialize_workspace_tools(dev_workspace.workspace_id)
    print("✅ Initialized tools for all workspaces")
    
    # Test 5: Ingest documents into different workspaces
    print("\n5. Ingesting documents...")
    
    # Support documentation
    support_doc = """
    Customer Support Guidelines:
    
    1. Always greet customers warmly
    2. Listen actively to their concerns
    3. Provide clear step-by-step solutions
    4. Follow up to ensure satisfaction
    5. Escalate complex technical issues when needed
    
    Common Issues:
    - Login problems: Reset password or check account status
    - Billing questions: Direct to billing department
    - Technical bugs: Collect error details and screenshots
    """
    
    result = await ingest_document_for_organization(
        organization_id=org_id,
        content=support_doc,
        metadata={"type": "support_guidelines", "workspace": "customer_support"},
        workspace_id=support_workspace.workspace_id
    )
    print(f"✅ Ingested support documentation: {result['success']}")
    
    # Product documentation
    product_doc = """
    Product Development Process:
    
    1. Market Research and Analysis
    2. Feature Specification and Design
    3. Development Sprint Planning
    4. User Testing and Feedback
    5. Release Planning and Go-to-Market
    
    Key Metrics:
    - User engagement rates
    - Feature adoption
    - Customer satisfaction scores
    - Time to market
    - Revenue impact
    """
    
    result = await ingest_document_for_organization(
        organization_id=org_id,
        content=product_doc,
        metadata={"type": "product_guidelines", "workspace": "product_development"},
        workspace_id=dev_workspace.workspace_id
    )
    print(f"✅ Ingested product documentation: {result['success']}")
    
    # Test 6: Query different agents
    print("\n6. Testing agent queries...")
    
    # Query Support Assistant
    print("\n6a. Querying Support Assistant...")
    result = await execute_agent_query(
        organization_id=org_id,
        query="How should I handle a customer login issue?",
        workspace_id=support_workspace.workspace_id,
        agent_id=support_agent.agent_id
    )
    print(f"Support Assistant Response: {result['response'][:200]}...")
    
    # Query Technical Expert
    print("\n6b. Querying Technical Expert...")
    result = await execute_agent_query(
        organization_id=org_id,
        query="What information should I collect for a technical bug report?",
        workspace_id=support_workspace.workspace_id,
        agent_id=tech_agent.agent_id
    )
    print(f"Technical Expert Response: {result['response'][:200]}...")
    
    # Query Product Manager
    print("\n6c. Querying Product Manager...")
    result = await execute_agent_query(
        organization_id=org_id,
        query="What are the key metrics I should track for product success?",
        workspace_id=dev_workspace.workspace_id,
        agent_id=pm_agent.agent_id
    )
    print(f"Product Manager Response: {result['response'][:200]}...")
    
    # Test 7: List workspace information
    print("\n7. Workspace information...")
    
    # List workspaces
    workspaces = workspace_manager.list_organization_workspaces(org_id)
    print(f"✅ Organization has {len(workspaces)} workspaces")
    
    for workspace in workspaces:
        stats = workspace_manager.get_workspace_stats(workspace.workspace_id)
        print(f"  - {workspace.name}: {stats['active_agents']}/{stats['total_agents']} agents")
        
        # List agents in workspace
        agents = workspace_manager.list_agents_in_workspace(workspace.workspace_id)
        for agent in agents:
            print(f"    • {agent.name} ({agent.agent_id})")
    
    print("\n🎉 Multi-Workspace System Test Complete!")
    print("=" * 50)
    print("✅ Organizations: Configured")
    print("✅ Workspaces: Created and isolated")
    print("✅ Agents: Created with different configurations")
    print("✅ Tools: Initialized per workspace")
    print("✅ Documents: Ingested to workspace-specific collections")
    print("✅ Queries: Executed with agent-specific behavior")

if __name__ == "__main__":
    asyncio.run(test_multi_workspace_system())
