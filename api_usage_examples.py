"""
Multi-Workspace AI Agent System - API Usage Examples

This system now supports:
1. Organizations with multiple workspaces
2. Workspaces with different agents and tool configurations  
3. Agent-specific behavior and instructions
4. Workspace-isolated knowledge bases
5. Environment-based configuration

API Endpoints Overview:
========================

Organization Management:
- POST /api/v1/setup-organization
- GET /api/v1/organizations

Workspace Management:
- POST /api/v1/workspaces
- GET /api/v1/organizations/{org_id}/workspaces
- GET /api/v1/workspaces/{workspace_id}
- DELETE /api/v1/workspaces/{workspace_id}
- GET /api/v1/workspaces/{workspace_id}/stats

Agent Management:
- POST /api/v1/workspaces/{workspace_id}/agents
- GET /api/v1/workspaces/{workspace_id}/agents
- GET /api/v1/workspaces/{workspace_id}/agents/{agent_id}
- PUT /api/v1/workspaces/{workspace_id}/agents/{agent_id}
- DELETE /api/v1/workspaces/{workspace_id}/agents/{agent_id}

Enhanced Query & Document Management:
- POST /api/v1/query (now supports workspace_id and agent_id)
- POST /api/v1/ingest (now supports workspace_id)
- GET /api/v1/knowledge-base/{org_id} (now supports workspace_id parameter)

Usage Examples:
===============
"""

import requests
import json

# Base URL for your API
BASE_URL = "http://localhost:8000/api/v1"

def example_setup_organization():
    """Example: Set up a new organization"""
    print("1. Setting up organization...")
    
    payload = {
        "organization_id": "acme_corp",
        "enabled_tools": ["rag"],
        "rag_settings": {
            "top_k": 5,
            "similarity_threshold": 0.7
        }
    }
    
    response = requests.post(f"{BASE_URL}/setup-organization", json=payload)
    print(f"Response: {response.json()}")
    return response.json()

def example_create_workspace():
    """Example: Create a workspace for customer support"""
    print("\n2. Creating customer support workspace...")
    
    payload = {
        "organization_id": "acme_corp",
        "name": "Customer Support",
        "description": "Workspace for customer support team",
        "shared_tools": ["rag"],
        "shared_settings": {
            "response_style": "friendly"
        }
    }
    
    response = requests.post(f"{BASE_URL}/workspaces", json=payload)
    print(f"Response: {response.json()}")
    return response.json()

def example_create_agents(workspace_id):
    """Example: Create different agents in the workspace"""
    print(f"\n3. Creating agents in workspace {workspace_id}...")
    
    # Agent 1: General Support Agent
    agent1_payload = {
        "workspace_id": workspace_id,
        "name": "General Support Assistant",
        "description": "Handles general customer inquiries",
        "enabled_tools": ["rag"],
        "custom_instructions": "Be friendly, helpful, and always try to resolve issues quickly. Use a warm, conversational tone.",
        "system_prompt": "You are a helpful customer support agent for Acme Corp. Always be polite and solution-focused.",
        "model_settings": {
            "temperature": 0.3,
            "max_tokens": 500
        }
    }
    
    response1 = requests.post(f"{BASE_URL}/workspaces/{workspace_id}/agents", json=agent1_payload)
    print(f"General Agent Response: {response1.json()}")
    
    # Agent 2: Technical Support Specialist
    agent2_payload = {
        "workspace_id": workspace_id,
        "name": "Technical Support Specialist",
        "description": "Handles complex technical issues",
        "enabled_tools": ["rag"],
        "custom_instructions": "Provide detailed technical explanations. Be thorough and accurate. Ask for specific error details when needed.",
        "system_prompt": "You are a technical support specialist. Focus on diagnosing and solving technical problems systematically.",
        "model_settings": {
            "temperature": 0.1,
            "max_tokens": 800
        }
    }
    
    response2 = requests.post(f"{BASE_URL}/workspaces/{workspace_id}/agents", json=agent2_payload)
    print(f"Technical Agent Response: {response2.json()}")
    
    return response1.json().get("agent_id"), response2.json().get("agent_id")

def example_ingest_documents(workspace_id):
    """Example: Ingest workspace-specific documents"""
    print(f"\n4. Ingesting documents to workspace {workspace_id}...")
    
    # Support knowledge base
    support_doc = """
    Acme Corp Customer Support Guidelines:
    
    Response Time Standards:
    - General inquiries: 24 hours
    - Technical issues: 48 hours  
    - Billing questions: 4 hours
    
    Common Solutions:
    1. Password Reset: Direct users to account settings
    2. Billing Issues: Check payment method and subscription status
    3. Feature Questions: Refer to user manual or provide demo
    4. Technical Bugs: Collect error logs and system information
    
    Escalation Process:
    - Level 1: General support agents
    - Level 2: Technical specialists
    - Level 3: Engineering team
    """
    
    payload = {
        "content": support_doc,
        "organization_id": "acme_corp",
        "workspace_id": workspace_id,
        "metadata": {
            "document_type": "support_guidelines",
            "version": "1.0",
            "department": "customer_support"
        }
    }
    
    response = requests.post(f"{BASE_URL}/ingest", json=payload)
    print(f"Document Ingestion Response: {response.json()}")

def example_query_different_agents(workspace_id, general_agent_id, tech_agent_id):
    """Example: Query different agents with the same question"""
    print(f"\n5. Querying different agents...")
    
    user_question = "I'm having trouble logging into my account. What should I do?"
    
    # Query General Support Agent
    print("\n5a. Querying General Support Agent...")
    payload1 = {
        "query": user_question,
        "organization_id": "acme_corp",
        "workspace_id": workspace_id,
        "agent_id": general_agent_id
    }
    
    response1 = requests.post(f"{BASE_URL}/query", json=payload1)
    result1 = response1.json()
    print(f"General Agent Response: {result1['response'][:200]}...")
    
    # Query Technical Support Specialist
    print("\n5b. Querying Technical Support Specialist...")
    payload2 = {
        "query": user_question,
        "organization_id": "acme_corp", 
        "workspace_id": workspace_id,
        "agent_id": tech_agent_id
    }
    
    response2 = requests.post(f"{BASE_URL}/query", json=payload2)
    result2 = response2.json()
    print(f"Technical Agent Response: {result2['response'][:200]}...")

def example_workspace_management(workspace_id):
    """Example: Workspace management operations"""
    print(f"\n6. Workspace management...")
    
    # Get workspace details
    response = requests.get(f"{BASE_URL}/workspaces/{workspace_id}")
    workspace_info = response.json()
    print(f"Workspace Info: {json.dumps(workspace_info, indent=2)}")
    
    # Get workspace statistics
    response = requests.get(f"{BASE_URL}/workspaces/{workspace_id}/stats")
    stats = response.json()
    print(f"Workspace Stats: {stats}")
    
    # List all agents in workspace
    response = requests.get(f"{BASE_URL}/workspaces/{workspace_id}/agents")
    agents = response.json()
    print(f"Agents in workspace: {len(agents['agents'])}")

def example_update_agent(workspace_id, agent_id):
    """Example: Update an agent's configuration"""
    print(f"\n7. Updating agent configuration...")
    
    update_payload = {
        "custom_instructions": "Be extra friendly and always offer additional help. End responses with 'Is there anything else I can help you with?'",
        "model_settings": {
            "temperature": 0.4,
            "max_tokens": 600
        }
    }
    
    response = requests.put(f"{BASE_URL}/workspaces/{workspace_id}/agents/{agent_id}", json=update_payload)
    print(f"Update Response: {response.json()}")

def run_complete_example():
    """Run a complete example workflow"""
    print("🚀 Multi-Workspace AI Agent System - Complete Example")
    print("=" * 60)
    
    try:
        # 1. Setup organization
        org_result = example_setup_organization()
        
        # 2. Create workspace
        workspace_result = example_create_workspace()
        workspace_id = workspace_result.get("workspace_id")
        
        if not workspace_id:
            print("❌ Failed to create workspace")
            return
        
        # 3. Create agents
        general_agent_id, tech_agent_id = example_create_agents(workspace_id)
        
        # 4. Ingest documents
        example_ingest_documents(workspace_id)
        
        # 5. Query agents
        if general_agent_id and tech_agent_id:
            example_query_different_agents(workspace_id, general_agent_id, tech_agent_id)
        
        # 6. Workspace management
        example_workspace_management(workspace_id)
        
        # 7. Update agent
        if general_agent_id:
            example_update_agent(workspace_id, general_agent_id)
        
        print("\n🎉 Complete example finished successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during example: {str(e)}")

if __name__ == "__main__":
    # Make sure your API server is running on localhost:8000
    print("Before running this example, make sure to:")
    print("1. Start the API server: python main.py")
    print("2. Ensure your .env file has the correct API keys")
    print("3. Run this example: python api_usage_examples.py")
    print()
      # Uncomment the line below to run the complete example
    run_complete_example()
    
    print("""
Environment Variables Setup:
===========================

Make sure your .env file contains:

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration  
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Multi-tenant Configuration
ENABLE_MULTI_TENANT=true

# Workspace Configuration
ENABLE_WORKSPACE_ISOLATION=true
MAX_WORKSPACES_PER_ORG=10
MAX_AGENTS_PER_WORKSPACE=5

# Agent Configuration
DEFAULT_AGENT_MODEL=gpt-4
DEFAULT_AGENT_TEMPERATURE=0.1
DEFAULT_AGENT_MAX_TOKENS=1000

API Architecture:
================

Organizations
    └── Workspaces (isolated environments)
        └── Agents (with specific configurations)
            └── Tools (RAG, HTTP API, etc.)
                └── Knowledge Base (workspace-specific)

Each level inherits configuration from above but can override settings.
Agents in different workspaces have isolated knowledge bases and can 
have completely different behaviors, tools, and model settings.
    """)
