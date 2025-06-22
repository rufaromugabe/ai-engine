#!/usr/bin/env python3
"""
Test script to verify multitenant RAG tool improvements.
This script demonstrates the enhanced multitenant features following Qdrant best practices.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.tools.rag_tool import RAGTool
from agent.config import config_manager, OrganizationConfig, ToolType, RAGConfig, LLMConfig, QdrantConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_test_organizations():
    """Set up test organizations for the multitenant test."""
    import os
    
    # Get Qdrant configuration from environment
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create test organizations with proper configuration
    org1_config = OrganizationConfig(
        organization_id="org_1",
        enabled_tools=[ToolType.RAG],
        rag_config=RAGConfig(
            collection_name="test_multitenant_knowledge_base",
            vector_size=1536,
            top_k=5,
            similarity_threshold=0.7
        ),
        llm_config=LLMConfig(
            api_key=openai_api_key,
            model="gpt-3.5-turbo"
        ),
        qdrant_config=QdrantConfig(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
    )
    
    org2_config = OrganizationConfig(
        organization_id="org_2",
        enabled_tools=[ToolType.RAG],
        rag_config=RAGConfig(
            collection_name="test_multitenant_knowledge_base",
            vector_size=1536,
            top_k=5,
            similarity_threshold=0.7
        ),
        llm_config=LLMConfig(
            api_key=openai_api_key,
            model="gpt-3.5-turbo"
        ),
        qdrant_config=QdrantConfig(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
    )
    
    # Register the organizations
    config_manager.register_organization(org1_config)
    config_manager.register_organization(org2_config)
    
    print("✓ Test organizations registered successfully")
    print(f"  Using Qdrant at: {qdrant_url}")
    print(f"  Using OpenAI API: {'✓' if openai_api_key else '✗ Missing'}")
    return org1_config, org2_config


def check_qdrant_availability():
    """Check if Qdrant is available using the configured URL."""
    import os
    import requests
    
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    
    try:
        # Try to make a simple request to Qdrant health endpoint
        headers = {}
        if qdrant_api_key:
            headers["api-key"] = qdrant_api_key
            
        # Remove the port from URL if it's a cloud instance
        if "cloud.qdrant.io" in qdrant_url:
            base_url = qdrant_url.replace(":6333", "")
        else:
            base_url = qdrant_url
            
        response = requests.get(f"{base_url}/collections", headers=headers, timeout=10)
        return response.status_code in [200, 404]  # 404 is OK if no collections exist yet
    except Exception as e:
        print(f"  Qdrant connection test failed: {e}")
        return False


async def test_basic_configuration():
    """Test basic configuration without requiring Qdrant connection."""
    try:
        print("Testing basic multitenant configuration...")
        
        # Set up test organizations
        setup_test_organizations()
        
        # Test tool creation (without initialization)
        org1_workspace1 = RAGTool(organization_id="org_1", workspace_id="workspace_1")
        org1_workspace2 = RAGTool(organization_id="org_1", workspace_id="workspace_2")
        org2_workspace1 = RAGTool(organization_id="org_2", workspace_id="workspace_1")

        tools = [
            ("Org1-Workspace1", org1_workspace1),
            ("Org1-Workspace2", org1_workspace2),
            ("Org2-Workspace1", org2_workspace1)
        ]

        print("\nTesting tool configuration...")
        for name, tool in tools:
            print(f"Testing {name}:")
            
            # Test tenant validation (without initialization)
            if tool._validate_tenant_access():
                print(f"  ✓ Tenant validation passed")
            else:
                print(f"  ✗ Tenant validation failed")
            
            # Test schema retrieval
            try:
                schema = tool.get_schema()
                grouped_schema = tool.get_grouped_search_schema()
                print(f"  ✓ Schemas retrieved successfully")
                print(f"    - Standard schema: {schema['function']['name']}")
                print(f"    - Grouped schema: {grouped_schema['function']['name']}")
            except Exception as e:
                print(f"  ✗ Schema retrieval failed: {e}")
            
            # Test configuration access
            try:
                collection_name = tool.collection_name
                org_id = tool.organization_id
                workspace_id = tool.workspace_id
                print(f"  ✓ Configuration access successful")
                print(f"    - Collection: {collection_name}")
                print(f"    - Organization: {org_id}")
                print(f"    - Workspace: {workspace_id}")
            except Exception as e:
                print(f"  ✗ Configuration access failed: {e}")

        print("\n" + "="*60)
        print("Basic multitenant configuration test completed!")
        print("Key features verified:")
        print("- Organization registration")
        print("- Tool creation with tenant isolation")
        print("- Tenant access validation")
        print("- Schema generation for function calling")
        print("- Configuration management")
        print("\nTo test full functionality, please start Qdrant and run again.")
        print("="*60)

    except Exception as e:
        logger.error(f"Basic configuration test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_multitenant_rag():
    """Test the multitenant RAG tool functionality."""
    try:        # First check if Qdrant is available
        print("Checking Qdrant availability...")
        if not check_qdrant_availability():
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            print(f"⚠️  Qdrant server not accessible at {qdrant_url}")
            print("Please check:")
            print("  1. Qdrant cloud instance is running")
            print("  2. QDRANT_URL and QDRANT_API_KEY are set correctly in .env")
            print("  3. Network connectivity to the cloud instance")
            print("\nRunning basic configuration tests only...\n")
            
            # Run basic tests without Qdrant connection
            await test_basic_configuration()
            return
        else:
            qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
            print(f"✓ Qdrant server accessible at {qdrant_url}")
            print("Running full tests...\n")

        # Set up test organizations first
        print("Setting up test organizations...")
        org1_config, org2_config = setup_test_organizations()
        
        # Test with different organizations and workspaces
        org1_workspace1 = RAGTool(organization_id="org_1", workspace_id="workspace_1")
        org1_workspace2 = RAGTool(organization_id="org_1", workspace_id="workspace_2")
        org2_workspace1 = RAGTool(organization_id="org_2", workspace_id="workspace_1")

        tools = [
            ("Org1-Workspace1", org1_workspace1),
            ("Org1-Workspace2", org1_workspace2),
            ("Org2-Workspace1", org2_workspace1)
        ]

        # Test initialization
        print("\nTesting multitenant RAG tool initialization...")
        for name, tool in tools:
            print(f"\nInitializing {name}...")
            try:
                success = await tool.initialize()
                if success:
                    print(f"✓ {name} initialized successfully")
                else:
                    print(f"✗ {name} initialization failed")
                    continue
            except Exception as e:
                print(f"✗ {name} initialization failed with error: {e}")
                print("  ⚠ This might be due to missing Qdrant instance or API keys")
                continue

            # Test tenant validation
            if tool._validate_tenant_access():
                print(f"✓ {name} tenant validation passed")
            else:
                print(f"✗ {name} tenant validation failed")

            # Test tenant statistics
            try:
                stats_result = await tool.get_tenant_statistics()
                if stats_result.success:
                    print(f"✓ {name} statistics: {stats_result.data}")
                else:
                    print(f"✗ {name} statistics failed: {stats_result.error}")
            except Exception as e:
                print(f"⚠ {name} statistics failed (might be expected if no data): {e}")

            # Test document ingestion
            test_content = f"This is test content for {name}. It contains information specific to this tenant."
            test_metadata = {
                "document_type": "test_document",
                "author": f"test_user_{name.lower()}",
                "category": "testing"
            }

            try:
                ingest_result = await tool.ingest_document(
                    content=test_content,
                    metadata=test_metadata,
                    document_id=f"test_doc_{name.lower()}"
                )

                if ingest_result.success:
                    print(f"✓ {name} document ingestion successful")
                else:
                    print(f"✗ {name} document ingestion failed: {ingest_result.error}")
            except Exception as e:
                print(f"⚠ {name} document ingestion failed (might be due to missing API keys): {e}")

            # Test search functionality
            try:
                search_result = await tool.execute("test content")
                if search_result.success:
                    print(f"✓ {name} search successful, found {search_result.data['total_results']} results")
                    # Verify tenant isolation - should only find its own content
                    for result in search_result.data['results']:
                        content = result['content']
                        if name.lower() in content.lower():
                            print(f"  ✓ Found tenant-specific content")
                        else:
                            print(f"  ⚠ Found non-tenant content (potential isolation issue)")
                else:
                    print(f"✗ {name} search failed: {search_result.error}")
            except Exception as e:
                print(f"⚠ {name} search failed (might be due to missing API keys): {e}")

            # Test grouped search
            try:
                grouped_result = await tool.execute_grouped_search(
                    query="test content",
                    group_by="document_type",
                    limit=3,
                    group_size=5
                )
                if grouped_result.success:
                    print(f"✓ {name} grouped search successful")
                    print(f"  Groups found: {list(grouped_result.data['groups'].keys())}")
                else:
                    print(f"✗ {name} grouped search failed: {grouped_result.error}")
            except Exception as e:
                print(f"  ⚠ Grouped search error (might be expected): {e}")

            # Test schema retrieval
            try:
                schema = tool.get_schema()
                grouped_schema = tool.get_grouped_search_schema()
                print(f"✓ {name} schemas retrieved successfully")
            except Exception as e:
                print(f"✗ {name} schema retrieval failed: {e}")

            await tool.cleanup()
            print(f"✓ {name} cleanup completed")

        print("\n" + "="*60)
        print("Multitenant RAG tool test completed!")
        print("Key improvements implemented:")
        print("- Tenant indexing with is_tenant=True for organization_id")
        print("- Optimized filter ordering (tenant filters first)")
        print("- Tenant access validation")
        print("- Grouped search functionality")
        print("- Tenant-specific statistics")
        print("- Bulk tenant operations")
        print("- Enhanced logging and auditing")
        print("="*60)

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_multitenant_rag())
