"""Test script for the refactored sharded RAG system."""

import asyncio
import logging
import os
from typing import Dict, Any
from dotenv import load_dotenv

from src.agent.config import config_manager, OrganizationConfig, ToolType, RAGConfig
from src.agent.tools.rag_tool import RAGTool

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_sharded_rag_system():
    """Test the new sharded RAG system with multiple organizations and workspaces."""
    
    # Create test organizations
    org1_id = "test_org_1"
    org2_id = "test_org_2"
    
    # Create test workspaces
    workspace1_id = "workspace_alpha"
    workspace2_id = "workspace_beta"
    
    print("🚀 Testing Sharded RAG System")
    print("=" * 50)
    
    try:
        # Set up organization configs
        for org_id in [org1_id, org2_id]:
            org_config = OrganizationConfig(
                organization_id=org_id,
                enabled_tools=[ToolType.RAG]
            )
            config_manager.register_organization(org_config)
            print(f"✅ Registered organization: {org_id}")
        
        # Create RAG tools for different org/workspace combinations
        rag_tools = {
            "org1_ws1": RAGTool(org1_id, workspace1_id),
            "org1_ws2": RAGTool(org1_id, workspace2_id),
            "org2_ws1": RAGTool(org2_id, workspace1_id),
            "org2_ws2": RAGTool(org2_id, workspace2_id),
        }
        
        # Initialize all tools
        for name, tool in rag_tools.items():
            success = await tool.initialize()
            if success:
                print(f"✅ Initialized RAG tool: {name}")
            else:
                print(f"❌ Failed to initialize RAG tool: {name}")
                return
        
        print("\n📝 Testing Document Ingestion")
        print("-" * 30)
        
        # Test documents for each workspace
        test_documents = {
            "org1_ws1": "This is confidential data for Organization 1, Workspace Alpha. It contains sensitive financial information.",
            "org1_ws2": "This is technical documentation for Organization 1, Workspace Beta. It includes API specifications.",
            "org2_ws1": "This is marketing content for Organization 2, Workspace Alpha. It contains product descriptions.",
            "org2_ws2": "This is HR policy documentation for Organization 2, Workspace Beta. It includes employee guidelines.",
        }
        
        # Ingest documents
        for tool_name, content in test_documents.items():
            tool = rag_tools[tool_name]
            result = await tool.ingest_document(
                content=content,
                metadata={"document_type": "test", "tool_name": tool_name},
                document_id=f"doc_{tool_name}"
            )
            
            if result.success:
                print(f"✅ Ingested document for {tool_name}")
            else:
                print(f"❌ Failed to ingest document for {tool_name}: {result.error}")
        
        print("\n🔍 Testing Cross-Workspace Data Isolation")
        print("-" * 40)
        
        # Test that each workspace only retrieves its own data
        test_queries = [
            ("financial information", "org1_ws1"),
            ("API specifications", "org1_ws2"), 
            ("product descriptions", "org2_ws1"),
            ("employee guidelines", "org2_ws2"),
        ]
        
        for query, expected_tool in test_queries:
            print(f"\nTesting query: '{query}'")
            
            for tool_name, tool in rag_tools.items():
                result = await tool.execute(query, top_k=3)
                
                if result.success and result.data["results"]:
                    found_relevant = any(
                        query.lower() in res["content"].lower() 
                        for res in result.data["results"]
                    )
                    
                    if tool_name == expected_tool and found_relevant:
                        print(f"  ✅ {tool_name}: Found expected results")
                    elif tool_name != expected_tool and found_relevant:
                        print(f"  ⚠️  {tool_name}: Found results (potential data leak!)")
                    else:
                        print(f"  ✅ {tool_name}: No results (correct isolation)")
                else:
                    print(f"  ✅ {tool_name}: No results (correct isolation)")
        
        print("\n📊 Testing Collection Information")
        print("-" * 30)
        
        # Get collection info from each tool
        for tool_name, tool in rag_tools.items():
            result = await tool.get_collection_info()
            if result.success:
                data = result.data
                print(f"{tool_name}:")
                print(f"  Collection: {data['collection_name']}")
                print(f"  Org: {data['organization_id']}")
                print(f"  Workspace: {data['workspace_id']}")
                print(f"  Workspace Documents: {data['workspace_documents']}")
        
        print("\n🧹 Testing Document Deletion")
        print("-" * 25)
        
        # Test deletion (should only delete from specific workspace)
        tool = rag_tools["org1_ws1"]
        result = await tool.delete_document("doc_org1_ws1")
        
        if result.success:
            print("✅ Document deletion successful")
            
            # Verify deletion
            search_result = await tool.execute("financial information", top_k=5)
            if search_result.success and not search_result.data["results"]:
                print("✅ Document successfully deleted from workspace")
            else:
                print("⚠️  Document may still exist in workspace")
        else:
            print(f"❌ Document deletion failed: {result.error}")
        
        print("\n🎉 Sharded RAG System Test Complete!")
        
        # Cleanup
        for tool in rag_tools.values():
            await tool.cleanup()
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_sharded_rag_system())
