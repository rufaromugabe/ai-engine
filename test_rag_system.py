"""Utility script for testing and managing the RAG-enabled AI agent system."""

import asyncio
import logging
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from agent.config import config_manager, OrganizationConfig, ToolType, RAGConfig
from agent.tool_manager import tool_manager
from agent.enhanced_graph import enhanced_graph as graph, ingest_document_for_organization, get_organization_knowledge_base_info

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgentManager:
    """Utility class for managing the RAG agent system."""
    
    def __init__(self):
        self.graph = graph
    
    async def setup_organization(
        self, 
        organization_id: str, 
        enabled_tools: Optional[list] = None,
        rag_config_overrides: Optional[Dict] = None
    ) -> bool:
        """Set up a new organization with custom configuration."""
        try:
            # Create organization configuration
            org_config = OrganizationConfig(
                organization_id=organization_id,
                enabled_tools=enabled_tools or [ToolType.RAG]
            )
            
            # Apply RAG config overrides if provided
            if rag_config_overrides:
                for key, value in rag_config_overrides.items():
                    if hasattr(org_config.rag_config, key):
                        setattr(org_config.rag_config, key, value)
            
            # Register the organization
            config_manager.register_organization(org_config)
            
            # Initialize tools
            success = await tool_manager.initialize_tools_for_organization(organization_id)
            
            if success:
                logger.info(f"Successfully set up organization: {organization_id}")
                return True
            else:
                logger.error(f"Failed to initialize tools for organization: {organization_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error setting up organization {organization_id}: {str(e)}")
            return False
    
    async def ingest_sample_documents(self, organization_id: str) -> bool:
        """Ingest sample documents for testing."""
        try:
            sample_docs = [
                {
                    "content": """
                    LangGraph is a library for building stateful, multi-actor applications with LLMs.
                    It extends LangChain with the ability to coordinate multiple chains (or actors) 
                    across multiple steps of computation in a cyclic manner. The key features include:
                    
                    1. Cycles and Branching: Unlike DAGs, LangGraph supports cycles and conditional edges
                    2. Persistence: LangGraph has built-in persistence for long-running applications
                    3. Human-in-the-loop: Easy integration for human feedback and approval
                    4. Streaming Support: Support for streaming intermediate steps
                    """,
                    "metadata": {"type": "documentation", "topic": "langgraph", "source": "official_docs"},
                    "document_id": "langgraph_intro"
                },
                {
                    "content": """
                    Qdrant is a vector similarity search engine that provides a convenient API 
                    for storing, searching, and managing points—vectors with an additional payload.
                    Key features include:
                    
                    1. Fast and accurate vector search
                    2. Advanced filtering capabilities
                    3. Rich data types support
                    4. Distributed deployment options
                    5. REST API and gRPC interfaces
                    """,
                    "metadata": {"type": "documentation", "topic": "qdrant", "source": "official_docs"},
                    "document_id": "qdrant_intro"
                },
                {
                    "content": """
                    OpenAI's GPT models are large language models trained on diverse internet text.
                    The models can be used for various tasks including:
                    
                    1. Text generation and completion
                    2. Question answering
                    3. Summarization
                    4. Translation
                    5. Code generation
                    
                    The API provides access to different model variants optimized for different use cases.
                    """,
                    "metadata": {"type": "documentation", "topic": "openai", "source": "official_docs"},
                    "document_id": "openai_intro"
                }
            ]
            
            for doc in sample_docs:
                result = await ingest_document_for_organization(
                    organization_id=organization_id,
                    content=doc["content"],
                    metadata=doc["metadata"],
                    document_id=doc["document_id"]
                )
                
                if result["success"]:
                    logger.info(f"Ingested document: {doc['document_id']}")
                else:
                    logger.error(f"Failed to ingest document {doc['document_id']}: {result['error']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error ingesting sample documents: {str(e)}")
            return False
    
    async def test_query(self, organization_id: str, query: str) -> Dict[str, Any]:
        """Test a query against the RAG system."""
        try:
            # Prepare state
            from langchain_core.messages import HumanMessage
            
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "organization_id": organization_id,
                "current_tool": None,
                "tool_results": [],
                "user_query": query,
                "context": {}
            }
            
            # Configure the graph
            config = {
                "configurable": {
                    "organization_id": organization_id
                }
            }
            
            # Run the graph
            result = await self.graph.ainvoke(initial_state, config)
            
            # Extract the final response
            final_message = result["messages"][-1]
            
            return {
                "success": True,
                "query": query,
                "response": final_message.content,
                "tool_results": result.get("tool_results", []),
                "organization_id": organization_id
            }
            
        except Exception as e:
            logger.error(f"Error testing query: {str(e)}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "organization_id": organization_id
            }
    
    async def get_knowledge_base_status(self, organization_id: str) -> Dict[str, Any]:
        """Get the status of the knowledge base for an organization."""
        return await get_organization_knowledge_base_info(organization_id)

async def main():
    """Main function for testing the RAG system."""
    manager = RAGAgentManager()
    
    # Test organization ID
    test_org = "test_company"
    
    print("🚀 Setting up RAG-enabled AI Agent System")
    print("=" * 50)
    
    # Step 1: Set up organization
    print(f"1. Setting up organization: {test_org}")
    success = await manager.setup_organization(test_org)
    if not success:
        print("❌ Failed to set up organization")
        return
    print("✅ Organization set up successfully")
    
    # Step 2: Ingest sample documents
    print("\n2. Ingesting sample documents...")
    success = await manager.ingest_sample_documents(test_org)
    if not success:
        print("❌ Failed to ingest sample documents")
        return
    print("✅ Sample documents ingested successfully")
    
    # Step 3: Check knowledge base status
    print("\n3. Checking knowledge base status...")
    kb_status = await manager.get_knowledge_base_status(test_org)
    if kb_status["success"]:
        print(f"✅ Knowledge base info: {kb_status['data']}")
    else:
        print(f"❌ Failed to get knowledge base info: {kb_status['error']}")
    
    # Step 4: Test queries
    test_queries = [
        "What is LangGraph?",
        "Tell me about Qdrant's features",
        "How can I use OpenAI models?",
        "What are the benefits of vector search?"
    ]
    
    print("\n4. Testing queries...")
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        result = await manager.test_query(test_org, query)
        
        if result["success"]:
            print(f"Response: {result['response']}")
            if result.get("tool_results"):
                print(f"Tool results: {len(result['tool_results'])} results")
        else:
            print(f"❌ Query failed: {result['error']}")
    
    print("\n🎉 RAG system testing completed!")

if __name__ == "__main__":
    asyncio.run(main())
