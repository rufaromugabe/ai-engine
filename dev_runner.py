#!/usr/bin/env python3
"""Development script for LangGraph agent testing and management."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent.graph import graph, test_query, get_graph_info, create_dev_config
from agent.enhanced_graph import execute_agent_query

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_query():
    """Test basic query functionality."""
    print("🧪 Testing basic query...")
    result = await test_query("What is LangGraph and how does it work?")
    print(f"✅ Result: {json.dumps(result, indent=2)}")
    return result

async def test_rag_query():
    """Test RAG functionality."""
    print("🔍 Testing RAG query...")
    result = await execute_agent_query(
        organization_id="dev-org",
        query="Search for information about artificial intelligence",
        workspace_id="dev-workspace",
        agent_id="dev-agent"
    )
    print(f"✅ RAG Result: {json.dumps(result, indent=2)}")
    return result

async def test_workspace_isolation():
    """Test workspace isolation."""
    print("🏢 Testing workspace isolation...")
    
    # Test with different workspaces
    workspace1_result = await execute_agent_query(
        organization_id="test-org",
        query="What tools are available?",
        workspace_id="workspace-1"
    )
    
    workspace2_result = await execute_agent_query(
        organization_id="test-org", 
        query="What tools are available?",
        workspace_id="workspace-2"
    )
    
    print(f"✅ Workspace 1: {json.dumps(workspace1_result, indent=2)}")
    print(f"✅ Workspace 2: {json.dumps(workspace2_result, indent=2)}")
    
    return workspace1_result, workspace2_result

def print_graph_info():
    """Print graph structure information."""
    print("📊 Graph Information:")
    info = get_graph_info()
    print(json.dumps(info, indent=2))

async def interactive_mode():
    """Interactive mode for testing queries."""
    print("🎮 Interactive Mode - Type 'quit' to exit")
    
    while True:
        query = input("\n💬 Enter your query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        try:
            result = await execute_agent_query(
                organization_id="dev-org",
                query=query,
                workspace_id="dev-workspace",
                agent_id="dev-agent"
            )
            
            print(f"🤖 Response: {result.get('response', 'No response')}")
            
            if result.get('tool_results'):
                print(f"🔧 Tool Results: {len(result['tool_results'])} results")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")

async def run_all_tests():
    """Run all available tests."""
    print("🚀 Running all tests...")
    
    print_graph_info()
    
    await test_basic_query()
    await test_rag_query()
    await test_workspace_isolation()
    
    print("✅ All tests completed!")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LangGraph Development Tool")
    parser.add_argument(
        "command",
        choices=["test", "interactive", "info", "basic", "rag", "workspace"],
        help="Command to run"
    )
    
    args = parser.parse_args()
    
    if args.command == "test":
        asyncio.run(run_all_tests())
    elif args.command == "interactive":
        asyncio.run(interactive_mode())
    elif args.command == "info":
        print_graph_info()
    elif args.command == "basic":
        asyncio.run(test_basic_query())
    elif args.command == "rag":
        asyncio.run(test_rag_query())
    elif args.command == "workspace":
        asyncio.run(test_workspace_isolation())

if __name__ == "__main__":
    main()
