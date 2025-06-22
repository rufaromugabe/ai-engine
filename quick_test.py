#!/usr/bin/env python3

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test all imports."""
    try:
        print("Testing config import...")
        from agent.config import config_manager
        print("✅ Config import successful")
        
        print("Testing tool manager import...")
        from agent.tool_manager import tool_manager
        print("✅ Tool manager import successful")
        
        print("Testing RAG tool import...")
        from agent.tools.rag_tool import RAGTool
        print("✅ RAG tool import successful")
        
        print("Testing base tool import...")
        from agent.tools.base_tool import BaseTool, ToolResult
        print("✅ Base tool import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration system."""
    try:
        from agent.config import config_manager, OrganizationConfig, ToolType
        
        # Create test config
        test_config = OrganizationConfig(
            organization_id="test_import",
            enabled_tools=[ToolType.RAG]
        )
        
        config_manager.register_organization(test_config)
        retrieved = config_manager.get_organization_config("test_import")
        
        assert retrieved.organization_id == "test_import"
        print("✅ Configuration system working")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick Import Test")
    print("=" * 30)
    
    imports_ok = test_imports()
    config_ok = test_config()
    
    if imports_ok and config_ok:
        print("\n✅ All basic tests passed!")
    else:
        print("\n❌ Some tests failed")
