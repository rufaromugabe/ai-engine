#!/usr/bin/env python3
"""Simple test script to verify RAGTool import"""

try:
    from src.agent.tools.rag_tool import RAGTool
    print("✅ RAGTool import successful!")
    print("✅ All syntax issues have been resolved!")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
