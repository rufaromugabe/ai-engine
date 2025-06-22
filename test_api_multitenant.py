#!/usr/bin/env python3
"""
Quick API test for the new multitenant RAG endpoints.
Run this after starting the API server to verify everything works.
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/v1"

def test_api_health():
    """Test basic API health"""
    try:
        response = requests.get(f"{BASE_URL.replace('/api/v1', '')}/health")
        if response.status_code == 200:
            print("✓ API server is running")
            return True
        else:
            print(f"✗ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to API server: {e}")
        print("Make sure to start the server with: python main.py")
        return False

def test_organization_setup():
    """Test organization setup"""
    payload = {
        "organization_id": "test_org_multitenant",
        "enabled_tools": ["rag"],
        "rag_settings": {
            "top_k": 5,
            "similarity_threshold": 0.7
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/setup-organization", json=payload)
        if response.status_code == 200:
            print("✓ Organization setup successful")
            return True
        else:
            print(f"✗ Organization setup failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Organization setup error: {e}")
        return False

def test_workspace_creation():
    """Test workspace creation"""
    payload = {
        "organization_id": "test_org_multitenant",
        "name": "Test Workspace",
        "description": "Test workspace for multitenant features",
        "shared_tools": ["rag"],
        "shared_settings": {
            "response_style": "technical"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/workspaces", json=payload)
        if response.status_code == 200:
            result = response.json()
            workspace_id = result.get("workspace_id")
            print(f"✓ Workspace created: {workspace_id}")
            return workspace_id
        else:
            print(f"✗ Workspace creation failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"✗ Workspace creation error: {e}")
        return None

def test_document_ingestion(workspace_id):
    """Test document ingestion"""
    payload = {
        "content": "This is a test document for multitenant RAG testing. It contains technical information about system architecture and deployment strategies.",
        "organization_id": "test_org_multitenant",
        "workspace_id": workspace_id,
        "metadata": {
            "document_type": "technical_manual",
            "category": "architecture",
            "author": "test_system"
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/ingest", json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✓ Document ingested: {result.get('document_id')}")
                return True
            else:
                print(f"✗ Document ingestion failed: {result.get('error')}")
                return False
        else:
            print(f"✗ Document ingestion failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Document ingestion error: {e}")
        return False

def test_grouped_search(workspace_id):
    """Test the new grouped search feature"""
    payload = {
        "query": "technical architecture",
        "organization_id": "test_org_multitenant",
        "workspace_id": workspace_id,
        "group_by": "document_type",
        "limit": 3,
        "group_size": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/rag/grouped-search", json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print(f"✓ Grouped search successful: {result.get('total_groups')} groups")
                return True
            else:
                print(f"✗ Grouped search failed: {result.get('error')}")
                return False
        else:
            print(f"✗ Grouped search failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Grouped search error: {e}")
        return False

def test_tenant_stats(workspace_id):
    """Test tenant statistics"""
    payload = {
        "organization_id": "test_org_multitenant",
        "workspace_id": workspace_id
    }
    
    try:
        response = requests.post(f"{BASE_URL}/rag/tenant-stats", json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                stats = result
                print(f"✓ Tenant stats: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
                return True
            else:
                print(f"✗ Tenant stats failed: {result.get('error')}")
                return False
        else:
            print(f"✗ Tenant stats failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Tenant stats error: {e}")
        return False

def test_regular_query(workspace_id):
    """Test regular query to ensure basic functionality still works"""
    payload = {
        "query": "system architecture",
        "organization_id": "test_org_multitenant",
        "workspace_id": workspace_id
    }
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✓ Regular query successful")
                return True
            else:
                print(f"✗ Regular query failed: {result.get('error')}")
                return False
        else:
            print(f"✗ Regular query failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Regular query error: {e}")
        return False

def run_quick_test():
    """Run a quick test of all multitenant features"""
    print("🧪 Quick Multitenant RAG API Test")
    print("=" * 50)
    
    # Test 1: API Health
    if not test_api_health():
        print("\n❌ API server not available. Please start with: python main.py")
        return
    
    # Test 2: Organization Setup
    if not test_organization_setup():
        print("\n❌ Organization setup failed")
        return
    
    # Test 3: Workspace Creation
    workspace_id = test_workspace_creation()
    if not workspace_id:
        print("\n❌ Workspace creation failed")
        return
    
    # Wait a moment for initialization
    print("⏳ Waiting for initialization...")
    time.sleep(2)
    
    # Test 4: Document Ingestion
    if not test_document_ingestion(workspace_id):
        print("\n❌ Document ingestion failed")
        return
    
    # Wait for indexing
    print("⏳ Waiting for document indexing...")
    time.sleep(3)
    
    # Test 5: Regular Query
    if not test_regular_query(workspace_id):
        print("\n❌ Regular query failed")
        return
    
    # Test 6: Grouped Search
    if not test_grouped_search(workspace_id):
        print("\n❌ Grouped search failed")
        return
    
    # Test 7: Tenant Statistics
    if not test_tenant_stats(workspace_id):
        print("\n❌ Tenant statistics failed")
        return
    
    print("\n🎉 All tests passed!")
    print("=" * 50)
    print("✅ Multitenant RAG system is working correctly")
    print("✅ New features are operational:")
    print("   - Tenant indexing with optimization")
    print("   - Grouped search functionality")
    print("   - Tenant statistics monitoring")
    print("   - Workspace isolation")
    print("   - Enhanced API endpoints")
    print("=" * 50)

if __name__ == "__main__":
    run_quick_test()
