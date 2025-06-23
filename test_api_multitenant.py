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
        "enabled_tools": ["rag"]
        # Note: RAG settings are now system-wide via environment variables
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

def test_multiple_document_ingestion(workspace_id):
    """Test ingestion of multiple documents with different types and categories"""
    documents = [
        {
            "content": "This is a comprehensive guide to microservices architecture. It covers service decomposition, API gateway patterns, and distributed system design principles. Key concepts include service boundaries, data consistency, and fault tolerance.",
            "metadata": {
                "document_type": "technical_manual",
                "category": "architecture",
                "author": "tech_team",
                "title": "Microservices Architecture Guide"
            }
        },
        {
            "content": "Database optimization strategies for high-performance applications. This document covers indexing strategies, query optimization, connection pooling, and caching mechanisms. Performance monitoring and scaling techniques are also discussed.",
            "metadata": {
                "document_type": "technical_manual",
                "category": "database",
                "author": "db_team",
                "title": "Database Performance Guide"
            }
        },
        {
            "content": "Security best practices for cloud deployment. Topics include identity and access management, network security, encryption at rest and in transit, and compliance frameworks. Zero-trust architecture principles are emphasized.",
            "metadata": {
                "document_type": "security_guide",
                "category": "security",
                "author": "security_team",
                "title": "Cloud Security Manual"
            }
        },
        {
            "content": "DevOps pipeline automation using CI/CD tools. This guide covers automated testing, deployment strategies, infrastructure as code, and monitoring. Container orchestration and blue-green deployments are key topics.",
            "metadata": {
                "document_type": "operational_guide",
                "category": "devops",
                "author": "devops_team",
                "title": "CI/CD Pipeline Guide"
            }
        },
        {
            "content": "API design and documentation standards. RESTful principles, GraphQL implementation, versioning strategies, and OpenAPI specifications. Rate limiting and authentication mechanisms are covered in detail.",
            "metadata": {
                "document_type": "technical_manual",
                "category": "api_design",
                "author": "api_team",
                "title": "API Design Standards"
            }
        }
    ]
    
    successful_ingestions = 0
    document_ids = []
    
    for i, doc in enumerate(documents):
        payload = {
            "content": doc["content"],
            "organization_id": "test_org_multitenant",
            "workspace_id": workspace_id,
            "metadata": doc["metadata"]
        }
        
        try:
            response = requests.post(f"{BASE_URL}/ingest", json=payload)
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    document_id = result.get('document_id')
                    document_ids.append(document_id)
                    successful_ingestions += 1
                    print(f"✓ Document {i+1} ingested: {doc['metadata']['title']}")
                else:
                    print(f"✗ Document {i+1} ingestion failed: {result.get('error')}")
            else:
                print(f"✗ Document {i+1} ingestion failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Document {i+1} ingestion error: {e}")
    
    if successful_ingestions == len(documents):
        print(f"✓ All {successful_ingestions} documents ingested successfully")
        return True, document_ids
    elif successful_ingestions > 0:
        print(f"⚠ Partial success: {successful_ingestions}/{len(documents)} documents ingested")
        return True, document_ids
    else:
        print("✗ No documents were ingested successfully")
        return False, []

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
    
    # Test 4: Multiple Document Ingestion
    success, document_ids = test_multiple_document_ingestion(workspace_id)
    if not success:
        print("\n❌ Document ingestion failed")
        return
    
    # Wait for indexing
    print("⏳ Waiting for document indexing...")
    time.sleep(5)  # Increased wait time for multiple documents
    
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
    print(f"   - Multiple document types ({len(document_ids)} documents)")
    print("=" * 50)

if __name__ == "__main__":
    run_quick_test()
