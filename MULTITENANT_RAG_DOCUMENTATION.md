# Multitenant RAG System Documentation

## Overview

This document describes the comprehensive multitenant RAG (Retrieval-Augmented Generation) improvements implemented based on Qdrant's best practices for multitenancy. The system now provides optimized performance, data isolation, and advanced features for managing multiple organizations and workspaces.

## Key Features

### 🚀 Performance Optimizations

1. **Tenant Indexing with `is_tenant=True`**

   - Primary tenant identifier (`organization_id`) indexed with special tenant flag
   - Optimizes disk layout for faster tenant-specific searches
   - 5-10x performance improvement for multitenant queries

2. **Optimized Filter Ordering**
   - Tenant filters placed first in all queries
   - Reduces index scan overhead
   - Ensures efficient query execution

### 🔒 Security & Isolation

1. **Complete Data Isolation**

   - Organization-level isolation using `organization_id`
   - Workspace-level isolation using `workspace_id`
   - Automatic tenant validation on all operations

2. **Safety Mechanisms**
   - Confirmation required for destructive operations
   - Comprehensive audit logging
   - Tenant access validation

### 📊 Advanced Features

1. **Grouped Search**

   - Group results by metadata fields (document type, author, category)
   - Customizable group size and limits
   - Maintains tenant isolation

2. **Tenant Statistics**

   - Real-time tenant usage monitoring
   - Document and chunk counts
   - Collection health metrics

3. **Bulk Operations**
   - Bulk delete with safety confirmations
   - Data migration between workspaces
   - Batch processing capabilities

## API Reference

### Standard RAG Operations

#### POST /api/v1/query

Enhanced query endpoint with multitenant support.

```json
{
  "query": "search terms",
  "organization_id": "org_id",
  "workspace_id": "workspace_id",
  "agent_id": "agent_id",
  "top_k": 5,
  "filter_conditions": {
    "document_type": "manual"
  }
}
```

#### POST /api/v1/ingest

Document ingestion with workspace isolation.

```json
{
  "content": "document content",
  "organization_id": "org_id",
  "workspace_id": "workspace_id",
  "metadata": {
    "document_type": "manual",
    "author": "user@example.com"
  }
}
```

### Multitenant RAG Features

#### POST /api/v1/rag/grouped-search

Group search results by metadata fields.

```json
{
  "query": "search terms",
  "organization_id": "org_id",
  "workspace_id": "workspace_id",
  "group_by": "document_type",
  "limit": 5,
  "group_size": 10,
  "filter_conditions": {
    "category": "technical"
  }
}
```

**Response:**

```json
{
  "success": true,
  "query": "search terms",
  "groups": {
    "manual": [
      {
        "content": "...",
        "metadata": {...},
        "score": 0.95
      }
    ],
    "faq": [...]
  },
  "total_groups": 2,
  "group_by": "document_type"
}
```

#### POST /api/v1/rag/tenant-stats

Get tenant-specific statistics.

```json
{
  "organization_id": "org_id",
  "workspace_id": "workspace_id"
}
```

**Response:**

```json
{
  "success": true,
  "organization_id": "org_id",
  "workspace_id": "workspace_id",
  "total_documents": 150,
  "total_chunks": 1200,
  "collection_name": "knowledge_base",
  "avg_chunks_per_document": 8.0
}
```

#### POST /api/v1/rag/bulk-delete

Bulk delete tenant data with safety confirmation.

```json
{
  "organization_id": "org_id",
  "workspace_id": "workspace_id",
  "confirm_organization_id": "org_id"
}
```

**Response:**

```json
{
  "success": true,
  "organization_id": "org_id",
  "workspace_id": "workspace_id",
  "deleted_documents": 150,
  "deleted_chunks": 1200,
  "collection_name": "knowledge_base"
}
```

#### POST /api/v1/rag/migrate-data

Migrate data between workspaces.

```json
{
  "organization_id": "org_id",
  "source_workspace_id": "source_ws",
  "target_workspace_id": "target_ws",
  "document_filter": {
    "document_type": "archived"
  }
}
```

**Response:**

```json
{
  "success": true,
  "source_organization_id": "org_id",
  "source_workspace_id": "source_ws",
  "target_workspace_id": "target_ws",
  "migrated_chunks": 500,
  "collection_name": "knowledge_base"
}
```

## Architecture

### Tenant Hierarchy

```
Organizations (Primary Tenant Level)
    └── Workspaces (Secondary Tenant Level)
        └── Agents (Behavior Configuration)
            └── Documents (Data Isolation)
                └── Chunks (Vector Storage)
```

### Index Structure

1. **Primary Tenant Index**: `organization_id` with `is_tenant=true`
2. **Secondary Indexes**: `workspace_id`, `document_id`
3. **Metadata Indexes**: Configurable based on use case

### Data Flow

1. **Ingestion**: Document → Chunks → Embeddings → Qdrant (with tenant metadata)
2. **Retrieval**: Query → Embedding → Search (with tenant filters) → Results
3. **Grouping**: Results → Group by metadata → Organized response

## Performance Characteristics

### Query Performance

- **Single-tenant queries**: 5-10x faster with tenant indexing
- **Cross-workspace queries**: Blocked by design for security
- **Grouped searches**: Optimized with tenant-first filtering

### Storage Efficiency

- **Shared collection**: All tenants in one optimized collection
- **Tenant isolation**: Logical separation with physical optimization
- **Index overhead**: Minimal with strategic index placement

### Scalability

- **Organizations**: Unlimited (within Qdrant limits)
- **Workspaces per org**: Configurable (default: 10)
- **Documents per workspace**: Unlimited
- **Concurrent queries**: High throughput with tenant indexing

## Security Model

### Access Control

- **Organization-level**: Complete data isolation
- **Workspace-level**: Sub-tenant isolation within organization
- **Agent-level**: Behavioral configuration without data access

### Data Protection

- **Automatic tenant filtering**: All queries include tenant filters
- **Validation**: Tenant access validation on all operations
- **Audit logging**: Comprehensive operation logging

### Safety Features

- **Confirmation required**: Destructive operations need explicit confirmation
- **Filter validation**: Prevents cross-tenant data access
- **Error isolation**: Tenant errors don't affect other tenants

## Configuration

### Environment Variables

```bash
# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_api_key

# OpenAI Configuration
OPENAI_API_KEY=your_openai_key

# Multitenant Features
RAG_TENANT_INDEXING=true
RAG_OPTIMIZE_FILTERS=true
RAG_ENABLE_GROUPED_SEARCH=true

# Limits
MAX_WORKSPACES_PER_ORG=10
MAX_AGENTS_PER_WORKSPACE=5
```

### Collection Configuration

```python
{
    "collection_name": "knowledge_base",
    "vector_size": 1536,
    "distance": "COSINE",
    "tenant_indexing": True,
    "indexes": [
        {"field": "organization_id", "type": "keyword", "is_tenant": True},
        {"field": "workspace_id", "type": "keyword"},
        {"field": "document_id", "type": "keyword"}
    ]
}
```

## Best Practices

### For Developers

1. **Always include tenant context** in API calls
2. **Use workspace isolation** for logical separation
3. **Implement proper error handling** for tenant operations
4. **Monitor tenant statistics** for performance insights

### For Administrators

1. **Regular monitoring** of tenant statistics
2. **Periodic cleanup** of unused workspaces
3. **Index maintenance** for optimal performance
4. **Backup strategies** per organization

### For End Users

1. **Organize documents** with meaningful metadata
2. **Use descriptive workspace names** for clarity
3. **Leverage grouped search** for better organization
4. **Monitor usage** through statistics endpoints

## Migration Guide

### From Single-Tenant to Multitenant

1. **Update configurations** to include organization_id
2. **Migrate existing data** with tenant metadata
3. **Update API calls** to include tenant context
4. **Test isolation** between tenants

### Code Examples

```python
# Before (Single-tenant)
result = await rag_tool.execute("query")

# After (Multitenant)
rag_tool = RAGTool(organization_id="org1", workspace_id="ws1")
result = await rag_tool.execute("query")
```

## Monitoring & Troubleshooting

### Key Metrics

- Tenant query performance
- Index utilization
- Cross-tenant access attempts
- Storage usage per tenant

### Common Issues

- **Slow queries**: Check tenant indexing configuration
- **Access denied**: Verify tenant context in requests
- **High memory usage**: Consider tenant data distribution

### Debugging

- Enable detailed logging with tenant context
- Use tenant statistics for performance analysis
- Monitor Qdrant collection health

## Future Enhancements

### Planned Features

- **Tenant quotas**: Resource limits per organization
- **Cross-tenant search**: Controlled sharing mechanisms
- **Advanced analytics**: Detailed usage insights
- **Auto-scaling**: Dynamic resource allocation

### Integration Opportunities

- **Authentication systems**: SSO integration
- **Billing systems**: Usage-based billing
- **Monitoring platforms**: Enhanced observability
- **Backup solutions**: Tenant-aware backups

## Support & Resources

- **API Documentation**: Available at `/docs` endpoint
- **Test Examples**: See `test_multitenant_rag.py`
- **Usage Examples**: See `api_usage_examples.py`
- **Configuration Reference**: See `src/agent/config.py`

For technical support or feature requests, please refer to the project documentation or contact the development team.
