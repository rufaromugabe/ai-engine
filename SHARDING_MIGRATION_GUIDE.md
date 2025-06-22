# Migration Guide: Single Collection with Custom Sharding

This document provides a comprehensive guide for migrating from the old multi-collection approach to the new single collection with custom sharding approach in your RAG system.

## Overview of Changes

### Before (Multi-Collection Approach)
- **Collection per Workspace**: Each workspace had its own collection (e.g., `default_collection_org1_wsA`)
- **Hard Isolation**: Complete separation at the collection level
- **Management Overhead**: Thousands of collections to manage
- **Resource Inefficiency**: Each collection had its own resource allocation

### After (Single Collection with Sharding)
- **Single Global Collection**: One collection for all organizations (e.g., `ai_engine_knowledge_base`)
- **Organization-Level Sharding**: Each organization gets its own shard (hard isolation)
- **Workspace-Level Filtering**: Workspaces within an organization use metadata filtering (soft isolation)
- **Improved Scalability**: Better resource management and performance

## Key Technical Changes

### 1. Collection Creation
```python
# OLD: Multiple collections
collection_name = f"{config.collection_name}_{organization_id}_{workspace_id}"

# NEW: Single global collection with sharding
collection_name = config.collection_name  # e.g., "ai_engine_knowledge_base"
sharding_method = ShardingMethod.CUSTOM
shard_number = int(os.getenv("QDRANT_SHARD_NUMBER", "4"))
```

### 2. Document Ingestion
```python
# OLD: Simple upsert to workspace-specific collection
self.qdrant_client.upsert(
    collection_name=self.collection_name,
    points=points
)

# NEW: Upsert with shard key and workspace metadata
self.qdrant_client.upsert(
    collection_name=self.collection_name,
    points=points,
    shard_key_selector=self.organization_id  # Routes to org's shard
)

# Point payload now includes workspace_id
point_metadata = {
    "content": chunk,
    "document_id": doc_id,
    "chunk_index": i,
    "organization_id": self.organization_id,
    "workspace_id": self.workspace_id or "org_default"  # NEW FIELD
}
```

### 3. Search Operations
```python
# OLD: Simple search on workspace collection
search_results = self.qdrant_client.search(
    collection_name=self.collection_name,
    query_vector=query_embedding,
    limit=top_k,
    query_filter=filter
)

# NEW: Search with shard key and workspace filter
must_conditions = [
    FieldCondition(
        key="workspace_id", 
        match=MatchValue(value=self.workspace_id or "org_default")
    )
]
search_filter = Filter(must=must_conditions)

search_results = self.qdrant_client.search(
    collection_name=self.collection_name,
    query_vector=query_embedding,
    limit=top_k,
    query_filter=search_filter,
    shard_key_selector=self.organization_id  # Search only org's shard
)
```

### 4. Document Deletion
```python
# OLD: Delete from workspace collection
self.qdrant_client.delete(
    collection_name=self.collection_name,
    points_selector=Filter(must=[
        FieldCondition(key="document_id", match=MatchValue(value=document_id))
    ])
)

# NEW: Delete with shard key and workspace filter
self.qdrant_client.delete(
    collection_name=self.collection_name,
    points_selector=Filter(must=[
        FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        FieldCondition(key="workspace_id", match=MatchValue(value=self.workspace_id or "org_default"))
    ]),
    shard_key_selector=self.organization_id
)
```

## Migration Steps

### Step 1: Update Environment Variables
Add the new shard configuration:
```bash
# .env file
RAG_COLLECTION_NAME=ai_engine_knowledge_base
QDRANT_SHARD_NUMBER=4  # Adjust based on your Qdrant cluster size
```

### Step 2: Backup Existing Data
Before migrating, backup your existing Qdrant collections:
```python
# Create backup script
import json
from qdrant_client import QdrantClient

def backup_collection(client, collection_name, backup_file):
    # Scroll through all points and save to file
    offset = None
    all_points = []
    
    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )
        
        points, next_offset = result
        all_points.extend(points)
        
        if next_offset is None:
            break
        offset = next_offset
    
    with open(backup_file, 'w') as f:
        json.dump([{
            'id': point.id,
            'vector': point.vector,
            'payload': point.payload
        } for point in all_points], f)
```

### Step 3: Run Migration Script
Create and run a migration script to move data from old collections to the new sharded collection:

```python
async def migrate_to_sharded_collection():
    client = QdrantClient(url="your_qdrant_url")
    
    # Get all existing collections
    collections = client.get_collections()
    old_collections = [col.name for col in collections.collections 
                      if col.name.startswith("default_collection_")]
    
    # Create new global collection with sharding
    client.create_collection(
        collection_name="ai_engine_knowledge_base",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        sharding_method=ShardingMethod.CUSTOM,
        shard_number=4
    )
    
    for old_collection in old_collections:
        # Parse organization and workspace from collection name
        parts = old_collection.replace("default_collection_", "").split("_")
        org_id = parts[0]
        workspace_id = "_".join(parts[1:]) if len(parts) > 1 else "org_default"
        
        # Migrate points
        offset = None
        while True:
            result = client.scroll(
                collection_name=old_collection,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=True
            )
            
            points, next_offset = result
            
            # Update point payloads with workspace_id
            migrated_points = []
            for point in points:
                new_payload = point.payload.copy()
                new_payload["workspace_id"] = workspace_id
                
                migrated_points.append(PointStruct(
                    id=point.id,
                    vector=point.vector,
                    payload=new_payload
                ))
            
            if migrated_points:
                client.upsert(
                    collection_name="ai_engine_knowledge_base",
                    points=migrated_points,
                    shard_key_selector=org_id
                )
            
            if next_offset is None:
                break
            offset = next_offset
        
        print(f"Migrated collection: {old_collection}")
```

### Step 4: Update Application Code
Deploy the updated RAG tool code with the new sharding logic.

### Step 5: Verify Migration
Run tests to ensure:
1. All data has been migrated correctly
2. Workspace isolation is maintained
3. Performance is improved
4. No data leakage between workspaces

### Step 6: Clean Up Old Collections
After verifying the migration:
```python
# Delete old collections (BE VERY CAREFUL!)
for old_collection in old_collections:
    # Double-check this is an old collection
    if old_collection.startswith("default_collection_"):
        client.delete_collection(old_collection)
        print(f"Deleted old collection: {old_collection}")
```

## Testing the New System

Use the provided test script (`test_sharded_rag.py`) to verify:

1. **Data Isolation**: Each workspace only accesses its own data
2. **Organization Sharding**: Organizations are properly isolated at the shard level
3. **Performance**: Search operations are faster with proper sharding
4. **Scalability**: System can handle more organizations and workspaces

```bash
python test_sharded_rag.py
```

## Environment Configuration

### Required Environment Variables
```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_api_key_here
QDRANT_SHARD_NUMBER=4  # Adjust based on your cluster size

# RAG Configuration  
RAG_COLLECTION_NAME=ai_engine_knowledge_base
RAG_VECTOR_SIZE=1536
RAG_DISTANCE_METRIC=Cosine
RAG_TOP_K=5
RAG_SIMILARITY_THRESHOLD=0.7

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
DEFAULT_EMBEDDING_MODEL=text-embedding-3-small
```

## Benefits After Migration

### Performance Benefits
- **Faster Searches**: Queries only search within the organization's shard
- **Better Resource Utilization**: Qdrant manages one large collection more efficiently
- **Reduced Latency**: Less overhead from collection management

### Scalability Benefits  
- **Unlimited Organizations**: No practical limit on the number of organizations
- **Simplified Management**: One collection to backup, monitor, and maintain
- **Elastic Scaling**: Easy to add more shards as you grow

### Operational Benefits
- **Simplified Backups**: One collection to backup instead of thousands
- **Easier Monitoring**: Monitor one collection's health and performance
- **Consistent Configuration**: All tenants share the same vector parameters

## Troubleshooting

### Common Issues

1. **Shard Key Not Found**: Ensure `shard_key_selector` is provided in all operations
2. **Data Leakage**: Verify workspace filters are applied correctly in all queries
3. **Performance Issues**: Check that shard numbers are appropriate for your cluster
4. **Migration Failures**: Ensure old collection data includes organization_id in payload

### Monitoring and Observability

Add logging to track:
- Shard key usage in operations
- Workspace filter effectiveness
- Query performance metrics
- Data isolation verification

## Conclusion

The migration to a single collection with custom sharding provides significant benefits in terms of performance, scalability, and operational efficiency. While it requires careful migration planning, the long-term benefits far outweigh the initial migration effort.

For enterprise SaaS applications, this architecture is the recommended approach for multi-tenant vector databases at scale.
