"""RAG (Retrieval-Augmented Generation) tool using Qdrant and OpenAI embeddings."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import tiktoken

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    CollectionInfo,
    ShardingMethod
)
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .base_tool import BaseTool, ToolResult
from ..config import config_manager, DistanceMetric

logger = logging.getLogger(__name__)


class RAGTool(BaseTool):
    """RAG tool for document ingestion and retrieval using Qdrant with multitenant optimization.
    
    This implementation follows Qdrant's best practices for multitenancy:
    - Uses tenant indexing with is_tenant=True for organization_id for optimal performance
    - Places tenant filters first in all queries for efficient filtering
    - Supports grouped search for organizing results by metadata fields
    - Provides tenant-specific statistics and monitoring
    - Ensures data isolation between organizations and workspaces
    
    Multitenant Architecture:
    - organization_id: Primary tenant identifier (indexed with is_tenant=True)
    - workspace_id: Secondary tenant identifier for workspace-level isolation
    - All queries automatically include tenant filters for data security
    """

    def __init__(self, organization_id: str, workspace_id: Optional[str] = None):
        super().__init__(
            name="rag_tool",
            description="Retrieve relevant information from organization's knowledge base using semantic search",
            organization_id=organization_id,
            workspace_id=workspace_id
        )       
        self.qdrant_client: Optional[QdrantClient] = None
        self.embeddings: Optional[OpenAIEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.config = config_manager.get_rag_config(organization_id, workspace_id)
        self.qdrant_config = config_manager.get_qdrant_config(organization_id, workspace_id)
        self.collection_name = self.config.collection_name  # e.g., "ai_engine_knowledge_base"

    async def initialize(self) -> bool:
        """Initialize the RAG tool with Qdrant client and embeddings."""
        try:
            logger.info(f"Initializing RAG tool for organization '{self.organization_id}', workspace '{self.workspace_id or 'org_default'}'")
            
            self.qdrant_client = QdrantClient(
                url=self.qdrant_config.url,
                api_key=self.qdrant_config.api_key,
                timeout=self.qdrant_config.timeout
            )

            llm_config = config_manager.get_llm_config(self.organization_id, self.workspace_id)
            self.embeddings = OpenAIEmbeddings(
                model=self.config.embedding_model,
                openai_api_key=llm_config.api_key
            )

            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            await self._ensure_collection_exists()

            self._initialized = True
            logger.info(f"RAG tool initialized for organization: {self.organization_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG tool: {str(e)}")
            return False

    async def _ensure_collection_exists(self) -> None:
        """Ensure the global collection exists in Qdrant with proper configuration."""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                distance_map = {
                    DistanceMetric.COSINE: Distance.COSINE,
                    DistanceMetric.EUCLIDEAN: Distance.EUCLID,
                    DistanceMetric.DOT: Distance.DOT
                }

                # Create collection without custom sharding for cloud instances
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.config.vector_size,
                        distance=distance_map[self.config.distance_metric]
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")

            # Create indexes for filter fields
            self._create_indexes()

        except Exception as e:
            logger.error(f"Error ensuring collection exists: {str(e)}")
            raise

    def _create_indexes(self) -> None:
        """Create necessary indexes for filtering with multitenant optimization."""
        # Create tenant index for organization_id with is_tenant flag for optimal performance
        try:
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="organization_id",
                field_schema={
                    "type": "keyword",
                    "is_tenant": True  # Marks this as primary tenant identifier for optimization
                }
            )
            logger.info(f"Created tenant index for 'organization_id' in collection: {self.collection_name}")
        except Exception as e:
            if "already exists" in str(e).lower() or "index" in str(e).lower():
                logger.info(f"Tenant index for 'organization_id' already exists in collection: {self.collection_name}")
            else:
                logger.warning(f"Error creating tenant index for 'organization_id': {str(e)}")
        
        # Create secondary indexes for other fields
        secondary_indexes = [
            ("workspace_id", {"type": "keyword"}),
            ("document_id", {"type": "keyword"})
        ]
        
        for field_name, field_schema in secondary_indexes:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
                logger.info(f"Created index for field '{field_name}' in collection: {self.collection_name}")
                
            except Exception as e:
                # Index might already exist, which is fine
                if "already exists" in str(e).lower() or "index" in str(e).lower():
                    logger.info(f"Index for field '{field_name}' already exists in collection: {self.collection_name}")
                else:
                    logger.warning(f"Error creating index for field '{field_name}': {str(e)}")
        
        logger.info(f"Multitenant index creation completed for collection: {self.collection_name}")   
    async def execute(self, query: str, top_k: Optional[int] = None, filter_conditions: Optional[Dict] = None) -> ToolResult:
        """Execute RAG retrieval with multitenant optimization."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        if not self._validate_tenant_access():
            return ToolResult(
                success=False,
                data=None,
                error="Invalid tenant access configuration"
            )
        try:
            query_embedding = await self._get_embedding(query)
            search_top_k = top_k or self.config.top_k

            # Build optimized multitenant filter - tenant filter first for best performance
            must_conditions = [
                # Primary tenant identifier first for optimal filtering performance
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=self.organization_id)
                ),
                # Secondary tenant/workspace identifier
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=self.workspace_id or "org_default")
                )
            ]

            # Add additional filter conditions after tenant filters
            if filter_conditions:
                for key, value in filter_conditions.items():
                    # Avoid duplicate tenant filters
                    if key not in ["organization_id", "workspace_id"]:
                        must_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

            search_filter = Filter(must=must_conditions)
            
            search_params = {
                "collection_name": self.collection_name,
                "query_vector": query_embedding,
                "limit": search_top_k,
                "query_filter": search_filter
            }
            if self.config.similarity_threshold is not None:
                search_params["score_threshold"] = self.config.similarity_threshold

            search_results = self.qdrant_client.search(**search_params)

            results = []
            for result in search_results:
                payload = result.payload
                metadata = {
                    k: v for k, v in payload.items()
                    if k not in ["content", "document_id", "chunk_index", "organization_id", "workspace_id"]
                }
                results.append({
                    "content": payload.get("content", ""),
                    "metadata": metadata,
                    "score": result.score,
                    "document_id": payload.get("document_id", "")
                })

            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                },
                metadata={
                    "collection": self.collection_name,
                    "top_k": search_top_k,
                    "similarity_threshold": self.config.similarity_threshold
                }
            )

        except Exception as e:
            logger.error(f"RAG retrieval error: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"RAG retrieval failed: {str(e)}"
            )

    async def ingest_document(self, content: str, metadata: Optional[Dict] = None, document_id: Optional[str] = None) -> ToolResult:
        """Ingest a document into the knowledge base with multitenant isolation."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        if not self._validate_tenant_access():
            return ToolResult(
                success=False,
                data=None,
                error="Invalid tenant access configuration"
            )

        try:
            doc_id = document_id or str(uuid4())
            chunks = self.text_splitter.split_text(content)
            embeddings = await self._get_embeddings_batch(chunks)

            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_metadata = {
                    "content": chunk,
                    "document_id": doc_id,
                    "chunk_index": i,
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id or "org_default"
                }

                if metadata:
                    point_metadata.update(metadata)

                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=point_metadata
                ))
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            self._log_tenant_operation("document_ingest", {
                "document_id": doc_id,
                "chunks_created": len(chunks)
            })

            return ToolResult(
                success=True,
                data={
                    "document_id": doc_id,
                    "chunks_created": len(chunks),
                    "collection": self.collection_name
                },
                metadata={"operation": "ingest"}
            )

        except Exception as e:
            logger.error(f"Document ingestion error: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Document ingestion failed: {str(e)}"
            )

    async def delete_document(self, document_id: str) -> ToolResult:
        """Delete a document from the knowledge base."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        try:
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        ),
                        FieldCondition(
                            key="organization_id",
                            match=MatchValue(value=self.organization_id)
                        ),
                        FieldCondition(
                            key="workspace_id",
                            match=MatchValue(value=self.workspace_id or "org_default")
                        )
                    ]
                )
            )

            logger.info(f"Deleted document {document_id} from org '{self.organization_id}' workspace '{self.workspace_id}'")

            return ToolResult(
                success=True,
                data={"document_id": document_id},
                metadata={"operation": "delete"}
            )

        except Exception as e:
            logger.error(f"Document deletion error: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Document deletion failed: {str(e)}"
            )

    async def get_collection_info(self) -> ToolResult:
        """Get information about the collection."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="organization_id",
                            match=MatchValue(value=self.organization_id)
                        ),
                        FieldCondition(
                            key="workspace_id",
                            match=MatchValue(value=self.workspace_id or "org_default")
                        )
                    ]
                ),
                limit=1,
                with_payload=False,
                with_vectors=False
            )

            return ToolResult(
                success=True,
                data={
                    "collection_name": self.collection_name,
                    "total_points": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance.name,
                    "workspace_documents": len(scroll_result[0]),
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id or "org_default"
                }
            )

        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to get collection info: {str(e)}"
            )

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        try:
            embedding = await self.embeddings.aembed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts."""
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise

    def _build_filter(self, conditions: Dict) -> Filter:
        """Build optimized Qdrant filter from conditions with multitenant best practices."""
        # Start with tenant filters for optimal performance
        must_conditions = [
            # Primary tenant identifier first
            FieldCondition(
                key="organization_id",
                match=MatchValue(value=self.organization_id)
            ),
            # Secondary tenant/workspace identifier
            FieldCondition(
                key="workspace_id",
                match=MatchValue(value=self.workspace_id or "org_default")
            )
        ]

        # Add additional conditions, excluding tenant fields to avoid duplicates
        for key, value in conditions.items():
            if key not in ["organization_id", "workspace_id"]:
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )

        return Filter(must=must_conditions)

    def get_schema(self) -> Dict[str, Any]:
        """Get the tool's parameter schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": f"{self.description}. Optimized for multitenant access with tenant-specific filtering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Maximum number of results to return (optional)",
                            "minimum": 1,
                            "maximum": 20
                        },
                        "filter_conditions": {
                            "type": "object",
                            "description": "Optional filters to apply to the search (e.g., document type, date range). Tenant filters are automatically applied."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def get_grouped_search_schema(self) -> Dict[str, Any]:
        """Get the schema for grouped search functionality."""
        return {
            "type": "function",
            "function": {
                "name": "rag_grouped_search",
                "description": "Retrieve information grouped by specific fields (e.g., document type, author, category). Optimized for multitenant access.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant information"
                        },
                        "group_by": {
                            "type": "string",
                            "description": "Field to group results by (e.g., 'document_type', 'author', 'category')"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of groups to return",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5
                        },
                        "group_size": {
                            "type": "integer",
                            "description": "Maximum number of results per group",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5
                        },
                        "filter_conditions": {
                            "type": "object",
                            "description": "Optional filters to apply before grouping. Tenant filters are automatically applied."
                        }
                    },
                    "required": ["query", "group_by"]
                }
            }
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.qdrant_client:
            self.qdrant_client.close()
        self._initialized = False
        logger.info(f"RAG tool cleanup completed for organization: {self.organization_id}")

    async def execute_grouped_search(self, query: str, group_by: str, limit: int = 5, group_size: int = 5, filter_conditions: Optional[Dict] = None) -> ToolResult:
        """Execute RAG retrieval with results grouped by a specific field."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        try:
            query_embedding = await self._get_embedding(query)

            # Build optimized multitenant filter - tenant filter first for best performance
            must_conditions = [
                # Primary tenant identifier first for optimal filtering performance
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=self.organization_id)
                ),
                # Secondary tenant/workspace identifier
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=self.workspace_id or "org_default")
                )
            ]

            # Add additional filter conditions after tenant filters
            if filter_conditions:
                for key, value in filter_conditions.items():
                    # Avoid duplicate tenant filters
                    if key not in ["organization_id", "workspace_id"]:
                        must_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

            search_filter = Filter(must=must_conditions)

            # Perform grouped search
            try:
                from qdrant_client.models import QueryRequest
                search_request = QueryRequest(
                    query=query_embedding,
                    filter=search_filter,
                    limit=group_size,
                    with_payload=True,
                    with_vector=False
                )

                # Use scroll with grouping if supported, otherwise fall back to regular search
                # Note: Grouped search API may vary based on Qdrant version
                scroll_result = self.qdrant_client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=search_filter,
                    limit=limit * group_size,
                    with_payload=True,
                    with_vectors=False
                )

                # Group results manually by the specified field
                groups = {}
                for point in scroll_result[0]:
                    if point.payload and group_by in point.payload:
                        group_key = point.payload[group_by]
                        if group_key not in groups:
                            groups[group_key] = []
                        if len(groups[group_key]) < group_size:
                            groups[group_key].append({
                                "content": point.payload.get("content", ""),
                                "metadata": {
                                    k: v for k, v in point.payload.items()
                                    if k not in ["content", "document_id", "chunk_index", "organization_id", "workspace_id"]
                                },
                                "document_id": point.payload.get("document_id", ""),
                                "point_id": point.id
                            })

                # Limit to the specified number of groups
                limited_groups = dict(list(groups.items())[:limit])

                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "groups": limited_groups,
                        "total_groups": len(limited_groups),
                        "group_by": group_by
                    },
                    metadata={
                        "collection": self.collection_name,
                        "limit": limit,
                        "group_size": group_size,
                        "organization_id": self.organization_id,
                        "workspace_id": self.workspace_id or "org_default"
                    }
                )

            except ImportError:
                # Fallback to regular search if advanced features not available
                return await self.execute(query, top_k=limit * group_size, filter_conditions=filter_conditions)

        except Exception as e:
            logger.error(f"Grouped RAG retrieval error: {str(e)}")
            return ToolResult(                
                success=False,
                data=None,
                error=f"Grouped RAG retrieval failed: {str(e)}"
            )

    async def get_tenant_statistics(self) -> ToolResult:
        """Get statistics for the current tenant (organization/workspace)."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        try:
            # Use a more efficient approach - count with a single scroll request
            document_ids = set()
            total_chunks = 0
            max_points_to_scan = 1000  # Limit to prevent infinite loops
            
            # Get tenant-specific points with a reasonable limit
            scroll_result = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="organization_id",
                            match=MatchValue(value=self.organization_id)
                        ),
                        FieldCondition(
                            key="workspace_id",
                            match=MatchValue(value=self.workspace_id or "org_default")
                        )
                    ]
                ),
                limit=max_points_to_scan,
                with_payload=True,
                with_vectors=False
            )

            # Process the results
            points, next_page_offset = scroll_result
            for point in points:
                total_chunks += 1
                if point.payload and "document_id" in point.payload:
                    document_ids.add(point.payload["document_id"])

            # If there are more points, we'll indicate this in the response
            has_more = next_page_offset is not None
            
            return ToolResult(
                success=True,
                data={
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id or "org_default",
                    "total_documents": len(document_ids),
                    "total_chunks": total_chunks,
                    "collection_name": self.collection_name,
                    "avg_chunks_per_document": round(total_chunks / len(document_ids), 2) if document_ids else 0,
                    "scanned_limit": max_points_to_scan,
                    "has_more_data": has_more
                },
                metadata={
                    "operation": "tenant_statistics",
                    "collection": self.collection_name
                }
            )

        except Exception as e:
            logger.error(f"Error getting tenant statistics: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Failed to get tenant statistics: {str(e)}"
            )

    async def bulk_delete_tenant_data(self, confirm_organization_id: str) -> ToolResult:
        """
        Bulk delete all data for the current tenant (organization/workspace).
        
        Args:
            confirm_organization_id: Must match the current organization_id as a safety check
            
        Returns:
            ToolResult with deletion statistics
        """
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        # Safety check - require explicit confirmation of organization ID
        if confirm_organization_id != self.organization_id:
            return ToolResult(
                success=False,
                data=None,
                error="Organization ID confirmation does not match. Operation cancelled for safety."
            )

        try:
            # Get count before deletion
            stats_before = await self.get_tenant_statistics()
            
            # Delete all points for this tenant
            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="organization_id",
                            match=MatchValue(value=self.organization_id)
                        ),
                        FieldCondition(
                            key="workspace_id",
                            match=MatchValue(value=self.workspace_id or "org_default")
                        )
                    ]
                )
            )

            logger.warning(f"Bulk deleted all data for org '{self.organization_id}' workspace '{self.workspace_id}'")

            return ToolResult(
                success=True,
                data={
                    "organization_id": self.organization_id,
                    "workspace_id": self.workspace_id or "org_default",
                    "deleted_documents": stats_before.data.get("total_documents", 0) if stats_before.success else "unknown",
                    "deleted_chunks": stats_before.data.get("total_chunks", 0) if stats_before.success else "unknown",
                    "collection": self.collection_name
                },
                metadata={"operation": "bulk_delete_tenant"}
            )

        except Exception as e:
            logger.error(f"Bulk deletion error: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Bulk deletion failed: {str(e)}"
            )

    async def migrate_tenant_data(self, target_workspace_id: str, document_filter: Optional[Dict] = None) -> ToolResult:
        """
        Migrate documents from current workspace to target workspace within the same organization.
        
        Args:
            target_workspace_id: Target workspace ID to migrate data to
            document_filter: Optional filter to specify which documents to migrate
            
        Returns:
            ToolResult with migration statistics
        """
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        try:
            # Build filter for source data
            must_conditions = [
                FieldCondition(
                    key="organization_id",
                    match=MatchValue(value=self.organization_id)
                ),
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=self.workspace_id or "org_default")
                )
            ]

            if document_filter:
                for key, value in document_filter.items():
                    if key not in ["organization_id", "workspace_id"]:
                        must_conditions.append(
                            FieldCondition(key=key, match=MatchValue(value=value))
                        )

            source_filter = Filter(must=must_conditions)

            # Get all points to migrate
            migrated_count = 0
            next_page_offset = None
            
            while True:
                scroll_params = {
                    "collection_name": self.collection_name,
                    "scroll_filter": source_filter,
                    "limit": 100,
                    "with_payload": True,
                    "with_vectors": True
                }
                
                if next_page_offset:
                    scroll_params["offset"] = next_page_offset
                
                batch_result = self.qdrant_client.scroll(**scroll_params)
                points, next_page_offset = batch_result
                
                if not points:
                    break

                # Update points with new workspace_id and upsert
                updated_points = []
                for point in points:
                    new_payload = point.payload.copy()
                    new_payload["workspace_id"] = target_workspace_id
                    
                    updated_points.append(PointStruct(
                        id=str(uuid4()),  # New ID to avoid conflicts
                        vector=point.vector,
                        payload=new_payload
                    ))

                # Insert updated points
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=updated_points
                )
                
                migrated_count += len(updated_points)

            logger.info(f"Migrated {migrated_count} chunks from workspace '{self.workspace_id}' to '{target_workspace_id}' in org '{self.organization_id}'")

            return ToolResult(
                success=True,
                data={
                    "source_organization_id": self.organization_id,
                    "source_workspace_id": self.workspace_id or "org_default",
                    "target_workspace_id": target_workspace_id,
                    "migrated_chunks": migrated_count,
                    "collection": self.collection_name
                },
                metadata={"operation": "migrate_tenant_data"}
            )

        except Exception as e:
            logger.error(f"Migration error: {str(e)}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Migration failed: {str(e)}"
            )

    def _validate_tenant_access(self) -> bool:
        """Validate that tenant identifiers are properly set for multitenant access."""
        if not self.organization_id:
            logger.error("Organization ID is required for multitenant access")
            return False
            
        if not isinstance(self.organization_id, str) or len(self.organization_id.strip()) == 0:
            logger.error("Organization ID must be a non-empty string")
            return False
            
        # Workspace ID can be None (defaults to "org_default"), but if provided must be valid
        if self.workspace_id is not None:
            if not isinstance(self.workspace_id, str) or len(self.workspace_id.strip()) == 0:
                logger.error("Workspace ID must be a non-empty string if provided")
                return False
                
        logger.debug(f"Tenant access validated for org '{self.organization_id}', workspace '{self.workspace_id or 'org_default'}'")
        return True

    def _log_tenant_operation(self, operation: str, details: Dict = None) -> None:
        """Log tenant-specific operations for auditing purposes."""
        log_data = {
            "operation": operation,
            "organization_id": self.organization_id,
            "workspace_id": self.workspace_id or "org_default",
            "collection": self.collection_name
        }
        
        if details:
            log_data.update(details)
            
        logger.info(f"Tenant operation: {log_data}")
