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
    """RAG tool for document ingestion and retrieval using Qdrant."""

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
        """Create necessary indexes for filtering."""
        indexes_to_create = [
            ("organization_id", "keyword"),
            ("workspace_id", "keyword"), 
            ("document_id", "keyword")
        ]
        
        for field_name, field_type in indexes_to_create:
            try:
                self.qdrant_client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=field_type
                )
                logger.info(f"Created index for field '{field_name}' in collection: {self.collection_name}")
                
            except Exception as e:
                # Index might already exist, which is fine
                if "already exists" in str(e).lower() or "index" in str(e).lower():
                    logger.info(f"Index for field '{field_name}' already exists in collection: {self.collection_name}")
                else:
                    logger.warning(f"Error creating index for field '{field_name}': {str(e)}")
        
        logger.info(f"Index creation completed for collection: {self.collection_name}")

    async def execute(self, query: str, top_k: Optional[int] = None, filter_conditions: Optional[Dict] = None) -> ToolResult:
        """Execute RAG retrieval."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
            )

        try:
            query_embedding = await self._get_embedding(query)
            search_top_k = top_k or self.config.top_k

            must_conditions = [
                FieldCondition(
                    key="workspace_id",
                    match=MatchValue(value=self.workspace_id or "org_default")
                )
            ]

            if filter_conditions:
                for key, value in filter_conditions.items():
                    must_conditions.append(
                        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
                    )

            search_filter = Filter(must=must_conditions)
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=search_top_k,
                score_threshold=self.config.similarity_threshold,
                query_filter=search_filter
            )

            results = []
            for result in search_results:
                results.append({
                    "content": result.payload.get("content", ""),
                    "metadata": result.payload.get("metadata", {}),
                    "score": result.score,
                    "document_id": result.payload.get("document_id", "")
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
        """Ingest a document into the knowledge base."""
        if not self._initialized:
            return ToolResult(
                success=False,
                data=None,
                error="RAG tool not initialized"
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
                    point_metadata["metadata"] = metadata

                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=point_metadata
                ))

            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            logger.info(f"Ingested document {doc_id} to org '{self.organization_id}' workspace '{self.workspace_id}'")

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
        """Build Qdrant filter from conditions."""
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

        for key, value in conditions.items():
            if key not in ["organization_id", "workspace_id"]:
                must_conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
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
                "description": self.description,
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
                            "description": "Optional filters to apply to the search (e.g., document type, date range)"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.qdrant_client:
            self.qdrant_client.close()
        self._initialized = False
        logger.info(f"RAG tool cleanup completed for organization: {self.organization_id}")
