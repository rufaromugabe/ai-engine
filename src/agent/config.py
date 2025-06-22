"""Configuration management for the multi-tenant AI agent system."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

class ToolType(str, Enum):
    """Available tool types in the system."""
    RAG = "rag"
    HTTP_API = "http_api"
    CALCULATOR = "calculator"
    SEARCH = "search"
    DATABASE = "database"

class DistanceMetric(str, Enum):
    """Qdrant distance metrics."""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclidean"
    DOT = "Dot"

@dataclass
class RAGConfig:
    """Configuration for RAG tool."""
    collection_name: str = field(default_factory=lambda: os.getenv("RAG_COLLECTION_NAME", "ai_engine_knowledge_base"))
    vector_size: int = field(default_factory=lambda: int(os.getenv("RAG_VECTOR_SIZE", "1536")))
    distance_metric: DistanceMetric = field(default_factory=lambda: DistanceMetric(os.getenv("RAG_DISTANCE_METRIC", "Cosine")))
    top_k: int = field(default_factory=lambda: int(os.getenv("RAG_TOP_K", "5")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7")))
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = field(default_factory=lambda: os.getenv("DEFAULT_EMBEDDING_MODEL", "text-embedding-3-small"))

@dataclass
class HTTPAPIConfig:
    """Configuration for HTTP API tool."""
    timeout: int = field(default_factory=lambda: int(os.getenv("DEFAULT_API_TIMEOUT", "30")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    allowed_domains: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)

@dataclass
class DatabaseConfig:
    """Configuration for database tool."""
    connection_string: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))
    query_timeout: int = 30
    max_results: int = 100

@dataclass
class LLMConfig:
    """Configuration for LLM settings."""
    model: str = field(default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4"))
    temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.1")))
    max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "1000")))
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))

@dataclass
class QdrantConfig:
    """Configuration for Qdrant vector database."""
    url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "http://localhost:6333"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY"))
    timeout: int = 30

@dataclass
class WorkspaceSpecificConfig:
    """Configuration specific to a workspace within an organization."""
    workspace_id: str
    organization_id: str
    rag_config: Optional[RAGConfig] = None
    http_api_config: Optional[HTTPAPIConfig] = None
    database_config: Optional[DatabaseConfig] = None
    llm_config: Optional[LLMConfig] = None
    qdrant_config: Optional[QdrantConfig] = None
    custom_settings: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OrganizationConfig:
    """Configuration for a specific organization/tenant."""
    organization_id: str
    enabled_tools: List[ToolType] = field(default_factory=list)
    rag_config: RAGConfig = field(default_factory=RAGConfig)
    http_api_config: HTTPAPIConfig = field(default_factory=HTTPAPIConfig)
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    qdrant_config: QdrantConfig = field(default_factory=QdrantConfig)
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    workspace_configs: Dict[str, WorkspaceSpecificConfig] = field(default_factory=dict)

class ConfigurationManager:
    """Manages configurations for multi-tenant system."""
    
    def __init__(self):
        self._organizations: Dict[str, OrganizationConfig] = {}
        self._workspace_configs: Dict[str, WorkspaceSpecificConfig] = {}  # workspace_id -> config
    
    def register_organization(self, config: OrganizationConfig) -> None:
        """Register a new organization configuration."""
        self._organizations[config.organization_id] = config
        
    def get_organization_config(self, organization_id: str) -> OrganizationConfig:
        """Get configuration for a specific organization."""
        if organization_id not in self._organizations:
            raise ValueError(f"Organization '{organization_id}' not found. Please set it up first.")
        return self._organizations[organization_id]
    
    def register_workspace_config(self, workspace_config: WorkspaceSpecificConfig) -> None:
        """Register workspace-specific configuration."""
        self._workspace_configs[workspace_config.workspace_id] = workspace_config
        
        # Also add to organization's workspace configs
        org_config = self.get_organization_config(workspace_config.organization_id)
        org_config.workspace_configs[workspace_config.workspace_id] = workspace_config
    
    def get_workspace_config(self, workspace_id: str) -> Optional[WorkspaceSpecificConfig]:
        """Get workspace-specific configuration."""
        return self._workspace_configs.get(workspace_id)
    
    def get_effective_config(self, organization_id: str, workspace_id: Optional[str] = None, config_type: str = "rag"):
        """Get effective configuration, prioritizing workspace-specific settings."""
        org_config = self.get_organization_config(organization_id)
        
        if workspace_id and workspace_id in self._workspace_configs:
            workspace_config = self._workspace_configs[workspace_id]
            
            # Return workspace-specific config if available, otherwise fall back to org config
            if config_type == "rag":
                return workspace_config.rag_config or org_config.rag_config
            elif config_type == "llm":
                return workspace_config.llm_config or org_config.llm_config
            elif config_type == "qdrant":
                return workspace_config.qdrant_config or org_config.qdrant_config
            elif config_type == "http_api":
                return workspace_config.http_api_config or org_config.http_api_config
            elif config_type == "database":
                return workspace_config.database_config or org_config.database_config
        
        # Fallback to organization config
        if config_type == "rag":
            return org_config.rag_config
        elif config_type == "llm":
            return org_config.llm_config
        elif config_type == "qdrant":
            return org_config.qdrant_config
        elif config_type == "http_api":
            return org_config.http_api_config
        elif config_type == "database":
            return org_config.database_config
    
    def update_organization_config(self, organization_id: str, updates: Dict[str, Any]) -> None:
        """Update organization configuration."""
        if organization_id in self._organizations:
            config = self._organizations[organization_id]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    def update_workspace_config(self, workspace_id: str, updates: Dict[str, Any]) -> bool:
        """Update workspace-specific configuration."""
        if workspace_id in self._workspace_configs:
            config = self._workspace_configs[workspace_id]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return True
        return False
    
    def is_tool_enabled(self, organization_id: str, tool: ToolType) -> bool:
        """Check if a tool is enabled for an organization."""
        config = self.get_organization_config(organization_id)
        return tool in config.enabled_tools
    
    def get_rag_config(self, organization_id: str, workspace_id: Optional[str] = None) -> RAGConfig:
        """Get RAG configuration for an organization/workspace."""
        return self.get_effective_config(organization_id, workspace_id, "rag")
    
    def get_llm_config(self, organization_id: str, workspace_id: Optional[str] = None) -> LLMConfig:
        """Get LLM configuration for an organization/workspace."""
        return self.get_effective_config(organization_id, workspace_id, "llm")
    
    def get_qdrant_config(self, organization_id: str, workspace_id: Optional[str] = None) -> QdrantConfig:
        """Get Qdrant configuration for an organization/workspace."""
        return self.get_effective_config(organization_id, workspace_id, "qdrant")

# Global configuration manager instance
config_manager = ConfigurationManager()
