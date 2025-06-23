# Multi-Workspace AI Agent System

A comprehensive multi-tenant AI agent system that supports organizations with multiple workspaces, each containing specialized agents with different tools and configurations.

## 🏗️ System Architecture

```
Organizations
    └── Workspaces (isolated environments)
        └── Agents (with specific configurations)
            └── Tools (RAG, HTTP API, etc.)
                └── Knowledge Base (workspace-specific collections)
```

## 🌟 Key Features

### 🏢 Organization Management

- **Multi-tenant support**: Complete isolation between organizations
- **Configurable tools**: Each organization can enable/disable specific tools
- **Custom settings**: Organization-level configuration overrides

### 🎯 Workspace Management

- **Isolated environments**: Each workspace operates independently
- **Shared resources**: Common tools and settings across workspace agents
- **Flexible configuration**: Workspace-specific overrides for tools and models

### 🤖 Agent Management

- **Specialized agents**: Each agent can have unique behavior and instructions
- **Custom personalities**: System prompts and custom instructions per agent
- **Tool selection**: Agents can use different subsets of available tools
- **Model configuration**: Per-agent model settings (temperature, max_tokens, etc.)

### 📚 Knowledge Base Isolation

- **Workspace-specific collections**: Documents are isolated per workspace
- **Agent context**: Each agent accesses only its workspace's knowledge
- **Scalable storage**: Qdrant vector database with isolated collections

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with your configuration:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Multi-tenant Configuration
ENABLE_MULTI_TENANT=true

# Workspace Configuration
ENABLE_WORKSPACE_ISOLATION=true
MAX_WORKSPACES_PER_ORG=10
MAX_AGENTS_PER_WORKSPACE=5

# Agent Configuration
DEFAULT_AGENT_MODEL=gpt-4
DEFAULT_AGENT_TEMPERATURE=0.1
DEFAULT_AGENT_MAX_TOKENS=1000
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the API Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### 4. Test the System

```bash
# Run the workspace system test
python test_workspace_system.py

# Or check the API usage examples
python api_usage_examples.py
```

## 📋 API Reference

### Organization Management

#### Setup Organization

```http
POST /api/v1/setup-organization
Content-Type: application/json

{
    "organization_id": "acme_corp",
    "enabled_tools": ["rag"]
}
```

**Note**: RAG, LLM, Qdrant, and Database configurations are now system-wide and managed via environment variables only. Individual organizations cannot override these settings.

### Workspace Management

#### Create Workspace

```http
POST /api/v1/workspaces
Content-Type: application/json

{
    "organization_id": "acme_corp",
    "name": "Customer Support",
    "description": "Customer support team workspace",
    "shared_tools": ["rag"],
    "shared_settings": {
        "response_style": "friendly"
    }
}
```

#### List Organization Workspaces

```http
GET /api/v1/organizations/{organization_id}/workspaces
```

#### Get Workspace Details

```http
GET /api/v1/workspaces/{workspace_id}
```

### Agent Management

#### Create Agent

```http
POST /api/v1/workspaces/{workspace_id}/agents
Content-Type: application/json

{
    "name": "Support Assistant",
    "description": "Helpful customer support agent",
    "enabled_tools": ["rag"],
    "custom_instructions": "Be friendly and solution-focused",
    "system_prompt": "You are a customer support agent",
    "model_settings": {
        "temperature": 0.3,
        "max_tokens": 500
    }
}
```

#### Update Agent

```http
PUT /api/v1/workspaces/{workspace_id}/agents/{agent_id}
Content-Type: application/json

{
    "custom_instructions": "Updated instructions",
    "model_settings": {
        "temperature": 0.4
    }
}
```

#### List Workspace Agents

```http
GET /api/v1/workspaces/{workspace_id}/agents
```

### Enhanced Query System

#### Query Specific Agent

```http
POST /api/v1/query
Content-Type: application/json

{
    "query": "How can I reset my password?",
    "organization_id": "acme_corp",
    "workspace_id": "workspace_123",
    "agent_id": "agent_456"
}
```

### Document Management

#### Ingest Document to Workspace

```http
POST /api/v1/ingest
Content-Type: application/json

{
    "content": "Document content here...",
    "organization_id": "acme_corp",
    "workspace_id": "workspace_123",
    "metadata": {
        "document_type": "support_guide",
        "version": "1.0"
    }
}
```

## 🎨 Use Cases

### 1. Customer Support Organization

```python
# Organization: TechCorp
# Workspace 1: General Support
#   - Agent 1: Friendly Assistant (general inquiries)
#   - Agent 2: Billing Specialist (payment issues)
# Workspace 2: Technical Support
#   - Agent 1: Bug Hunter (technical issues)
#   - Agent 2: Integration Expert (API support)
```

### 2. E-commerce Company

```python
# Organization: ShopCorp
# Workspace 1: Customer Service
#   - Agent 1: Order Assistant (order status, shipping)
#   - Agent 2: Returns Specialist (returns, refunds)
# Workspace 2: Product Support
#   - Agent 1: Product Expert (product questions)
#   - Agent 2: Recommendation Engine (product suggestions)
```

### 3. SaaS Platform

```python
# Organization: CloudSoft
# Workspace 1: User Onboarding
#   - Agent 1: Setup Assistant (account setup)
#   - Agent 2: Training Coach (feature tutorials)
# Workspace 2: Enterprise Support
#   - Agent 1: Account Manager (enterprise clients)
#   - Agent 2: Technical Architect (complex integrations)
```

## 🔧 Configuration Options

### Agent Configuration

```python
{
    "name": "Agent Name",
    "description": "Agent description",
    "enabled_tools": ["rag", "http_api"],
    "custom_instructions": "Specific behavior instructions",
    "system_prompt": "System-level prompt",
    "model_settings": {
        "model": "gpt-4",
        "temperature": 0.3,
        "max_tokens": 800
    },
    "tool_configurations": {
        "rag": {
            "top_k": 3,
            "similarity_threshold": 0.8
        }
    }
}
```

### Workspace Configuration

```python
{
    "organization_id": "org_id",
    "name": "Workspace Name",
    "description": "Workspace description",
    "shared_tools": ["rag"],
    "shared_settings": {
        "default_language": "en",
        "response_format": "markdown"
    }
}
```

## 🔐 Security & Isolation

- **Organization Isolation**: Complete data separation between organizations
- **Workspace Isolation**: Knowledge bases are isolated per workspace
- **Agent Permissions**: Agents only access their workspace's tools and data
- **API Security**: Organization and workspace validation on all endpoints

## 📊 Monitoring & Analytics

### Workspace Statistics

```http
GET /api/v1/workspaces/{workspace_id}/stats
```

Returns:

- Total agents
- Active agents
- Tool usage
- Document count
- Creation/update timestamps

### Knowledge Base Information

```http
GET /api/v1/knowledge-base/{organization_id}?workspace_id={workspace_id}
```

Returns workspace-specific knowledge base metrics.

## 🛠️ Development

### Project Structure

```
src/
├── agent/
│   ├── workspace_manager.py    # Workspace and agent management
│   ├── enhanced_graph.py       # Multi-workspace graph execution
│   ├── config.py              # Configuration management
│   ├── tool_manager.py        # Tool management with workspace support
│   └── tools/
│       ├── base_tool.py       # Enhanced base tool with workspace context
│       └── rag_tool.py        # Workspace-aware RAG tool
├── main.py                    # FastAPI application with all endpoints
└── test_workspace_system.py   # Comprehensive test suite
```

### Adding New Tools

1. Extend `BaseTool` class with workspace context
2. Register in `ToolManager`
3. Add to `ToolType` enum
4. Update agent configurations

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python test_workspace_system.py

# Test API endpoints
python api_usage_examples.py
```

## 🎯 Roadmap

- [ ] **Advanced Routing**: Intelligent agent selection based on query content
- [ ] **Tool Marketplace**: Plugin system for custom tools
- [ ] **Analytics Dashboard**: Web interface for monitoring and management
- [ ] **Role-based Access**: User permissions and role management
- [ ] **Webhook Integration**: Real-time notifications and integrations
- [ ] **Multi-language Support**: Internationalization features
- [ ] **Performance Optimization**: Caching and performance improvements

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📞 Support

For support and questions:

- Create an issue in the repository
- Check the API usage examples
- Review the test files for implementation guidance
