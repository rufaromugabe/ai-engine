# Multi-Workspace LangGraph Agent

[![CI](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/langchain-ai/new-langgraph-project/actions/workflows/integration-tests.yml)

This project demonstrates an enhanced multi-workspace AI agent system built with [LangGraph](https://github.com/langchain-ai/langgraph). It supports multiple organizations, workspaces, and agents with RAG capabilities and tool management.

<div align="center">
  <img src="./static/studio_ui.png" alt="Graph view in LangGraph studio UI" width="75%" />
</div>

## Features

- 🏢 **Multi-Organization Support**: Isolate data and configurations by organization
- 🏠 **Workspace Management**: Create separate workspaces within organizations  
- 🤖 **Agent Configuration**: Configure agents per workspace with specific tools and instructions
- 🔍 **RAG Integration**: Built-in Retrieval-Augmented Generation with Qdrant vector store
- 🛠️ **Extensible Tool System**: Modular tool architecture for easy extension
- 🎯 **Smart Routing**: Intelligent query routing based on content and agent configuration
- 📊 **LangGraph Studio Integration**: Visual debugging and development interface

## Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key
- Qdrant vector database (for RAG functionality)

### Installation

1. **Install dependencies and LangGraph CLI**:

```bash
cd path/to/your/app
pip install -e . "langgraph-cli[inmem]"
```

2. **Configure environment variables**:

The `.env` file is already configured with the necessary settings. Update the following values:

```bash
# Required: Add your OpenAI API key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Configure Qdrant for RAG (using cloud instance by default)
QDRANT_URL=https://your-qdrant-instance.com:6333
QDRANT_API_KEY=your_qdrant_api_key

# Optional: Enable LangSmith tracing
LANGSMITH_API_KEY=your_langsmith_api_key
```

3. **Start the LangGraph Development Server**:

**Option 1: Using the startup script (Recommended)**
```bash
# On Windows (Git Bash/WSL)
./start_dev.sh

# On Windows (Command Prompt)
start_dev.bat
```

**Option 2: Using make commands**
```bash
# Start development server
make dev

# Or explicitly start studio
make studio
```

**Option 3: Direct command**
```bash
langgraph dev --allow-blocking
```

This will start:
- **LangGraph Studio** on `http://localhost:8123` (visual interface)
- **API Server** on `http://localhost:8000` (REST API)

**Note**: The `--allow-blocking` flag is needed for development to handle synchronous operations properly.

### Development Tools

#### Quick Testing

Use the development runner for quick testing:

```bash
# Run all tests
python dev_runner.py test

# Interactive mode
python dev_runner.py interactive

# Test basic functionality
python dev_runner.py basic

# Test RAG functionality  
python dev_runner.py rag

# Test workspace isolation
python dev_runner.py workspace

# Show graph structure
python dev_runner.py info
```

#### LangGraph Studio

Open `http://localhost:8123` in your browser to access LangGraph Studio for:
- Visual graph debugging
- State inspection
- Step-by-step execution
- Configuration testing

## Architecture

### Graph Structure

The agent follows this workflow:

1. **Route Query**: Analyzes the user query and determines which tools to use
2. **Tool Execution**: Executes the appropriate tool (e.g., RAG search)
3. **Response Generation**: Generates a final response using the LLM with tool results

### Key Components

- **`src/agent/graph.py`**: Main graph entry point for LangGraph
- **`src/agent/enhanced_graph.py`**: Core graph implementation with multi-workspace support
- **`src/agent/workspace_manager.py`**: Workspace and agent configuration management
- **`src/agent/tool_manager.py`**: Tool registration and execution system
- **`src/agent/tools/`**: Individual tool implementations

### Multi-Workspace System

```
Organization
├── Workspace A
│   ├── Agent 1 (tools: RAG, API)
│   └── Agent 2 (tools: RAG only)
└── Workspace B
    ├── Agent 3 (tools: RAG, SQL)
    └── Agent 4 (tools: all tools)
```

Each level provides isolation and customization:
- **Organizations**: Complete data separation
- **Workspaces**: Project-level isolation within organizations
- **Agents**: Specific configurations and tool access within workspaces

## API Usage

### Execute Agent Query

```python
from src.agent.enhanced_graph import execute_agent_query

result = await execute_agent_query(
    organization_id="my-org",
    query="Search for information about AI",
    workspace_id="my-workspace",
    agent_id="my-agent"
)

print(result["response"])
```

### Ingest Documents for RAG

```python
from src.agent.enhanced_graph import ingest_document_for_organization

result = await ingest_document_for_organization(
    organization_id="my-org",
    content="Document content here...",
    metadata={"title": "My Document", "source": "upload"},
    workspace_id="my-workspace"
)
```

For more information on getting started with LangGraph Server, [see here](https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/).

## How to Customize

### 1. Add New Tools

Create a new tool in `src/agent/tools/`:

```python
from .base_tool import BaseTool

class MyCustomTool(BaseTool):
    async def execute(self, **kwargs):
        # Your tool implementation
        return ToolResult(success=True, data={"result": "custom result"})
```

Register it in `tool_manager.py`:

```python
self.register_tool("my_custom_tool", MyCustomTool)
```

### 2. Modify Agent Behavior

Update the routing logic in `enhanced_graph.py` to customize how queries are processed and tools are selected.

### 3. Extend Workspace Configuration

Add new fields to the `AgentConfig` class in `workspace_manager.py` to support additional agent customization.

## Development

### Hot Reload

LangGraph Studio supports hot reload - changes to your graph code will be automatically applied without restarting the server.

### State Management

You can edit past state and rerun your app from previous states in LangGraph Studio to debug specific nodes.

### Thread Management

Follow-up requests extend the same conversation thread. Use the `+` button in LangGraph Studio to create entirely new threads.

### Testing

Run the test suite:

```bash
# Unit tests
python -m pytest tests/unit_tests/

# Integration tests  
python -m pytest tests/integration_tests/

# Custom test runner
python dev_runner.py test
```

<!--
Configuration auto-generated by `langgraph template lock`. DO NOT EDIT MANUALLY.
{
  "config_schemas": {
    "agent": {
      "type": "object",
      "properties": {}
    }
  }
}
-->
# ai-engine
