# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenVela is a Python library for building agentic workflows using language models. It supports multiple workflow paradigms (Chain of Thought, Tree of Thought, Fluid Chain of Thought, AutoSelect) with various agent types and LLM integrations (OpenAI, Groq, Ollama).

## Common Commands

### Development Setup
```bash
# Install dependencies (includes new Pydantic, Typer, Rich)
poetry install

# Run the modern CLI
poetry run openvela --help

# Or use the new CLI directly
poetry run python -m openvela.cli_new.main --help
```

### CLI Commands
```bash
# Run API server
openvela serve --host 0.0.0.0 --port 8000

# Run interactive interface
openvela interface

# Execute workflow
openvela run --provider openai --workflow_type cot --base_url_or_api_key YOUR_API_KEY --model gpt-4o-mini --agents path/to/agents.json --task "Your task"
```

### Testing
```bash
# Run tests (simple test structure, no pytest in dependencies)
python3 -m unittest discover tests/
```

### Build
```bash
# Build package
poetry build
```

## Architecture

### New Architecture (Improved)

**Core** (`openvela/core/`): Type-safe foundation with modern Python patterns
- `config.py`: Pydantic models for type-safe configuration (AgentConfig, WorkflowConfig, ModelConfig)
- `protocols.py`: Interface definitions using Protocol pattern
- `exceptions.py`: Comprehensive exception hierarchy with rich context
- `factories.py`: Factory pattern for clean object creation
- `types.py`: Enums for type safety (AgentType, WorkflowType, ModelProvider)

**Agents** (`openvela/agents_new/`): Improved agent implementations
- `base.py`: Abstract BaseAgent with proper error handling and logging
- `basic.py`: Agent, StartAgent, EndAgent with type safety
- `supervisor.py`: SupervisorAgent with enum-based configuration
- `fluid.py`: FluidAgent for dynamic generation
- `sql.py`: SQLAgent for database operations

**CLI** (`openvela/cli_new/`): Modern CLI using Typer and Rich
- Rich terminal output with colors and formatting
- Better error messages and validation
- Project initialization command
- Configuration validation tools

### Legacy Components (Original)

**Workflows** (`openvela/workflows.py`): Abstract base class defining workflow execution patterns
- `ChainOfThoughtWorkflow`: Sequential agent processing
- `TreeOfThoughtWorkflow`: Parallel thought generation
- `FluidChainOfThoughtWorkflow`: Dynamic agent generation
- `AutoSelectWorkflow`: Adaptive agent selection

**Agents** (`openvela/agents.py`): Core execution units (legacy)
- `Agent`: Base agent class
- `StartAgent`/`EndAgent`: Workflow boundary agents
- `SupervisorAgent`: Workflow orchestration with two modes
- `FluidAgent`: Dynamic agent generation
- `SQLAgent`: Specialized for read-only SQL operations

**LLMs** (`openvela/llms.py`): Model abstraction layer
- `OpenAIModel`: OpenAI integration
- `GroqModel`: Groq integration  
- `OllamaModel`: Ollama integration

**Memory** (`openvela/memory.py`): Conversation and workflow state management
- `WorkflowMemory`: Cross-workflow conversation history
- `JsonShortTermMemory`: Agent-level memory
- `AgentMemory`: Individual agent state

### Key Architectural Patterns

#### New Patterns (Improved)
1. **Factory Pattern**: Centralized object creation with automatic validation and registration
2. **Protocol-Based Design**: Interface segregation using typing.Protocol for better testability
3. **Type-Safe Configuration**: Pydantic models with automatic validation and serialization
4. **Comprehensive Error Handling**: Custom exception hierarchy with rich context information
5. **Dependency Injection**: Clean separation of concerns with proper dependency management

#### Legacy Patterns (Original)
1. **Agent-Supervisor Pattern**: SupervisorAgent orchestrates workflow by selecting next agent based on `agent_type` setting
2. **Memory Management**: JSON-based memory system with workflow-level and agent-level persistence  
3. **Tool Integration**: AIFunctionTool system for extending agent capabilities
4. **File Processing**: Support for audio/image files through specialized file handlers

### Entry Points

- CLI: `openvela/cli.py` - Command-line interface
- Server: `openvela/server.py` - FastAPI REST API  
- Interactive: `openvela/__main__.py` - Interactive workflow interface
- Runner: `openvela/runner.py` - Core workflow execution logic

### Configuration

#### New Type-Safe Configuration (Recommended)
```python
from openvela import AgentConfig, WorkflowConfig, ModelConfig, AgentType, WorkflowType, ModelProvider

# Type-safe agent configuration with validation
agent_config = AgentConfig(
    name="AgentName",
    prompt="Agent instructions",
    input="Initial input",
    agent_type=AgentType.SIMPLE,  # Enum for type safety
    options={"temperature": 0.7}
)

# Type-safe workflow configuration
workflow_config = WorkflowConfig(
    type=WorkflowType.CHAIN_OF_THOUGHT,
    validate_output=True,
    max_attempts=3
)

# Type-safe model configuration
model_config = ModelConfig(
    provider=ModelProvider.OPENAI,
    model_name="gpt-4o-mini",
    api_key="your-key",
    temperature=0.7
)
```

#### Legacy Dictionary Configuration (Still Supported)
```python
{
    "name": "AgentName",
    "prompt": "Agent instructions", 
    "input": "Initial input",
    "agent_type": "simple|selector"  # For SupervisorAgent
}
```

#### Modern Usage with Factories
```python
from openvela import AgentFactory, WorkflowFactory, ModelFactory

# Create using factories with automatic validation
model = ModelFactory.create_model(model_config)
agent = AgentFactory.create_agent(agent_config, model)
workflow = WorkflowFactory.create_workflow(workflow_config, agents=[agent], supervisor=agent)
```