# __init__.py

# Core components (new)
from openvela.core import (
    # Configuration
    AgentConfig,
    WorkflowConfig,
    ModelConfig,
    # Types
    AgentType,
    WorkflowType,
    ModelProvider,
    # Exceptions
    OpenVelaError,
    ConfigurationError,
    AgentError,
    WorkflowError,
    ModelError,
    MemoryError,
    ValidationError,
    FactoryError,
    # Protocols
    MemoryProtocol,
    ModelProtocol,
    AgentProtocol,
    WorkflowProtocol,
    # Factories
    AgentFactory,
    WorkflowFactory,
    ModelFactory,
    auto_register_factories,
)

# Legacy components (existing)
from openvela.agents import Agent, EndAgent, FluidAgent, StartAgent, SupervisorAgent
from openvela.files import File, OpenVelaAudioFile, OpenVelaImageFile
from openvela.llms import GroqModel, Model, OllamaModel, OpenAIModel
from openvela.logs import configure_logging
from openvela.memory import (
    AgentMemory,
    JsonMemoryFormat,
    JsonReader,
    JsonShortTermMemory,
    MemoryFormat,
    ShortTermMemory,
    WorkflowMemory,
)
from openvela.messages import AssistantMessage, Message, SystemMessage, UserMessage
from openvela.tasks import Task
from openvela.tools import AIFunctionTool, OpenAIFunction, Tool
from openvela.workflows import (
    ChainOfThoughtWorkflow,
    FluidChainOfThoughtWorkflow,
    TreeOfThoughtWorkflow,
    Workflow,
)

__all__ = [
    # Core (new)
    "AgentConfig",
    "WorkflowConfig", 
    "ModelConfig",
    "AgentType",
    "WorkflowType",
    "ModelProvider",
    "OpenVelaError",
    "ConfigurationError",
    "AgentError",
    "WorkflowError",
    "ModelError",
    "MemoryError",
    "ValidationError",
    "FactoryError",
    "MemoryProtocol",
    "ModelProtocol",
    "AgentProtocol",
    "WorkflowProtocol",
    "AgentFactory",
    "WorkflowFactory",
    "ModelFactory",
    "auto_register_factories",
    # Legacy (existing)
    "File",
    "OpenVelaAudioFile",
    "OpenVelaImageFile",
    "Task",
    "MemoryFormat",
    "JsonMemoryFormat",
    "ShortTermMemory",
    "JsonShortTermMemory",
    "WorkflowMemory",
    "AgentMemory",
    "JsonReader",
    "Tool",
    "OpenAIFunction",
    "AIFunctionTool",
    "Agent",
    "SupervisorAgent",
    "StartAgent",
    "EndAgent",
    "FluidAgent",
    "configure_logging",
    "Message",
    "UserMessage",
    "AssistantMessage",
    "SystemMessage",
    "Workflow",
    "ChainOfThoughtWorkflow",
    "TreeOfThoughtWorkflow",
    "FluidChainOfThoughtWorkflow",
    "Model",
    "OllamaModel",
    "OpenAIModel",
    "GroqModel",
]
