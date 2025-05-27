"""Core types and enums for OpenVela."""

from enum import Enum


class AgentType(str, Enum):
    """Types of agents supported by OpenVela."""
    SIMPLE = "simple"
    SELECTOR = "selector"
    FLUID = "fluid"
    SQL = "sql"


class WorkflowType(str, Enum):
    """Types of workflows supported by OpenVela."""
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHT = "tot" 
    FLUID = "fluid"
    AUTO_SELECT = "auto_select"


class MemoryType(str, Enum):
    """Types of memory supported by OpenVela."""
    JSON = "json"
    SHORT_TERM = "short_term"
    WORKFLOW = "workflow"


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    GROQ = "groq"
    OLLAMA = "ollama"