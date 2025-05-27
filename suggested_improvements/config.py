# Example: Type-safe configuration with Pydantic
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    SIMPLE = "simple"
    SELECTOR = "selector"


class WorkflowType(str, Enum):
    CHAIN_OF_THOUGHT = "cot"
    TREE_OF_THOUGHT = "tot" 
    FLUID = "fluid"
    AUTO_SELECT = "auto"


class AgentConfig(BaseModel):
    """Type-safe agent configuration."""
    name: str
    prompt: str
    input: str = ""
    description: str = ""
    extra_info: str = ""
    agent_type: Optional[AgentType] = None
    options: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class WorkflowConfig(BaseModel):
    """Type-safe workflow configuration."""
    type: WorkflowType
    validate_output: bool = False
    max_attempts: int = 3
    max_previous_messages: Optional[int] = None