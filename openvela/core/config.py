"""Configuration models using Pydantic for type safety."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator

from .types import AgentType, WorkflowType, ModelProvider


class AgentConfig(BaseModel):
    """Type-safe agent configuration."""
    
    name: str = Field(..., min_length=1, description="Agent name")
    prompt: str = Field(..., min_length=1, description="Agent prompt/instructions")
    input: str = Field(default="", description="Initial input for the agent")
    description: str = Field(default="", description="Agent description")
    extra_info: str = Field(default="", description="Additional context")
    agent_type: Optional[AgentType] = Field(default=AgentType.SIMPLE, description="Type of agent")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")
    
    class Config:
        use_enum_values = True
        extra = "forbid"
    
    @validator('name')
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Agent name cannot be empty or whitespace')
        return v.strip()
    
    @validator('prompt')
    def prompt_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Agent prompt cannot be empty or whitespace')
        return v.strip()


class WorkflowConfig(BaseModel):
    """Type-safe workflow configuration."""
    
    type: WorkflowType = Field(..., description="Type of workflow")
    validate_output: bool = Field(default=False, description="Enable output validation")
    max_attempts: int = Field(default=3, ge=1, le=10, description="Maximum validation attempts")
    max_previous_messages: Optional[int] = Field(default=None, ge=1, description="Limit on previous messages")
    
    class Config:
        use_enum_values = True
        extra = "forbid"


class ModelConfig(BaseModel):
    """Type-safe model configuration."""
    
    provider: ModelProvider = Field(..., description="Model provider")
    model_name: str = Field(..., min_length=1, description="Name of the model")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Base URL for the provider")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens to generate")
    
    class Config:
        use_enum_values = True
        extra = "forbid"


class SQLAgentConfig(AgentConfig):
    """Configuration specific to SQL agents."""
    
    agent_type: AgentType = Field(default=AgentType.SQL, description="Must be SQL type")
    example_queries: List[Dict[str, str]] = Field(default_factory=list, description="Example queries")
    sql_dialect: str = Field(default="postgresql", description="SQL dialect")
    sqlalchemy_engine_url: str = Field(..., description="SQLAlchemy engine URL")
    database_structure: str = Field(default="", description="Database structure description")
    formatter_prompt: str = Field(default="", description="Result formatting prompt")
    
    @validator('agent_type')
    def must_be_sql_type(cls, v):
        if v != AgentType.SQL:
            raise ValueError('SQLAgentConfig must have agent_type=SQL')
        return v


class FluidAgentConfig(AgentConfig):
    """Configuration specific to fluid agents."""
    
    agent_type: AgentType = Field(default=AgentType.FLUID, description="Must be fluid type")
    max_agents: int = Field(default=5, ge=1, le=20, description="Maximum number of generated agents")
    
    @validator('agent_type')
    def must_be_fluid_type(cls, v):
        if v != AgentType.FLUID:
            raise ValueError('FluidAgentConfig must have agent_type=FLUID')
        return v