"""Factory classes for creating OpenVela components."""

from typing import Dict, Any, Type, List, Optional, Union
import importlib

from .config import AgentConfig, WorkflowConfig, ModelConfig, SQLAgentConfig, FluidAgentConfig
from .protocols import AgentProtocol, WorkflowProtocol, ModelProtocol, MemoryProtocol
from .types import AgentType, WorkflowType, ModelProvider
from .exceptions import FactoryError, ConfigurationError


class AgentFactory:
    """Factory for creating agents with proper configuration validation."""
    
    _agent_types: Dict[AgentType, Type[AgentProtocol]] = {}
    
    @classmethod
    def register_agent_type(cls, agent_type: AgentType, agent_class: Type[AgentProtocol]) -> None:
        """Register a new agent type."""
        cls._agent_types[agent_type] = agent_class
    
    @classmethod
    def create_agent(
        cls,
        agent_config: Union[AgentConfig, Dict[str, Any]],
        model: ModelProtocol,
        memory: Optional[MemoryProtocol] = None
    ) -> AgentProtocol:
        """
        Create an agent from configuration.
        
        Args:
            agent_config: Agent configuration (Pydantic model or dict)
            model: Language model instance
            memory: Optional memory instance
            
        Returns:
            Configured agent instance
            
        Raises:
            FactoryError: If agent type not supported
            ConfigurationError: If configuration is invalid
        """
        # Handle both dict and Pydantic model inputs
        if isinstance(agent_config, dict):
            agent_config = cls._create_config_from_dict(agent_config)
        
        agent_type = agent_config.agent_type or AgentType.SIMPLE
        
        if agent_type not in cls._agent_types:
            raise FactoryError(f"Unsupported agent type: {agent_type}")
        
        try:
            agent_class = cls._agent_types[agent_type]
            agent = agent_class(config=agent_config, model=model)
            
            if memory:
                agent.set_memory(memory)
                
            return agent
        except Exception as e:
            raise FactoryError(f"Failed to create agent {agent_config.name}: {e}") from e
    
    @classmethod
    def _create_config_from_dict(cls, agent_dict: Dict[str, Any]) -> AgentConfig:
        """Create appropriate config type from dictionary."""
        agent_type = agent_dict.get("agent_type", AgentType.SIMPLE)
        
        try:
            if agent_type == AgentType.SQL:
                return SQLAgentConfig(**agent_dict)
            elif agent_type == AgentType.FLUID:
                return FluidAgentConfig(**agent_dict)
            else:
                return AgentConfig(**agent_dict)
        except Exception as e:
            raise ConfigurationError(f"Invalid agent configuration: {e}") from e
    
    @classmethod
    def create_from_dict(
        cls,
        agent_dict: Dict[str, Any],
        model: ModelProtocol,
        memory: Optional[MemoryProtocol] = None
    ) -> AgentProtocol:
        """Create agent from dictionary configuration."""
        config = cls._create_config_from_dict(agent_dict)
        return cls.create_agent(config, model, memory)


class WorkflowFactory:
    """Factory for creating workflows with proper dependency injection."""
    
    _workflow_types: Dict[WorkflowType, Type[WorkflowProtocol]] = {}
    
    @classmethod
    def register_workflow_type(cls, workflow_type: WorkflowType, workflow_class: Type[WorkflowProtocol]) -> None:
        """Register a new workflow type."""
        cls._workflow_types[workflow_type] = workflow_class
    
    @classmethod
    def create_workflow(
        cls,
        workflow_config: Union[WorkflowConfig, Dict[str, Any]],
        agents: List[AgentProtocol],
        supervisor: AgentProtocol,
        **kwargs: Any
    ) -> WorkflowProtocol:
        """
        Create a workflow from configuration.
        
        Args:
            workflow_config: Workflow configuration (Pydantic model or dict)
            agents: List of agents for the workflow
            supervisor: Supervisor agent
            **kwargs: Additional workflow-specific arguments
            
        Returns:
            Configured workflow instance
            
        Raises:
            FactoryError: If workflow type not supported
            ConfigurationError: If configuration is invalid
        """
        # Handle both dict and Pydantic model inputs
        if isinstance(workflow_config, dict):
            try:
                workflow_config = WorkflowConfig(**workflow_config)
            except Exception as e:
                raise ConfigurationError(f"Invalid workflow configuration: {e}") from e
        
        if workflow_config.type not in cls._workflow_types:
            raise FactoryError(f"Unsupported workflow type: {workflow_config.type}")
        
        try:
            workflow_class = cls._workflow_types[workflow_config.type]
            return workflow_class(
                config=workflow_config,
                agents=agents,
                supervisor=supervisor,
                **kwargs
            )
        except Exception as e:
            raise FactoryError(f"Failed to create workflow {workflow_config.type}: {e}") from e


class ModelFactory:
    """Factory for creating language model instances."""
    
    _model_providers: Dict[ModelProvider, Type[ModelProtocol]] = {}
    
    @classmethod
    def register_model_provider(cls, provider: ModelProvider, model_class: Type[ModelProtocol]) -> None:
        """Register a new model provider."""
        cls._model_providers[provider] = model_class
    
    @classmethod
    def create_model(
        cls,
        model_config: Union[ModelConfig, Dict[str, Any]]
    ) -> ModelProtocol:
        """
        Create a model from configuration.
        
        Args:
            model_config: Model configuration (Pydantic model or dict)
            
        Returns:
            Configured model instance
            
        Raises:
            FactoryError: If provider not supported
            ConfigurationError: If configuration is invalid
        """
        # Handle both dict and Pydantic model inputs
        if isinstance(model_config, dict):
            try:
                model_config = ModelConfig(**model_config)
            except Exception as e:
                raise ConfigurationError(f"Invalid model configuration: {e}") from e
        
        if model_config.provider not in cls._model_providers:
            raise FactoryError(f"Unsupported model provider: {model_config.provider}")
        
        try:
            model_class = cls._model_providers[model_config.provider]
            return model_class(
                model=model_config.model_name,
                api_key=model_config.api_key,
                base_url=model_config.base_url,
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens
            )
        except Exception as e:
            raise FactoryError(f"Failed to create model {model_config.provider}: {e}") from e


def auto_register_factories():
    """Automatically register available agent types, workflows, and models."""
    try:
        # Import and register existing classes
        from ..agents import Agent, StartAgent, EndAgent, SupervisorAgent, FluidAgent, SQLAgent
        from ..workflows import ChainOfThoughtWorkflow, TreeOfThoughtWorkflow, FluidChainOfThoughtWorkflow
        from ..llms import OpenAIModel, GroqModel, OllamaModel
        
        # Register agent types
        AgentFactory.register_agent_type(AgentType.SIMPLE, Agent)
        AgentFactory.register_agent_type(AgentType.SELECTOR, SupervisorAgent)
        AgentFactory.register_agent_type(AgentType.FLUID, FluidAgent)
        AgentFactory.register_agent_type(AgentType.SQL, SQLAgent)
        
        # Register workflow types
        WorkflowFactory.register_workflow_type(WorkflowType.CHAIN_OF_THOUGHT, ChainOfThoughtWorkflow)
        WorkflowFactory.register_workflow_type(WorkflowType.TREE_OF_THOUGHT, TreeOfThoughtWorkflow)
        WorkflowFactory.register_workflow_type(WorkflowType.FLUID, FluidChainOfThoughtWorkflow)
        
        # Register model providers
        ModelFactory.register_model_provider(ModelProvider.OPENAI, OpenAIModel)
        ModelFactory.register_model_provider(ModelProvider.GROQ, GroqModel)
        ModelFactory.register_model_provider(ModelProvider.OLLAMA, OllamaModel)
        
    except ImportError:
        # Some modules might not be available during initial setup
        pass