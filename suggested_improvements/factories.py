# Example: Factory patterns for better object creation
from typing import Dict, Any, Type, List
from .config import AgentConfig, WorkflowConfig, AgentType, WorkflowType
from .protocols import AgentProtocol, WorkflowProtocol, ModelProtocol, MemoryProtocol
from .exceptions import FactoryError


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
        agent_config: AgentConfig,
        model: ModelProtocol,
        memory: Optional[MemoryProtocol] = None
    ) -> AgentProtocol:
        """
        Create an agent from configuration.
        
        Args:
            agent_config: Validated agent configuration
            model: Language model instance
            memory: Optional memory instance
            
        Returns:
            Configured agent instance
            
        Raises:
            FactoryError: If agent type not supported
        """
        agent_type = agent_config.agent_type or AgentType.SIMPLE
        
        if agent_type not in cls._agent_types:
            raise FactoryError(f"Unsupported agent type: {agent_type}")
        
        agent_class = cls._agent_types[agent_type]
        agent = agent_class(config=agent_config, model=model)
        
        if memory:
            agent.set_memory(memory)
            
        return agent
    
    @classmethod
    def create_from_dict(
        cls,
        agent_dict: Dict[str, Any],
        model: ModelProtocol,
        memory: Optional[MemoryProtocol] = None
    ) -> AgentProtocol:
        """Create agent from dictionary configuration."""
        try:
            config = AgentConfig(**agent_dict)
            return cls.create_agent(config, model, memory)
        except Exception as e:
            raise FactoryError(f"Failed to create agent from dict: {e}") from e


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
        workflow_config: WorkflowConfig,
        agents: List[AgentProtocol],
        supervisor: AgentProtocol,
        **kwargs: Any
    ) -> WorkflowProtocol:
        """
        Create a workflow from configuration.
        
        Args:
            workflow_config: Validated workflow configuration
            agents: List of agents for the workflow
            supervisor: Supervisor agent
            **kwargs: Additional workflow-specific arguments
            
        Returns:
            Configured workflow instance
            
        Raises:
            FactoryError: If workflow type not supported
        """
        if workflow_config.type not in cls._workflow_types:
            raise FactoryError(f"Unsupported workflow type: {workflow_config.type}")
        
        workflow_class = cls._workflow_types[workflow_config.type]
        return workflow_class(
            agents=agents,
            supervisor=supervisor,
            config=workflow_config,
            **kwargs
        )


# Example registration (would be done in __init__.py files)
# AgentFactory.register_agent_type(AgentType.SIMPLE, SimpleAgent)
# AgentFactory.register_agent_type(AgentType.SELECTOR, SelectorAgent)
# WorkflowFactory.register_workflow_type(WorkflowType.CHAIN_OF_THOUGHT, ChainOfThoughtWorkflow)