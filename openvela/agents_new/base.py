"""Base agent implementation with improved type safety."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from uuid import uuid4

from ..core import (
    AgentConfig, 
    AgentError, 
    ConfigurationError,
    MemoryProtocol, 
    ModelProtocol, 
    AgentProtocol
)
from ..tools import AIFunctionTool


@dataclass
class BaseAgent(ABC):
    """
    Abstract base class for all agents with improved type safety and error handling.
    """
    
    config: AgentConfig
    model: ModelProtocol
    memory: Optional[MemoryProtocol] = None
    tools: Optional[List[AIFunctionTool]] = None
    tools_choice: Optional[str] = None
    _id: str = field(default_factory=lambda: str(uuid4()))
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._setup_logging()
    
    def _validate_config(self) -> None:
        """Validate agent configuration."""
        if not self.config.name:
            raise ConfigurationError("Agent name cannot be empty")
        if not self.config.prompt:
            raise ConfigurationError("Agent prompt cannot be empty")
    
    def _setup_logging(self) -> None:
        """Set up agent-specific logging."""
        self.logger = logging.getLogger(f"openvela.agent.{self.config.name}")
        self.logger.debug(f"Agent {self.config.name} initialized")
    
    @property 
    def name(self) -> str:
        """Agent name from configuration."""
        return self.config.name
    
    @property
    def prompt(self) -> str:
        """Agent prompt from configuration."""
        return self.config.prompt
    
    @property
    def agent_id(self) -> str:
        """Unique agent identifier."""
        return self._id
    
    def set_memory(self, memory: MemoryProtocol) -> None:
        """Inject memory dependency."""
        self.memory = memory
        self.logger.debug(f"Memory set for agent {self.name}")
    
    def generate(self, **kwargs: Any) -> str:
        """
        Generate agent response with proper error handling.
        
        Args:
            **kwargs: Additional arguments for generation
            
        Returns:
            Generated response string
            
        Raises:
            AgentError: If generation fails
        """
        if not self.memory:
            raise AgentError(f"No memory set for agent {self.name}", agent_name=self.name)
        
        try:
            self.logger.debug(f"Generating response for agent {self.name}")
            messages = self._prepare_messages(**kwargs)
            response = self._call_model(messages, **kwargs)
            self._store_interaction(response)
            self.logger.debug(f"Generated response for agent {self.name}: {len(response)} characters")
            return response
        except Exception as e:
            self.logger.error(f"Generation failed for agent {self.name}: {e}")
            raise AgentError(f"Generation failed for agent {self.name}: {e}", agent_name=self.name) from e
    
    def _prepare_messages(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Prepare messages for model input."""
        messages = self.memory.load() if self.memory else []
        
        # Apply max_previous_messages limit
        max_msgs = kwargs.get("max_previous_messages")
        if max_msgs:
            messages = messages[-max_msgs:]
        
        # Add system prompt
        messages.insert(0, {"role": "system", "content": self.prompt})
        
        # Add user input
        user_input = self._build_user_input()
        messages.append({"role": "user", "content": user_input})
        
        return messages
    
    def _build_user_input(self) -> str:
        """Build user input from configuration."""
        input_parts = [self.config.input]
        if self.config.extra_info:
            input_parts.append(self.config.extra_info)
        return "\n\n".join(filter(None, input_parts))
    
    def _call_model(self, messages: List[Dict[str, Any]], **kwargs: Any) -> str:
        """Call the language model with prepared messages."""
        # Merge agent options with kwargs
        model_kwargs = {**self.config.options, **kwargs}
        
        # Remove agent-specific kwargs that shouldn't go to the model
        model_kwargs.pop("max_previous_messages", None)
        
        return self.model.generate_response(
            messages,
            tools=self.tools,
            tool_choice=self.tools_choice,
            **model_kwargs
        )
    
    def _store_interaction(self, response: str) -> None:
        """Store the interaction in memory."""
        if self.memory:
            self.memory.add_message("", "user", self.config.input)
            self.memory.add_message(self.name, "assistant", response)
    
    @abstractmethod
    def get_agent_type(self) -> str:
        """Return the agent type identifier."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        return {
            "id": self._id,
            "name": self.name,
            "type": self.get_agent_type(),
            "config": self.config.dict(),
        }
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on agent ID."""
        if not isinstance(other, BaseAgent):
            return NotImplemented
        return self._id == other._id
    
    def __hash__(self) -> int:
        """Hash based on agent ID."""
        return hash(self._id)
    
    def __repr__(self) -> str:
        """String representation of agent."""
        return f"{self.__class__.__name__}(name='{self.name}', id='{self._id[:8]}...')"