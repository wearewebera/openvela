# Example: Improved Agent implementation
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from uuid import uuid4

from .config import AgentConfig
from .protocols import MemoryProtocol, ModelProtocol
from .exceptions import AgentError, ConfigurationError


@dataclass
class Agent:
    """
    Improved Agent implementation with type safety and better separation of concerns.
    """
    
    config: AgentConfig
    model: ModelProtocol
    memory: Optional[MemoryProtocol] = None
    tools: Optional[List[Any]] = None
    _id: str = field(default_factory=lambda: str(uuid4()))
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.config.name:
            raise ConfigurationError("Agent name cannot be empty")
        if not self.config.prompt:
            raise ConfigurationError("Agent prompt cannot be empty")
    
    @property 
    def name(self) -> str:
        """Agent name from configuration."""
        return self.config.name
    
    @property
    def prompt(self) -> str:
        """Agent prompt from configuration."""
        return self.config.prompt
    
    def set_memory(self, memory: MemoryProtocol) -> None:
        """Inject memory dependency."""
        self.memory = memory
    
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
            raise AgentError(f"No memory set for agent {self.name}")
        
        try:
            messages = self._prepare_messages(**kwargs)
            response = self.model.generate_response(messages, **kwargs)
            self._store_interaction(response)
            return response
        except Exception as e:
            raise AgentError(f"Generation failed for agent {self.name}: {e}") from e
    
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
    
    def _store_interaction(self, response: str) -> None:
        """Store the interaction in memory."""
        if self.memory:
            self.memory.add_message("", "user", self.config.input)
            self.memory.add_message(self.name, "assistant", response)
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on agent ID."""
        if not isinstance(other, Agent):
            return NotImplemented
        return self._id == other._id