# Example: Using protocols for better interface definition
from typing import Protocol, Any, Dict, List, Optional, runtime_checkable
from abc import abstractmethod


@runtime_checkable
class MemoryProtocol(Protocol):
    """Protocol for memory implementations."""
    
    def add_message(self, agent_name: str, role: str, content: str) -> None:
        """Add a message to memory."""
        ...
    
    def load(self) -> List[Dict[str, Any]]:
        """Load messages from memory."""
        ...
    
    def clear(self) -> None:
        """Clear memory."""
        ...


@runtime_checkable  
class ModelProtocol(Protocol):
    """Protocol for language model implementations."""
    
    def generate_response(
        self,
        messages: List[Dict[str, Any]], 
        **kwargs: Any
    ) -> str:
        """Generate a response from the model."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for agent implementations."""
    
    name: str
    
    def generate(self, **kwargs: Any) -> str:
        """Generate agent response."""
        ...
    
    def set_memory(self, memory: MemoryProtocol) -> None:
        """Set agent memory."""
        ...


@runtime_checkable
class WorkflowProtocol(Protocol):
    """Protocol for workflow implementations."""
    
    def run(self, **kwargs: Any) -> tuple[str, str]:  # (output, memory_id)
        """Execute the workflow."""
        ...