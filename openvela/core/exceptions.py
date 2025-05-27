"""Custom exceptions for OpenVela."""

from typing import Optional, Any


class OpenVelaError(Exception):
    """Base exception for all OpenVela errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigurationError(OpenVelaError):
    """Raised when there's an error in configuration."""
    pass


class AgentError(OpenVelaError):
    """Raised when there's an error with agent operations."""
    
    def __init__(self, message: str, agent_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, kwargs)
        self.agent_name = agent_name


class WorkflowError(OpenVelaError):
    """Raised when there's an error with workflow operations."""
    
    def __init__(self, message: str, workflow_type: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, kwargs)
        self.workflow_type = workflow_type


class ModelError(OpenVelaError):
    """Raised when there's an error with model operations."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(message, kwargs)
        self.model_name = model_name


class MemoryError(OpenVelaError):
    """Raised when there's an error with memory operations."""
    pass


class ValidationError(OpenVelaError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None) -> None:
        super().__init__(message)
        self.field = field
        self.value = value


class FactoryError(OpenVelaError):
    """Raised when factory operations fail."""
    pass