from abc import ABC, abstractmethod
from typing import Optional, Required, TypedDict


class Tool(ABC):
    """
    Abstract base class representing a generic tool.
    Subclasses must implement the `use` method to define tool-specific actions.
    """

    @abstractmethod
    def use(self):
        """
        Abstract method to execute the tool's primary function.

        Should be implemented by subclasses to perform specific actions.
        """
        pass


class OpenAIFunction(TypedDict):
    """
    Defines the structure for an OpenAI function tool.
    """

    name: Required[str]
    description: Optional[str]
    parameters: Optional[dict[str, str]]
    strict: Optional[bool]


class AIFunctionTool(TypedDict):
    """
    Defines the structure for an AI function tool, encapsulating its type and associated OpenAI function.
    """

    type: Required[str]
    function: OpenAIFunction
