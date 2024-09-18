from abc import ABC, abstractmethod
from typing import Optional, Required, TypedDict


class Tool(ABC):
    @abstractmethod
    def use(self):
        pass


class OpenAIFunction(TypedDict):
    name: Required[str]
    description: Optional[str]
    parameters: Optional[dict[str, str]]
    strict: Optional[bool]


class AIFunctionTool(TypedDict):
    type: Required[str]
    function: OpenAIFunction
