import logging
from dataclasses import dataclass, field
from typing import Optional, Required

from .llms import Model, OllamaModel
from .memory import JSONShortTermMemory, ShortTermMemory
from .messages import SystemMessage
from .tools import AIFunctionTool


@dataclass
class Agent:
    name: str = "Agent"
    prompt: str = (
        "You are an AI agent called Joaquin, capable of responding to questions"
    )
    model: Model = field(default_factory=OllamaModel)
    path: str = field(default="memory.json")
    tools: Optional[list[AIFunctionTool]] = None
    tools_choice: Optional[str] = None
    memory_mime_type: str = field(default_factory=".json")

    def __validate_mime_type(self):
        mime_types = [".json"]
        if self.memory_mime_type not in mime_types:
            raise ValueError(f"mime type is not suported: {self.memory_mime_type}")

    def __post_init__(self):
        valid_mime = self.__validate_mime_type()
        self.short_term_memory: ShortTermMemory
        if valid_mime:
            if self.memory_mime_type == ".json":
                self.short_term_memory = JSONShortTermMemory(self.path)
        else:
            raise ValueError(f"mime type is not suported: {self.memory_mime_type}")
        logging.info(f"{self.name} initialized with prompt: {self.prompt}")

    def respond(self, message: str) -> str:
        self.short_term_memory.remember("user", message)

        messages = [
            SystemMessage(role="system", content=self.prompt),
        ] + self.short_term_memory.recall()

        response = self.model.generate_response(
            messages, tools=self.tools, tool_choice=self.tools_choice
        )

        self.short_term_memory.remember("assistant", response)
        return response

    def memory(self) -> list:
        return self.short_term_memory.recall()
