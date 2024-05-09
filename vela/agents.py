import logging
from dataclasses import dataclass, field

from llms import Model, OllamaModel
from memory import ShortTermMemory, JSONShortTermMemory


@dataclass
class Agent:
    name: str = "Agent"
    prompt: str = "You are an AI agent capable of responding to questions"
    model: Model = field(default_factory=OllamaModel)

    def __post_init__(self):
        self.short_term_memory: ShortTermMemory = JSONShortTermMemory("memory.json")
        self.short_term_memory.remember("system", self.prompt)
        logging.info(f"{self.name} initialized with prompt: {self.prompt}")

    def respond(self, message: str) -> str:
        self.short_term_memory.remember("user", message)
        response = self.model.generate_response(self.short_term_memory.recall())
        self.short_term_memory.remember("assistant", response)
        return response

    def memory(self) -> list:
        return self.short_term_memory.recall()
