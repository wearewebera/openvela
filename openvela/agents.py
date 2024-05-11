import logging
from dataclasses import dataclass, field

from llms import Model, OllamaModel
from memory import ShortTermMemory, JSONShortTermMemory


@dataclass
class Agent:
    name: str = "Agent"
    prompt: str = (
        "You are an AI agent called Joaquin, capable of responding to questions"
    )
    model: Model = field(default_factory=OllamaModel)

    def __post_init__(self):
        self.short_term_memory: ShortTermMemory = JSONShortTermMemory("memory.json")
        logging.info(f"{self.name} initialized with prompt: {self.prompt}")

    def respond(self, message: str) -> str:
        self.short_term_memory.remember("user", message)

        messages = [
            {"role": "system", "content": self.prompt}
        ] + self.short_term_memory.recall()

        response = self.model.generate_response(messages)

        self.short_term_memory.remember("assistant", response)
        return response

    def memory(self) -> list:
        return self.short_term_memory.recall()
