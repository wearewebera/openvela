import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type

from .messages import Message, SystemMessage


class ShortTermMemory(ABC):
    @abstractmethod
    def __init__(self, prompt, *args, **kwargs):
        self.prompt = prompt

    @abstractmethod
    def remember(self, role, content): ...

    @abstractmethod
    def recall(self) -> list: ...

    @abstractmethod
    def clear_memory(self): ...


@dataclass
class SimpleShortTermMemory(ShortTermMemory):
    prompt: str
    messages: list = field(default_factory=list)

    def remember(self, role, content):
        self.messages.append(Message(role=role, content=content))

    def recall(self):
        self.messages.insert(0, {SystemMessage(role="system", content=self.prompt)})
        return self.messages

    def clear_memory(self):
        self.messages = []


@dataclass
class JSONShortTermMemory(ShortTermMemory):
    prompt: str
    messages: list = field(default_factory=list)
    file_path: str = "memory.json"

    def __post_init__(self):
        self._load_data()

    def _load_data(self):
        try:
            with open(self.file_path, "r") as file:
                self.messages = json.load(file)
        except FileNotFoundError:
            self.messages = []

    def _save_data(self):
        with open(self.file_path, "w") as file:
            json.dump(self.messages, file)

    def remember(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._save_data()

    def recall(self):
        self.messages.insert(0, {"role": "system", "content": self.prompt})
        return self.messages

    def clear_memory(self):
        self.messages = []
        self._save_data()


class MemoryManager:
    def __init__(
        self, memory_class: Type[ShortTermMemory], prompt: str, *args, **kwargs
    ):
        self.memory = memory_class(prompt, *args, **kwargs)

    def add_user_message(self, message: str):
        self.memory.remember("user", message)

    def add_assistant_message(self, message: str):
        self.memory.remember("assistant", message)

    def get_conversation_history(self):
        return self.memory.recall()


def test_memory():
    memory = MemoryManager(
        memory_class=SimpleShortTermMemory, prompt="Welcome to the chatbot!"
    )
    memory.add_user_message("Hello")
    memory.add_assistant_message("Hi there!")
    memory.add_user_message("How are you?")
    from pprint import pprint

    pprint(memory.get_conversation_history())

    json_memory = MemoryManager(
        memory_class=JSONShortTermMemory, prompt="Welcome to the chatbot!"
    )
    json_memory.add_user_message("Hello")
    json_memory.add_assistant_message("Hi there!")
    json_memory.add_user_message("How are you?")
    from pprint import pprint

    pprint(json_memory.get_conversation_history())


if __name__ == "__main__":
    test_memory()
