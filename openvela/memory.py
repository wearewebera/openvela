import json

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Type


class ShortTermMemory(ABC):
    @abstractmethod
    def remember(self, role, content):
        pass

    @abstractmethod
    def recall(self) -> list:
        pass


class MemoryManager:
    def __init__(
        self, memory_class: Type[ShortTermMemory], initial_prompt: str, *args, **kwargs
    ):
        self.memory = memory_class(*args, **kwargs)
        self.current_prompt = initial_prompt
        self.memory.clear_memory()
        self.memory.remember("system", f"Prompt: {self.current_prompt}")

    def update_prompt(self, new_prompt: str):
        self.current_prompt = new_prompt
        self.memory.remember("system", f"Prompt changed to: {self.current_prompt}")

    def add_user_message(self, message: str):
        self.memory.remember("user", message)

    def add_assistant_message(self, message: str):
        self.memory.remember("assistant", message)

    def get_conversation_history(self):
        return self.memory.recall()


@dataclass
class SimpleShortTermMemory(ShortTermMemory):
    messages: list = field(default_factory=list)

    def remember(self, role, content):
        self.messages.append({"role": role, "content": content})

    def recall(self):
        return self.messages


@dataclass
class JSONShortTermMemory(ShortTermMemory):
    file_path: str = "memory.json"
    messages: list = field(default_factory=list)

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
        return self.messages
