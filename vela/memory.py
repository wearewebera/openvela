import json

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ShortTermMemory(ABC):
    @abstractmethod
    def remember(self, role, content):
        pass

    @abstractmethod
    def recall(self) -> list:
        pass


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
