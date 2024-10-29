import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from messages import Message


class MemoryFormat(ABC):
    @abstractmethod
    def save(self, data: dict, file_path: str):
        pass

    @abstractmethod
    def load(self, file_path: str) -> dict:
        pass


class JsonMemoryFormat(MemoryFormat):
    def save(self, data: dict, file_path: str):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, file_path: str) -> dict:
        with open(file_path, "r") as f:
            return json.load(f)


@dataclass
class ShortTermMemory(ABC):
    prompt: str

    @abstractmethod
    def remember(self, role: str, content: str):
        pass

    @abstractmethod
    def recall(self) -> List[Dict[str, str]]:
        pass

    @abstractmethod
    def clear_memory(self):
        pass


@dataclass
class JsonShortTermMemory(ShortTermMemory):
    messages: List[Dict[str, str]] = field(default_factory=list)
    file_path: str = "memory.json"
    memory_format: MemoryFormat = field(default_factory=JsonMemoryFormat)

    def __post_init__(self):
        self._load_data()

    def _load_data(self):
        if os.path.exists(self.file_path):
            data = self.memory_format.load(self.file_path)
            self.messages = data.get("messages", [])
        else:
            self.messages = []

    def _save_data(self):
        data = {"messages": self.messages}
        self.memory_format.save(data, self.file_path)

    def remember(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._save_data()

    def recall(self) -> List[Dict[str, str]]:
        return [{"role": "system", "content": self.prompt}] + self.messages

    def clear_memory(self):
        self.messages = []
        self._save_data()


@dataclass
class WorkflowMemory:
    messages: List[Dict[str, str]] = field(default_factory=list)
    file_path: str = ".openvela/workflow_memory.json"
    memory_format: MemoryFormat = field(default_factory=JsonMemoryFormat)

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.load()

    def add_message(self, role: str, content: str):
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.save()

    def save(self):
        data = {"messages": self.messages}
        self.memory_format.save(data, self.file_path)

    def load(self):
        if os.path.exists(self.file_path):
            data = self.memory_format.load(self.file_path)
            self.messages = data.get("messages", [])
        else:
            self.messages = []
        return self.messages


@dataclass
class AgentMemory:
    agents_info: Dict[str, str] = field(default_factory=dict)
    file_path: str = ".openvela/agents_info.json"
    memory_format: MemoryFormat = field(default_factory=JsonMemoryFormat)

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.load()

    def add_agent_info(self, agent_name: str, description: str):
        self.agents_info[agent_name] = description
        self.save()

    def save(self):
        data = {"agents_info": self.agents_info}
        self.memory_format.save(data, self.file_path)

    def load(self):
        if os.path.exists(self.file_path):
            data = self.memory_format.load(self.file_path)
            self.agents_info = data.get("agents_info", {})
        else:
            self.agents_info = {}


@dataclass
class JsonReader:
    file_path: str

    def json_to_dict(self) -> Dict[str, Any]:
        memory_format = JsonMemoryFormat()
        return memory_format.load(self.file_path)
