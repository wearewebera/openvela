import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

from openvela.messages import Message


class MemoryFormat(ABC):
    """
    Abstract base class defining the interface for memory formatting.
    Subclasses must implement methods to save and load data in specific formats.
    """

    @abstractmethod
    def save(self, data: dict, file_path: str):
        """
        Saves the provided data to the specified file path.

        Args:
            data (dict): The data to be saved.
            file_path (str): The file system path where data will be saved.
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> dict:
        """
        Loads and returns data from the specified file path.

        Args:
            file_path (str): The file system path to load data from.

        Returns:
            dict: The loaded data.
        """
        pass


class JsonMemoryFormat(MemoryFormat):
    """
    Concrete implementation of MemoryFormat for handling JSON data.
    Provides methods to save and load data in JSON format.
    """

    def save(self, data: dict, file_path: str):
        """
        Saves the provided dictionary data as a JSON file.

        Args:
            data (dict): The data to be saved.
            file_path (str): The file system path where JSON data will be saved.
        """
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, file_path: str) -> dict:
        """
        Loads and returns data from a JSON file.

        Args:
            file_path (str): The file system path to load JSON data from.

        Returns:
            dict: The loaded JSON data.
        """
        with open(file_path, "r") as f:
            return json.load(f)


@dataclass
class ShortTermMemory(ABC):
    """
    Abstract base class representing short-term memory for agents.
    Defines methods for remembering, recalling, and clearing messages.
    """

    prompt: str

    @abstractmethod
    def remember(self, role: str, content: str):
        """
        Stores a message with the specified role and content.

        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        pass

    @abstractmethod
    def recall(self) -> List[Dict[str, str]]:
        """
        Retrieves all stored messages.

        Returns:
            List[Dict[str, str]]: A list of messages with roles and content.
        """
        pass

    @abstractmethod
    def clear_memory(self):
        """
        Clears all stored messages from memory.
        """
        pass


@dataclass
class JsonShortTermMemory(ShortTermMemory):
    """
    Concrete implementation of ShortTermMemory using JSON for storage.
    Maintains a list of messages and persists them to a JSON file.
    """

    messages: List[Dict[str, str]] = field(default_factory=list)
    file_path: str = "memory.json"
    memory_format: MemoryFormat = field(default_factory=JsonMemoryFormat)

    def __post_init__(self):
        """
        Post-initialization processing to load existing memory data if available.
        """
        self._load_data()

    def _load_data(self):
        """
        Loads existing messages from the JSON memory file.
        Initializes messages to an empty list if the file does not exist.
        """
        if os.path.exists(self.file_path):
            data = self.memory_format.load(self.file_path)
            self.messages = data.get("messages", [])
        else:
            self.messages = []

    def _save_data(self):
        """
        Saves the current list of messages to the JSON memory file.
        """
        data = {"messages": self.messages}
        self.memory_format.save(data, self.file_path)

    def remember(self, role: str, content: str):
        """
        Adds a new message to memory and persists the updated memory.

        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
        """
        self.messages.append({"role": role, "content": content})
        self._save_data()

    def recall(self) -> List[Dict[str, str]]:
        """
        Retrieves all messages, including the system prompt.

        Returns:
            List[Dict[str, str]]: A list of messages starting with the system prompt.
        """
        return [{"role": "system", "content": self.prompt}] + self.messages

    def clear_memory(self):
        """
        Clears all messages from memory and persists the cleared state.
        """
        self.messages = []
        self._save_data()


@dataclass
class WorkflowMemory:
    """
    Manages workflow-specific messages, allowing for the addition, saving, and loading of messages.
    Utilizes a JSON memory format for persistence.
    """

    memory_id: str
    messages: List[Dict[str, str]] = field(default_factory=list)

    memory_format: MemoryFormat = field(default_factory=JsonMemoryFormat)

    def __post_init__(self):
        """
        Ensures the directory for the memory file exists and loads existing messages.
        """
        self.file_path = f".openvela/workflows/{self.memory_id}.json"
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.load()

    def add_message(self, role: str, content: str):
        """
        Adds a new message to the workflow memory and saves the updated memory.

        Args:
            role (str): The role of the message sender.
            content (str): The content of the message.
        """
        message = {"role": role, "content": content}
        self.messages.append(message)
        self.save()

    def save(self):
        """
        Saves the current list of messages to the workflow memory file.
        """
        data = {"messages": self.messages}
        self.memory_format.save(data, self.file_path)

    def load(self):
        """
        Loads existing messages from the workflow memory file.
        Initializes messages to an empty list if the file does not exist.

        Returns:
            List[Dict[str, str]]: The loaded messages.
        """
        if os.path.exists(self.file_path):
            data = self.memory_format.load(self.file_path)
            self.messages = data.get("messages", [])
        else:
            self.messages = []
        return self.messages

    def clear_memory(self):
        """
        Clears all messages from the workflow memory and persists the cleared state.
        """
        self.messages = []
        self.save()


@dataclass
class AgentMemory:
    """
    Manages information about agents, such as their names and descriptions.
    Utilizes a JSON memory format for persistence.
    """

    agents_info: Dict[str, str] = field(default_factory=dict)
    file_path: str = ".openvela/agents_info.json"
    memory_format: MemoryFormat = field(default_factory=JsonMemoryFormat)

    def __post_init__(self):
        """
        Ensures the directory for the agents info file exists and loads existing agent information.
        """
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.load()

    def add_agent_info(self, agent_name: str, description: str):
        """
        Adds or updates the description of an agent and saves the updated information.

        Args:
            agent_name (str): The name of the agent.
            description (str): The description of the agent.
        """
        self.agents_info[agent_name] = description
        self.save()

    def save(self):
        """
        Saves the current agents information to the agents info file.
        """
        data = {"agents_info": self.agents_info}
        self.memory_format.save(data, self.file_path)

    def load(self):
        """
        Loads existing agents information from the agents info file.
        Initializes agents_info to an empty dictionary if the file does not exist.
        """
        if os.path.exists(self.file_path):
            data = self.memory_format.load(self.file_path)
            self.agents_info = data.get("agents_info", {})
        else:
            self.agents_info = {}


@dataclass
class JsonReader:
    """
    Utility class for reading JSON files and converting them to Python dictionaries.
    """

    file_path: str

    def json_to_dict(self) -> Dict[str, Any]:
        """
        Loads a JSON file and returns its content as a dictionary.

        Returns:
            Dict[str, Any]: The content of the JSON file as a dictionary.
        """
        memory_format = JsonMemoryFormat()
        return memory_format.load(self.file_path)
