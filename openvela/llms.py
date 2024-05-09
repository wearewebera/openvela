import logging

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Literal, Sequence

from ollama import Client, Message


class Model(ABC):
    @abstractmethod
    def generate_response(self, messages: list[dict]) -> str:
        pass


@dataclass
class OllamaModel(Model):
    host: str = "localhost"
    port: int = 11434
    client: Client = field(init=False)
    model: str = "mistral"

    def __post_init__(self):
        self.client = Client(f"http://{self.host}:{self.port}")
        logging.info(f"OllamaModel initialized with host: {self.host}")

    @staticmethod
    def _convert_to_messages(dict_list: list[dict]) -> Sequence[Message]:
        def validate_role(role: str) -> Literal["user", "assistant", "system"]:
            allowed_roles = ("user", "assistant", "system")
            if role not in allowed_roles:
                raise ValueError(
                    f"Invalid role: {role}. Allowed roles are {allowed_roles}."
                )
            return role

        messages: Sequence[Message] = []
        for item in dict_list:
            role = validate_role(item["role"])
            message = Message(
                role=role,
                content=item["content"],
                images=item.get("images", []),
            )
            messages.append(message)
        return messages

    def generate_response(self, messages: list[dict]) -> str:
        converted_messages = self._convert_to_messages(messages)
        response = self.client.chat(model=self.model, messages=converted_messages)
        if isinstance(response, Mapping):
            response = iter([response])
        response_mapping: Mapping[str, Any] = next(response)
        return response_mapping["message"]["content"]
