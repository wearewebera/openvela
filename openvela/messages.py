from typing import Literal, TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class UserMessage(Message):
    role: str = "user"
    content: str


class AssistantMessage(Message):
    role: str = "assistant"
    content: str


class SystemMessage(Message):
    role: str = "system"
    content: str
