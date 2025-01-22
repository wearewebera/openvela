from typing import Literal, Optional, TypedDict


class Message(TypedDict):
    """
    Represents a generic message structure with a role and content.
    """

    agent_name: Optional[str]
    role: Literal["user", "assistant", "system"]
    content: str


class UserMessage(Message):
    """
    Represents a message from the user.
    """

    role: str = "user"
    content: str


class AssistantMessage(Message):
    """
    Represents a message from the assistant.
    """

    role: str = "assistant"
    content: str


class SystemMessage(Message):
    """
    Represents a system-generated message.
    """

    role: str = "system"
    content: str
