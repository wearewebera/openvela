import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, TypedDict

from groq import Groq
from ollama import Client
from openai import OpenAI

from openvela.files import OpenVelaAudioFile, OpenVelaImageFile
from openvela.messages import Message, UserMessage
from openvela.tools import AIFunctionTool


class Model(ABC):
    @abstractmethod
    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[list[AIFunctionTool]] = None,
        tool_choice: Optional[str] = None,
        format: Optional[str] = None,
    ) -> str:
        pass

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
            message = Message(role=role, content=item["content"])
            messages.append(message)
        return messages

    @staticmethod
    def _convert_to_files(files: list[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        converted_files = []
        for file in files:
            if file["type"] == "audio":
                audio_file = OpenVelaAudioFile(file["path"])
                converted_files.append(audio_file.read())
            elif file["type"] == "image":
                image_file = OpenVelaImageFile(file["path"])
                converted_files.append(image_file.read())
            else:
                logging.warning(f"Unknown file type: {file['type']}")
        return converted_files

    @staticmethod
    def _functions_by_choices(
        tools: list[AIFunctionTool], tool_choice: list[str]
    ) -> AIFunctionTool:
        for tool in tools:
            if tool["function"]["name"] == tool_choice:
                return tool
        raise ValueError(f"Tool with name {tool_choice} not found.")


@dataclass
class OllamaModel(Model):
    host: str = "10.50.0.11"
    port: int = 11434
    client: Client = field(init=False)
    model: str = "llama3.1:70b"

    def __post_init__(self):
        self.client = Client(f"http://{self.host}:{self.port}/")
        logging.info(f"OllamaModel initialized with host: {self.host}")

    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[AIFunctionTool] = None,
        tool_choice: Optional[str] = None,
        format: Optional[str] = "",
        options: Optional[Dict[str, Any]] = {"num_ctx": 8192},
    ) -> str:
        converted_messages = self._convert_to_messages(messages)
        selected_tools = (
            self._functions_by_choices(tools, tool_choice) if tools else None
        )
        response = self.client.chat(
            model=self.model,
            messages=converted_messages,
            tools=selected_tools,
            options=options,
            format=format,
        )
        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["message"]["content"]


@dataclass
class OpenAIModel(Model):
    api_key: str
    model: str = "gpt-4o-mini"
    openai: OpenAI = field(init=False)
    transcription_model: str = "whisper-1"

    def __post_init__(self):
        self.openai = OpenAI(self.api_key)

    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[AIFunctionTool] = None,
        tool_choice: Optional[str] = None,
    ) -> str:
        converted_messages = self._convert_to_messages(messages)
        converted_files = self._convert_to_files(files) if files else None
        selected_tools = (
            self._functions_by_choices(tools, tool_choice) if tools else None
        )
        if converted_files:
            for file in converted_files:
                if isinstance(file, bytes) and file["type"] == "audio":
                    audio_transcription = self.transcribe_audio(file)
                    converted_messages.append(UserMessage(content=audio_transcription))
                    response = self.openai.chat.completions.create(
                        model=self.model, messages=converted_messages
                    )
        else:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=converted_messages,
                tools=selected_tools,
                tool_choice=tool_choice,
            )

        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["choices"][0]["message"]["content"]

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        response = self.openai.audio.transcriptions.create(
            model=self.transcription_model, file=audio_bytes
        )
        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["text"]


@dataclass
class GroqModel(Model):
    host: str = "10.50.0.11"
    port: int = 11434
    client: Groq = field(init=False)
    model: str = "llama3.1:70b"

    def __post_init__(self):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        logging.info(f"OllamaModel initialized with host: {self.host}")

    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[AIFunctionTool] = None,
        tool_choice: Optional[str] = None,
        format: Optional[str] = "",
        options: Optional[Dict[str, Any]] = {"num_ctx": 8192},
    ) -> str:
        converted_messages = self._convert_to_messages(messages)
        selected_tools = (
            self._functions_by_choices(tools, tool_choice) if tools else None
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=selected_tools,
            max_tokens=options["max_tokens"],
            response_format=format,
        )
        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["message"]["content"]
