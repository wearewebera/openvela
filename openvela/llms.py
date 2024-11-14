import inspect  # Import the inspect module
import logging
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
    """
    Abstract base class representing a generic language model.
    Subclasses must implement the `generate_response` method to interface with specific LLMs.
    """

    @abstractmethod
    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[list[AIFunctionTool]] = None,
        tool_choice: Optional[str] = None,
        format: Optional[str] = None,
    ) -> str:
        """
        Generates a response based on the provided messages and optional tools.
        """
        pass

    @staticmethod
    def _convert_to_messages(dict_list: list[dict]) -> Sequence[Message]:
        """
        Converts a list of dictionaries to a sequence of Message TypedDicts.
        """

        # (Implementation remains the same)
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
        """
        Converts a list of file dictionaries to a sequence of processed file contents.
        """
        # (Implementation remains the same)
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
        tools: list[AIFunctionTool], tool_choice: str
    ) -> AIFunctionTool:
        """
        Selects a tool based on the provided tool choice.
        """
        # (Implementation remains the same)
        for tool in tools:
            if tool["function"]["name"] == tool_choice:
                return tool
        raise ValueError(f"Tool with name {tool_choice} not found.")

    @staticmethod
    def _filter_kwargs_for_function(func, kwargs):
        """
        Filters kwargs to include only those parameters accepted by func.
        """
        sig = inspect.signature(func)
        valid_params = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        return filtered_kwargs


@dataclass
class OllamaModel(Model):
    """
    Concrete implementation of Model interfacing with the Ollama language model.
    """

    base_url: str = "http://localhost:11434/"
    client: Client = field(init=False)
    model: str = "llama3.2"

    def __post_init__(self):
        """
        Initializes the Ollama client upon instantiation.
        """
        self.client = Client(self.base_url)

    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[AIFunctionTool] = None,
        tool_choice: Optional[str] = None,
        format: Optional[str] = "",
        **kwargs,
    ) -> str:
        """
        Generates a response using the Ollama language model.
        """
        converted_messages = self._convert_to_messages(messages)
        selected_tools = (
            self._functions_by_choices(tools, tool_choice) if tools else None
        )

        # Filter kwargs for the client.chat function
        filtered_kwargs = self._filter_kwargs_for_function(self.client.chat, kwargs)

        response = self.client.chat(
            model=self.model,
            messages=converted_messages,
            tools=selected_tools,
            options=filtered_kwargs,  # Pass filtered kwargs as options
            format=format or kwargs.get("format"),
        )

        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["message"]["content"]


@dataclass
class OpenAIModel(Model):
    """
    Concrete implementation of Model interfacing with the OpenAI language model.
    Handles both text generation and audio transcription.
    """

    api_key: str
    model: str = "gpt-4o-mini"
    openai: OpenAI = field(init=False)
    transcription_model: str = "whisper-1"

    def __post_init__(self):
        """
        Initializes the OpenAI client with the provided API key.
        """
        self.openai = OpenAI(self.api_key)

    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[AIFunctionTool] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generates a response using the OpenAI language model.
        """
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
                    # Filter kwargs for the chat completion create function
                    filtered_kwargs = self._filter_kwargs_for_function(
                        self.openai.chat.completions.create, kwargs
                    )
                    response = self.openai.chat.completions.create(
                        model=self.model,
                        messages=converted_messages,
                        **filtered_kwargs,
                    )
        else:
            # Filter kwargs for the chat completion create function
            filtered_kwargs = self._filter_kwargs_for_function(
                self.openai.chat.completions.create, kwargs
            )
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=converted_messages,
                tools=selected_tools,
                tool_choice=tool_choice,
                **filtered_kwargs,
            )

        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["choices"][0]["message"]["content"]

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio content into text using OpenAI's transcription model.
        """
        # Filter kwargs for the audio transcription function
        filtered_kwargs = self._filter_kwargs_for_function(
            self.openai.audio.transcriptions.create, {}
        )
        response = self.openai.audio.transcriptions.create(
            model=self.transcription_model, file=audio_bytes, **filtered_kwargs
        )
        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["text"]


@dataclass
class GroqModel(Model):
    """
    Concrete implementation of Model interfacing with the Groq language model.
    """

    client: Groq = field(init=False)
    api_key: str = ""
    model: str = "llama-3.1-70b-versatile"

    def __post_init__(self):
        """
        Initializes the Groq client with the provided API key.
        """
        self.client = Groq(api_key=self.api_key)

    def _recognize_format(self, format: str) -> dict[str, str]:
        """
        Maps the requested format to Groq's expected response format.
        """
        if format == "json":
            return {"type": "json_object"}
        if format == "":
            return {}
        else:
            raise ValueError(f"Unknown format: {format}")

    def generate_response(
        self,
        messages: list[dict],
        files: Optional[list[Dict[str, Any]]] = None,
        tools: Optional[AIFunctionTool] = None,
        tool_choice: Optional[str] = None,
        format: Optional[str] = "",
        **kwargs,
    ):
        """
        Generates a response using the Groq language model.
        """
        converted_messages = self._convert_to_messages(messages)
        selected_tools = (
            self._functions_by_choices(tools, tool_choice) if tools else None
        )

        # Filter kwargs for the chat completion create function
        filtered_kwargs = self._filter_kwargs_for_function(
            self.client.chat.completions.create, kwargs
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=selected_tools,
            response_format=self._recognize_format(format),
            **filtered_kwargs,
        )

        return response.choices[0].message.content
