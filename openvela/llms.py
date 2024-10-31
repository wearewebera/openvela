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

        Args:
            messages (list[dict]): A list of messages constituting the conversation.
            files (Optional[list[Dict[str, Any]]], optional): A list of files to be processed. Defaults to None.
            tools (Optional[list[AIFunctionTool]], optional): A list of tools available for the model. Defaults to None.
            tool_choice (Optional[str], optional): The specific tool to use. Defaults to None.
            format (Optional[str], optional): The desired format of the response. Defaults to None.

        Returns:
            str: The generated response.
        """
        pass

    @staticmethod
    def _convert_to_messages(dict_list: list[dict]) -> Sequence[Message]:
        """
        Converts a list of dictionaries to a sequence of Message TypedDicts.

        Args:
            dict_list (list[dict]): The list of message dictionaries.

        Returns:
            Sequence[Message]: The converted sequence of messages.

        Raises:
            ValueError: If a message contains an invalid role.
        """

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

        Args:
            files (list[Dict[str, Any]]): The list of file dictionaries.

        Returns:
            Sequence[Dict[str, Any]]: The sequence of processed file contents.

        Logs a warning if an unknown file type is encountered.
        """
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
        """
        Selects a tool based on the provided tool choice.

        Args:
            tools (list[AIFunctionTool]): The list of available tools.
            tool_choice (list[str]): The list containing the name of the chosen tool.

        Returns:
            AIFunctionTool: The selected tool.

        Raises:
            ValueError: If the specified tool is not found.
        """
        for tool in tools:
            if tool["function"]["name"] == tool_choice:
                return tool
        raise ValueError(f"Tool with name {tool_choice} not found.")


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
        options: Optional[Dict[str, Any]] = {"num_ctx": 8192},
    ) -> str:
        """
        Generates a response using the Ollama language model.

        Args:
            messages (list[dict]): The conversation history.
            files (Optional[list[Dict[str, Any]]], optional): Files to process. Defaults to None.
            tools (Optional[AIFunctionTool], optional): Available tools. Defaults to None.
            tool_choice (Optional[str], optional): Specific tool to use. Defaults to None.
            format (Optional[str], optional): Response format. Defaults to "".
            options (Optional[Dict[str, Any]], optional): Additional options for the model. Defaults to {"num_ctx": 8192}.

        Returns:
            str: The generated response.
        """
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

        Handles both text-based and audio inputs, performing transcription if audio files are provided.

        Args:
            messages (list[dict]): The conversation history.
            files (Optional[list[Dict[str, Any]]], optional): Files to process. Defaults to None.
            tools (Optional[AIFunctionTool], optional): Available tools. Defaults to None.
            tool_choice (Optional[str], optional): Specific tool to use. Defaults to None.
            **kwargs: Additional keyword arguments for the model.

        Returns:
            str: The generated response.
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
                    response = self.openai.chat.completions.create(
                        model=self.model, messages=converted_messages
                    )
        else:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=converted_messages,
                tools=selected_tools,
                tool_choice=tool_choice,
                **kwargs,
            )

        response_mapping: Mapping[str, Any] = next(iter([response]))
        return response_mapping["choices"][0]["message"]["content"]

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio content into text using OpenAI's transcription model.

        Args:
            audio_bytes (bytes): The binary content of the audio file.

        Returns:
            str: The transcribed text.
        """
        response = self.openai.audio.transcriptions.create(
            model=self.transcription_model, file=audio_bytes
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

        Args:
            format (str): The desired response format.

        Returns:
            dict[str, str]: The format mapping.

        Raises:
            ValueError: If an unknown format is provided.
        """
        if format == "json":
            return {"type": "json_object"}
        if format == "":
            return
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

        Args:
            messages (list[dict]): The conversation history.
            files (Optional[list[Dict[str, Any]]], optional): Files to process. Defaults to None.
            tools (Optional[AIFunctionTool], optional): Available tools. Defaults to None.
            tool_choice (Optional[str], optional): Specific tool to use. Defaults to None.
            format (Optional[str], optional): Response format. Defaults to "".
            **kwargs: Additional keyword arguments for the model.

        Returns:
            str: The generated response.
        """
        converted_messages = self._convert_to_messages(messages)
        selected_tools = (
            self._functions_by_choices(tools, tool_choice) if tools else None
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=converted_messages,
            tools=selected_tools,
            response_format=self._recognize_format(format),
            **kwargs,
        )

        return response.choices[0].message.content
