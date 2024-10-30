import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


class File(ABC):
    """
    Abstract base class representing a generic file handler.
    Subclasses must implement the `read` method to handle specific file types.
    """

    @abstractmethod
    def read(self, path: str) -> bytes:
        """
        Abstract method to read a file from the given path.

        Args:
            path (str): The file system path to the file.

        Returns:
            bytes: The binary content of the file.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        pass


@dataclass
class OpenVelaAudioFile(File):
    """
    Concrete implementation of the File class for handling audio files.
    Specifically designed to read WAV files and other generic audio formats.
    """

    def read(self, path: str) -> bytes:
        """
        Reads an audio file from the specified path.

        For WAV files, it reads all frames using the wave module.
        For other audio formats, it reads the file in binary mode.

        Args:
            path (str): The file system path to the audio file.

        Returns:
            bytes: The binary content of the audio file.

        Raises:
            ValueError: If reading the audio file fails.
        """
        try:
            if path.endswith(".wav"):
                with wave.open(path, "rb") as audio_file:
                    return audio_file.readframes(audio_file.getnframes())
            else:
                with open(path, "rb") as f:
                    return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read audio file: {e}")


@dataclass
class OpenVelaImageFile(File):
    """
    Concrete implementation of the File class for handling image files.
    Utilizes the Pillow library to open and process images.
    """

    def read(self, path: str) -> bytes:
        """
        Reads an image file from the specified path and returns its binary content.

        Args:
            path (str): The file system path to the image file.

        Returns:
            bytes: The binary content of the image file.

        Raises:
            ValueError: If reading the image file fails.
        """

        try:
            with Image.open(path) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                return img_byte_arr.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to read image file: {e}")
