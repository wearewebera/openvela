import io
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


class File(ABC):
    @abstractmethod
    def read(self, path: str) -> bytes:
        pass


@dataclass
class OpenVelaAudioFile(File):
    def read(self, path: str) -> bytes:
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
    def read(self, path: str) -> bytes:
        try:
            with Image.open(path) as img:
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format=img.format)
                return img_byte_arr.getvalue()
        except Exception as e:
            raise ValueError(f"Failed to read image file: {e}")
