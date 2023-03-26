import io

from PIL import Image
from PIL.Image import Image as ImageType

from abc import ABC, abstractmethod
from typing import TypeVar, Type, Any


T = TypeVar('T', covariant=True)
R = TypeVar('R', covariant=True)


class BaseLoader(ABC):
    @staticmethod
    @abstractmethod
    def load(data: Type[T]) -> Any:
        pass


class CatsDogsImageLoader(BaseLoader):
    @staticmethod
    def load(data: bytes) -> ImageType:
        return Image.open(io.BytesIO(data))