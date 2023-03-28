import io
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from PIL import Image
from PIL.Image import Image as ImageType

DType = TypeVar("DType")
OutputDType = TypeVar("OutputDType")


class BaseLoader(ABC, Generic[DType, OutputDType]):
    @staticmethod
    @abstractmethod
    def load(data: DType) -> OutputDType:
        pass


class CatsDogsImageLoader(BaseLoader[bytes, ImageType]):
    @staticmethod
    def load(data: bytes) -> ImageType:
        return Image.open(io.BytesIO(data))
