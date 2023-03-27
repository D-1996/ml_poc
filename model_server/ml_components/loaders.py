import io
from PIL import Image
from PIL.Image import Image as ImageType
from abc import ABC, abstractmethod
from typing import TypeVar, Type, Generic


DType = TypeVar('DType')
OutputDType = TypeVar('OutputDType')


class BaseLoader(ABC, Generic[DType, OutputDType]):
    @staticmethod
    @abstractmethod
    def load(data: DType) -> OutputDType:
        pass


class CatsDogsImageLoader(BaseLoader[bytes, ImageType]):
    @staticmethod
    def load(data: bytes) -> ImageType:
        return Image.open(io.BytesIO(data))