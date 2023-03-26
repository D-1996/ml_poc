from abc import ABC, abstractmethod
from typing import Any, TypeVar, Type
from torch import Tensor
from model_server.ml_components.loaders import BaseLoader, CatsDogsImageLoader
from model_server.ml_components.transformers import BaseDataTransformer, CatsDogsDataTransformer
from PIL.Image import Image as ImageType

T = TypeVar('T')


class BasePreprocessor(ABC):
    def __init__(
        self,
        loader: BaseLoader,
        transformer: BaseDataTransformer,
    ) -> None:
        self._loader = loader
        self.transformer = transformer

    @abstractmethod
    def preprocess(self, data: Type[T]) -> Tensor:
        pass


class CatsDogsImagePreprocessor(BasePreprocessor):
    def __init__(
        self,
        loader: CatsDogsImageLoader,
        transformer: CatsDogsDataTransformer,
    ) -> None:
        self._loader = loader
        self.transformer = transformer

    def preprocess(self, data: bytes) -> Tensor:
        img: Type[ImageType] = self._loader.load(data)
        transformed_img = self.transformer.transform(img)
        return transformed_img