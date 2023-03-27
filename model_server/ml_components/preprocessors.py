from abc import ABC, abstractmethod
from typing import Any, TypeVar, Type, Generic
from torch import Tensor
from model_server.ml_components.loaders import BaseLoader, CatsDogsImageLoader
from model_server.ml_components.transformers import BaseDataTransformer, CatsDogsDataTransformer
from PIL.Image import Image as ImageType

DType = TypeVar('DType')
LoaderDType = TypeVar('LoaderDType')
LoaderOutputDType = TypeVar('LoaderOutputDType')
TransformerDType = TypeVar('TransformerDType')

class BasePreprocessor(ABC, Generic[DType, LoaderDType, LoaderOutputDType, TransformerDType]):
    def __init__(
        self,
        loader: BaseLoader[LoaderDType, LoaderOutputDType],
        transformer: BaseDataTransformer[TransformerDType],
    ) -> None:
        self._loader = loader
        self.transformer = transformer

    @abstractmethod
    def preprocess(self, data: DType) -> Tensor:
        pass


class CatsDogsImagePreprocessor(BasePreprocessor[bytes, bytes, ImageType, ImageType]):
    def __init__(
        self,
        loader: CatsDogsImageLoader,
        transformer: CatsDogsDataTransformer,
    ) -> None:
        self._loader = loader
        self.transformer = transformer

    def preprocess(self, data: bytes) -> Tensor:
        img = self._loader.load(data)
        transformed_img = self.transformer.transform(img)
        return transformed_img