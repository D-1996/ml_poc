from abc import ABC, abstractmethod
from torch import Tensor
from torchvision import transforms
from PIL.Image import Image as ImageType
from PIL import Image
from typing import TypeVar


T = TypeVar('T')

class BaseDataTransformer(ABC):
    @abstractmethod
    def transform(self, T) -> Tensor:
        ...


class CatsDogsDataTransformer(BaseDataTransformer):
    def __init__(self) -> None: #cfg path and some cfg to create those below + some piping %>
        self.transformations = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def transform(self, image: ImageType) -> Tensor:
        return self.transformations(image)[None, :, :, :]
