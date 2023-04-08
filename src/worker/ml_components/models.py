import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models as MODELS
from torchvision.models.densenet import DenseNet

from src.worker.ml_components.enums import CatsDogsClass

warnings.filterwarnings("ignore")


class CatsDogsVisionModel:
    MODEL_EXTENSION = ".pth"
    CLASS_MAPPING = {0: CatsDogsClass.CAT, 1: CatsDogsClass.DOG}

    def __init__(self, weights_path: Path) -> None:
        self._model_weights: dict[str, Any] = self._load_weights(weights_path)
        self.model: DenseNet = self.load_model()

    @staticmethod
    def _load_weights(path: Path, map_location: str = "cpu") -> dict[str, Any]:
        if not path.exists() or path.suffix != CatsDogsVisionModel.MODEL_EXTENSION:
            raise ValueError(
                f"Path: {path} does not exist or the file has incorrect extension."
                f" Expected: {CatsDogsVisionModel.MODEL_EXTENSION}"
            )

        return dict(torch.load(path, map_location=map_location))

    @staticmethod
    def _get_model_architecture() -> DenseNet:
        model = MODELS.densenet121(pretrained=False)
        model.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1),
        )
        return model

    def load_model(self) -> DenseNet:
        model = self._get_model_architecture()
        model.parameters = self._model_weights["parameters"]
        model.load_state_dict(self._model_weights["state_dict"])
        model.eval()
        return model

    @staticmethod
    def _tensor_to_dict(tensor: torch.Tensor) -> dict[int, float]:
        values, indices = torch.topk(tensor, k=2)
        values = values.tolist()[0]
        indices = indices.tolist()[0]
        return dict(zip(indices, values))

    def predict(self, input_data: torch.Tensor) -> dict[CatsDogsClass, float]:
        result = torch.exp(self.model(input_data))
        result_dict = self._tensor_to_dict(result)
        return {v: result_dict[k] for k, v in CatsDogsVisionModel.CLASS_MAPPING.items()}
