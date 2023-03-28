from pathlib import Path

from model_server.ml_components.loaders import CatsDogsImageLoader
from model_server.ml_components.models import CatsDogsVisionModel
from model_server.ml_components.preprocessors import CatsDogsImagePreprocessor
from model_server.ml_components.transformers import CatsDogsDataTransformer

__all__ = ["preprocessor", "model"]

MODEL_PATH = Path(
    "/Users/damiandymkowski/Documents/Personal/cat_dog_fastapi/model_server/resources/model_26032023.pth"  # noqa
)

model = CatsDogsVisionModel(MODEL_PATH)
loader = CatsDogsImageLoader()
data_transformer = CatsDogsDataTransformer()
preprocessor = CatsDogsImagePreprocessor(
    loader=loader,
    transformer=data_transformer,
)
