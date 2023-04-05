from pydantic import BaseModel

from src.worker.schemas import ModelPrediction


class InferenceResult(BaseModel):
    inference_id: str
    inference_request_id: str
    prediction: ModelPrediction
