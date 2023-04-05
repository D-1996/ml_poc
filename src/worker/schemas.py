from pydantic import BaseModel


class ModelPrediction(BaseModel):
    cat: float
    dog: float


class InferenceResult(BaseModel):
    inference_id: str
    inference_request_id: str
    prediction: ModelPrediction
