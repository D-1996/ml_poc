from pydantic import BaseModel


class ModelPrediction(BaseModel):
    cat: float
    dog: float
