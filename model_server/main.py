from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from model_server.ml_components import model, preprocessor
from model_server.ml_components.enums import CatsDogsClass


class ModelPrediction(BaseModel):
    cat: float
    dog: float


app = FastAPI()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"hello": "wrld"}


from typing import Type


@app.post("/predict")
async def predict(image: UploadFile) -> ModelPrediction:
    img = image.file.read()
    preprocessed_img = preprocessor.preprocess(img)
    prediction = model.predict(preprocessed_img)
    return ModelPrediction(
        cat=prediction[CatsDogsClass.CAT], dog=prediction[CatsDogsClass.DOG]
    )
