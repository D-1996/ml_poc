from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from model_server.ml_components import model, preprocessor
from model_server.ml_components.enums import CatsDogsClass


from uuid import uuid4
import base64 
import pika
import aio_pika


class ModelPrediction(BaseModel):
    cat: float
    dog: float

class PendingClassification(BaseModel):
    inference_id: str

app = FastAPI()


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"hello": "wrld"}




async def publish_to_rabbit(message: aio_pika.Message):
    rabbit_connection = await aio_pika.connect_robust(
        host='localhost',
        port=5672,
        login="my_user",
        password="my_password")
    channel = await rabbit_connection.channel()
    exchange = await channel.declare_exchange("classification_requests",aio_pika.ExchangeType.TOPIC)
    await exchange.publish(
        message,
        routing_key="inference_queue"
    )
    await rabbit_connection.close()

@app.post("/predict")
async def predict(image: UploadFile) -> PendingClassification:
    img = image.file.read()
    b64img = base64.b64encode(img) 
    inference_id = str(uuid4())
    message = aio_pika.Message(body=b64img, content_type="image/jpeg", headers={"inference_id": inference_id},)
    await publish_to_rabbit(message)

    return PendingClassification(inference_id=inference_id)


    # preprocessed_img = preprocessor.preprocess(img)
    # prediction = model.predict(preprocessed_img)
    # return ModelPrediction(
    #     cat=prediction[CatsDogsClass.CAT], dog=prediction[CatsDogsClass.DOG]
    # )
