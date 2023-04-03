import base64
from uuid import uuid4

import aio_pika
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from src.common.common import Settings, settings, setup_rabbitmq


class ModelPrediction(BaseModel):
    cat: float
    dog: float


class PendingClassification(BaseModel):
    inference_request_id: str


settings = Settings()
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    await setup_rabbitmq(app=app, settings=settings)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.rbmq_channel.close()
    await app.state.rbmq_connection.close()


async def publish_to_rabbit(
    message: aio_pika.Message,
    exchange: aio_pika.robust_exchange.RobustExchange,
    key: str,
) -> None:
    await exchange.publish(message, routing_key=key)


@app.post("/inference")
async def predict(image: UploadFile) -> PendingClassification:
    exchange: aio_pika.robust_exchange.RobustExchange = (
        app.state.inference_request_exchange
    )
    img = image.file.read()
    inference_request_id = str(uuid4())
    message = create_message(img, inference_request_id)
    await publish_to_rabbit(
        message=message,
        exchange=exchange,
        key=settings.INFERENCE_REQUEST_ROUTING_KEY,
    )
    return PendingClassification(inference_request_id=inference_request_id)


def create_message(img: bytes, request_id: str) -> aio_pika.Message:
    b64img = base64.b64encode(img)
    return aio_pika.Message(
        body=b64img,
        content_type="image/jpeg",
        headers={"request_id": request_id},
    )
