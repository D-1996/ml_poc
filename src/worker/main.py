import asyncio
import base64
from pathlib import Path
from uuid import uuid4

import aio_pika
from aio_pika import IncomingMessage
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings

from src.common.common import Settings, settings, setup_rabbitmq
from src.worker.ml_components import model, preprocessor


class ModelPrediction(BaseModel):
    cat: float
    dog: float


class PendingClassification(BaseModel):
    inference_id: str


app = FastAPI()


@app.on_event("startup")
async def startup_event():
    await setup_rabbitmq(app=app, settings=settings)
    loop = asyncio.get_running_loop()
    task = loop.create_task(infer(app.state.inference_request_queue))
    await task


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.rbmq_channel.close()
    await app.state.rbmq_connection.close()


async def on_message(message: IncomingMessage) -> ModelPrediction:
    async with message.process():
        img = base64.b64decode(message.body)
        preprocessed_img = preprocessor.preprocess(img)
        prediction = model.predict(preprocessed_img)
        print(message.properties.headers["request_id"])
        print(prediction)
        message.ack()


async def infer(queue) -> None:
    await queue.consume(callback=on_message)
