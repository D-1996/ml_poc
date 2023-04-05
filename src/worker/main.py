import asyncio
from functools import partial

from aio_pika.robust_queue import RobustQueue
from fastapi import FastAPI

from src.common.common import settings, setup_rabbitmq
from src.common.database import MongoDBDataAccess, get_mongo_db
from src.worker.service import InferenceWorkerService

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


async def infer(queue: RobustQueue) -> None:
    mongo_db = MongoDBDataAccess(settings.MONGO_URI, settings.MONGO_DB_NAME)
    worker_service = InferenceWorkerService(database=mongo_db)
    on_message_callback = partial(
        worker_service.on_message, collection_name=settings.MONGO_INFERENCE_COLLECTION
    )
    await queue.consume(callback=on_message_callback)
