import asyncio

import aio_pika
from fastapi import FastAPI
from pydantic import BaseSettings


class Settings(BaseSettings):
    RABBIT_HOST: str = "localhost"
    RABBIT_PORT: int = 5672
    RABBIT_LOGIN: str = "my_user"
    RABBIT_PASSWORD: str = "my_password"
    INFERENCE_REQUEST_EXCHANGE: str = "inference_request_exchange"
    INFERENCE_REQUEST_QUEUE: str = "inference_request_queue"
    INFERENCE_REQUEST_ROUTING_KEY: str = "inference_request"


settings = Settings()


async def setup_rabbitmq(app: FastAPI, settings: Settings) -> None:
    loop = asyncio.get_running_loop()
    connection = await aio_pika.connect_robust(
        host=settings.RABBIT_HOST,
        port=settings.RABBIT_PORT,
        login=settings.RABBIT_LOGIN,
        password=settings.RABBIT_PASSWORD,
        loop=loop,
    )

    channel = await connection.channel()
    inference_request_exchange = await channel.declare_exchange(
        settings.INFERENCE_REQUEST_EXCHANGE
    )
    inference_request_queue = await channel.declare_queue(
        settings.INFERENCE_REQUEST_QUEUE
    )
    await inference_request_queue.bind(
        exchange=inference_request_exchange,
        routing_key=settings.INFERENCE_REQUEST_ROUTING_KEY,
    )
    await channel.set_qos(prefetch_count=10)

    app.state.rbmq_connection = connection
    app.state.rbmq_channel = channel
    app.state.inference_request_exchange = inference_request_exchange
    app.state.inference_request_queue = inference_request_queue
