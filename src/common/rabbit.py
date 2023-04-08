import asyncio

import aio_pika
from fastapi import FastAPI
from src.common.settings import Settings



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


