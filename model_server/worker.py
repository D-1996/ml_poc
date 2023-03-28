
import asyncio
from aio_pika import IncomingMessage
import aio_pika
import base64
# from model_server.ml_components import model, preprocessor

RABBIT_HOST = "localhost"
RABBIT_PORT = 5672



async def on_message(message: IncomingMessage):
    image_bytes = base64.b64decode(message.body)
    print(image_bytes)



async def main(loop):
    rabbit_connection = await aio_pika.connect_robust(
        host=RABBIT_HOST,
        port=RABBIT_PORT,
        login="my_user",
        password="my_password",
        loop=loop
    )


    channel = await rabbit_connection.channel()
    inference_queue = await channel.declare_queue("inference_queue")

    await inference_queue.consume(callback=on_message)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(main(loop))
    loop.run_forever()