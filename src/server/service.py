import base64
from uuid import uuid4

import aio_pika
from starlette import datastructures


class InferenceService:
    def __init__(
        self, exchange: aio_pika.robust_exchange.RobustExchange, routing_key: str
    ) -> None:
        self.exchange = exchange
        self.routing_key = routing_key

    @staticmethod
    async def prepare_for_message(image: datastructures.UploadFile) -> aio_pika.Message:
        img = image.file.read()
        inference_request_id = str(uuid4())
        b64img = base64.b64encode(img)
        return aio_pika.Message(
            body=b64img,
            content_type="image/jpeg",
            headers={"request_id": inference_request_id},
        )

    async def publish_to_rabbit(self, message: aio_pika.Message) -> None:
        await self.exchange.publish(message, routing_key=self.routing_key)

    async def send_inference_request(self, image: datastructures.UploadFile) -> str:
        message = await self.prepare_for_message(image)
        inference_request_id = message.headers["request_id"]
        await self.publish_to_rabbit(message)
        return inference_request_id
