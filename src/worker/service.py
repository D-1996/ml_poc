import base64
from uuid import uuid4

from aio_pika import IncomingMessage

from src.common.database import MongoDBDataAccess
from src.worker.ml_components import model, preprocessor
from src.worker.ml_components.enums import CatsDogsClass
from src.worker.schemas import InferenceResult, ModelPrediction


class InferenceWorkerService:
    def __init__(self, database: MongoDBDataAccess) -> None:
        self.database = database

    @staticmethod
    def get_prediction_from_message(message: IncomingMessage) -> InferenceResult:
        img = base64.b64decode(message.body)
        inference_id = str(uuid4())
        preprocessed_img = preprocessor.preprocess(img)
        prediction = model.predict(preprocessed_img)
        return InferenceResult(
            inference_id=inference_id,
            inference_request_id=message.properties.headers["request_id"],
            prediction=ModelPrediction(
                cat=prediction[CatsDogsClass.CAT], dog=prediction[CatsDogsClass.DOG]
            ),
        )

    async def on_message(self, message: IncomingMessage, collection_name: str):
        async with message.process():
            result = self.get_prediction_from_message(message)
            await self.database.insert_one(collection_name, result.dict())
            message.ack()
