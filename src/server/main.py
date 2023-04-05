from typing import AsyncGenerator

from fastapi import Depends, FastAPI, UploadFile

from src.common.common import Settings, settings, setup_rabbitmq
from src.common.database import get_mongo_db
from src.server.schemas import PendingClassification
from src.server.service import InferenceService
from src.common.schemas import InferenceResult

settings = Settings()
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    await setup_rabbitmq(app=app, settings=settings)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    await app.state.rbmq_channel.close()
    await app.state.rbmq_connection.close()


@app.post("/inference")
async def predict(image: UploadFile) -> PendingClassification:
    inference_service = InferenceService(
        exchange=app.state.inference_request_exchange,
        routing_key=settings.INFERENCE_REQUEST_ROUTING_KEY,
    )
    inference_request_id = await inference_service.send_inference_request(image)
    return PendingClassification(inference_request_id=inference_request_id)


@app.get("/result")
async def result(
    inference_request_id: str, db: AsyncGenerator = Depends(get_mongo_db)
) -> InferenceResult:
    results = await db.find_one(
        settings.MONGO_INFERENCE_COLLECTION,
        {"inference_request_id": {"$eq": inference_request_id}},
    )
    return results
