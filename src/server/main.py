from fastapi import Depends, FastAPI, UploadFile

from src.common import settings
from src.common.database import MongoDBDataAccess, get_mongo_db
from src.common.rabbit import setup_rabbitmq
from src.common.schemas import InferenceResult
from src.server.schemas import PendingClassification
from src.server.service import InferenceOrchestratorService

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
    inference_service = InferenceOrchestratorService(
        exchange=app.state.inference_request_exchange,
        routing_key=settings.INFERENCE_REQUEST_ROUTING_KEY,
    )
    inference_request_id = await inference_service.send_inference_request(image)
    return PendingClassification(inference_request_id=inference_request_id)


@app.get("/result")
async def result(
    inference_request_id: str, db: MongoDBDataAccess = Depends(get_mongo_db)
) -> InferenceResult:
    inference_service = InferenceOrchestratorService(
        collection_name=settings.MONGO_INFERENCE_COLLECTION, database=db
    )

    result = await inference_service.get_result(inference_request_id)

    return result
