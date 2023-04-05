from fastapi import FastAPI, UploadFile

from src.common.common import Settings, settings, setup_rabbitmq
from src.server.schemas import PendingClassification
from src.server.service import InferenceService

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
