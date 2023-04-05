from pydantic import BaseModel


class PendingClassification(BaseModel):
    inference_request_id: str
