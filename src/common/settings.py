from pydantic import BaseSettings


class Settings(BaseSettings):
    RABBIT_HOST: str = "localhost"
    RABBIT_PORT: int = 5672
    RABBIT_LOGIN: str = "my_user"
    RABBIT_PASSWORD: str = "my_password"
    INFERENCE_REQUEST_EXCHANGE: str = "inference_request_exchange"
    INFERENCE_REQUEST_QUEUE: str = "inference_request_queue"
    INFERENCE_REQUEST_ROUTING_KEY: str = "inference_request"

    MONGO_URI: str = "mongodb://db_user:db_password@localhost:27017"
    MONGO_DB_NAME: str = "mongo_db"
    MONGO_INFERENCE_COLLECTION: str = "inference_collection"