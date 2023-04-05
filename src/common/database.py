from typing import Any

import motor.motor_asyncio

from src.common.common import settings
from src.common.schemas import InferenceResult


class MongoDBDataAccess:
    def __init__(self, uri: str, database: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self.db = self.client[database]

    async def insert_one(self, collection: str, document: dict[Any, Any]) -> str:
        result = await self.db[collection].insert_one(document)
        return result.inserted_id

    async def find_one(self, collection: str, query) -> InferenceResult:
        result = await self.db[collection].find_one(query)
        del result["_id"]
        return result

    # async def find_many(self, collection: str, query) -> list[dict[Any, Any]]:
    #     cursor = self.db[collection].find(query)
    #     results = await cursor.to_list(length=None)
    #     return results


async def get_mongo_db():
    try:
        client = MongoDBDataAccess(settings.MONGO_URI, settings.MONGO_DB_NAME)
        yield client
    except Exception:
        raise Exception
    finally:
        client.client.close()
