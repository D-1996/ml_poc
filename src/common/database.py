from typing import Any

import motor.motor_asyncio

MONGO_DETAILS = "mongodb://db_user:db_password@localhost:27017"
DATABASE = "Predictions"


class MongoDBDataAccess:
    def __init__(self, uri: str, database: str):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        self.db = self.client[database]

    async def insert_one(self, collection: str, document: dict[Any, Any]) -> str:
        result = await self.db[collection].insert_one(document)
        return result.inserted_id

    async def find_one(self, collection: str, query):
        result = await self.db[collection].find_one(query)
        return result

    async def find_many(self, collection: str, query):
        cursor = self.db[collection].find(query)
        results = await cursor.to_list(length=None)
        return results


async def get_mongo_db():
    client = MongoDBDataAccess(MONGO_DETAILS, DATABASE)
    return client
    # async with await client.client.start_session() as s:
    #     async with s.start_transaction():
    #         yield s
