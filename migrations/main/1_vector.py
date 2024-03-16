from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
    CREATE EXTENSION IF NOT EXISTS "vector";
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        """
