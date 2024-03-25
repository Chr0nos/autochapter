from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
    CREATE INDEX frames_embeddings ON frame USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
    """


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
    DROP INDEX frames_embeddings;
    """
