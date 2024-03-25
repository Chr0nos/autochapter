from tortoise import BaseDBAsyncClient


async def upgrade(db: BaseDBAsyncClient) -> str:
    return """
        CREATE TABLE IF NOT EXISTS "file" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "filename" VARCHAR(1000) NOT NULL,
    "scanned_at" TIMESTAMPTZ NOT NULL  DEFAULT CURRENT_TIMESTAMP,
    "fps" INT NOT NULL
);
CREATE INDEX IF NOT EXISTS "idx_file_filenam_d431ae" ON "file" ("filename");
        CREATE TABLE IF NOT EXISTS "frame" (
    "id" SERIAL NOT NULL PRIMARY KEY,
    "offset" BIGINT NOT NULL,
    "embedding" public.vector(512) NOT NULL,
    "file_id" INT NOT NULL REFERENCES "file" ("id") ON DELETE CASCADE
);"""


async def downgrade(db: BaseDBAsyncClient) -> str:
    return """
        DROP TABLE IF EXISTS "file";
        DROP TABLE IF EXISTS "frame";"""
