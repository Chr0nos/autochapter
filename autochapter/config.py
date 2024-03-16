from pydantic_config import SettingsModel, SettingsConfig
from pydantic import BaseModel

class PostgresConfig(BaseModel):
    host: str
    port: int = 5432
    username: str
    password: str
    db: str


class Config(SettingsModel):
    postgres: PostgresConfig

    model_config = SettingsConfig(env_nested_delimiter='__', env_file='.env',)


cfg = Config()
