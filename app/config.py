from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    llm_api_url: str = "mock"
    llm_api_key: str = "mock-key"
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600
    postgres_url: str = "postgresql://user:password@localhost:5432/semantic_cache"
    kafka_broker: str = "localhost:9092"
    similarity_threshold: float = 0.85

    model_config = {"env_file": ".env"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()