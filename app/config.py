from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "GKE Weaviate RAG"
    weaviate_http_host: str = "weaviate"
    weaviate_http_port: int = 8080
    weaviate_grpc_host: str = "weaviate"
    weaviate_grpc_port: int = 50051
    weaviate_secure: bool = False
    weaviate_api_key: str | None = None
    COLLECTION_NAME: str = "DocumentChunk"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    chunk_size: int = 800
    chunk_overlap: int = 120

    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"

    gcp_project_id: str | None = None
    pubsub_subscription: str = "document-ingestion-sub"


@lru_cache
def get_settings() -> Settings:
    return Settings()