"""
KnowledgeBrain API — Configuration
Centralized settings via pydantic-settings with .env support.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Anthropic ──────────────────────────────────
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    # ── Qdrant ─────────────────────────────────────
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "knowledgebrain_docs"

    # ── Neo4j ──────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "knowledgebrain_password"

    # ── Redis ──────────────────────────────────────
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # ── Azure Document Intelligence (Optional) ────
    azure_form_recognizer_endpoint: str = ""
    azure_form_recognizer_key: str = ""

    # ── Application ────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    max_upload_size_mb: int = 100
    default_top_k: int = 10
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    # ── Embedding ──────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
