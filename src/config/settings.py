from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Configuración del servicio RAG de AgroVision."""

    # General
    APP_NAME: str = "AgroVision RAG Service"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"
    APP_PORT: int = 8001
    DEBUG: bool = True

    # Base de datos
    DATABASE_URL: str = "postgresql://agrovision_user:agrovision_pass_2026@localhost:5432/agrovision_db"

    # OpenAI
    OPENAI_API_KEY: str = "your-openai-api-key"

    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "agrovision_docs"

    # LLM Generation
    LLM_MODEL: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 500

    # Retrieval
    RETRIEVER_TOP_K: int = 5
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50

    # Embedding
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072

    # Backend API
    BACKEND_API_URL: str = "http://localhost:4000/api/v1"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
