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

    # Google Gemini
    GOOGLE_API_KEY: str = "your-google-api-key"

    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "./data/chroma"
    CHROMA_COLLECTION_NAME: str = "agrovision_docs"

    # LLM Generation
    LLM_MODEL: str = "gemini-2.5-flash"
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    # Retrieval
    RETRIEVER_TOP_K: int = 5
    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 100

    # Embedding
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 768

    # Backend API
    BACKEND_API_URL: str = "http://localhost:4000/api/v1"

    # Upload
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE_MB: int = 50

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
