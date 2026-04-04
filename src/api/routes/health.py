from datetime import datetime
from fastapi import APIRouter, Depends

from src.config.settings import Settings, get_settings
from src.schemas.rag import HealthResponse
from src.services.rag_service import RAGService

router = APIRouter(tags=["Health"])

_rag_service = RAGService()


def get_rag_service() -> RAGService:
    return _rag_service


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check del servicio RAG",
)
async def health_check(
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    service = get_rag_service()

    return HealthResponse(
        status="ok",
        service="agrovision-rag",
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        chromadb_connected=service.chroma.is_connected,
        chromadb_collections=service.chroma.collection_count,
        chromadb_total_documents=service.chroma.document_count,
        openai_configured=service.openai_configured,
        database_connected=False,
        uptime_seconds=round(service.uptime, 2),
        timestamp=datetime.utcnow(),
    )
