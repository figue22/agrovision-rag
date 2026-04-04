from fastapi import APIRouter

from src.schemas.rag import DocumentsListResponse
from src.api.routes.health import get_rag_service

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.get(
    "",
    response_model=DocumentsListResponse,
    summary="Listar documentos indexados",
    description=(
        "Retorna la lista de documentos indexados en ChromaDB con su estado. "
        "La gestión completa de documentos (upload, re-indexación) se implementa en HU-047."
    ),
)
async def list_documents() -> DocumentsListResponse:
    service = get_rag_service()
    return service.get_documents_list()
