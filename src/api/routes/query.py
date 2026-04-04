from fastapi import APIRouter, HTTPException

from src.schemas.rag import QueryRequest, QueryResponse
from src.api.routes.health import get_rag_service

router = APIRouter(prefix="/query", tags=["RAG Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Consulta RAG en lenguaje natural",
    description=(
        "Realiza una consulta en lenguaje natural sobre temas agrícolas. "
        "El sistema busca fragmentos relevantes en los documentos indexados "
        "y genera una respuesta contextualizada con citación de fuentes. "
        "En esta fase la búsqueda usa ChromaDB; la generación con LLM se habilita en HU-046."
    ),
)
async def rag_query(request: QueryRequest) -> QueryResponse:
    try:
        service = get_rag_service()
        return await service.query(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")
