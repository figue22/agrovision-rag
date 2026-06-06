"""
HU-047 — Endpoint consultas RAG en lenguaje natural
POST /query: pregunta -> respuesta contextualizada + fuentes + logging
"""

import logging
import time

from fastapi import APIRouter, HTTPException, Request

from src.schemas.rag import QueryRequest, QueryResponse
from src.api.routes.health import get_rag_service

router = APIRouter(prefix="/query", tags=["RAG Query"])
logger = logging.getLogger("agrovision-rag")

# ── Respuestas predefinidas (fallback) ──
FALLBACK_RESPONSES = {
    "cafe": (
        "El café colombiano requiere temperaturas entre 18-24°C, altitudes de 1200-2000 msnm "
        "y precipitaciones de 1800-2800 mm anuales. Para consultas específicas, asegúrate de "
        "tener documentos técnicos indexados en el sistema."
    ),
    "cacao": (
        "El cacao colombiano crece mejor entre 24-30°C, a altitudes de 0-1200 msnm "
        "con precipitaciones de 1500-2500 mm anuales. Para información detallada, "
        "indexa documentos técnicos del ICA o AGROSAVIA."
    ),
    "plaga": (
        "El control de plagas depende del cultivo y la región. Consulta con el ICA "
        "o un técnico agrícola local. Para respuestas específicas, indexa manuales "
        "técnicos en el sistema."
    ),
    "fertilizacion": (
        "La fertilización debe basarse en análisis de suelo. Los nutrientes principales "
        "son nitrógeno, fósforo y potasio. Para recomendaciones específicas, "
        "indexa guías técnicas de fertilización."
    ),
    "default": (
        "No encontré información relevante en los documentos indexados. "
        "Te recomiendo consultar con un técnico agrícola del ICA o AGROSAVIA, "
        "o indexar más documentos técnicos en el sistema."
    ),
}


def get_fallback_response(pregunta: str) -> str:
    """Retorna una respuesta predefinida basada en palabras clave."""
    pregunta_lower = pregunta.lower()
    for keyword, response in FALLBACK_RESPONSES.items():
        if keyword != "default" and keyword in pregunta_lower:
            return response
    return FALLBACK_RESPONSES["default"]


@router.post(
    "",
    response_model=QueryResponse,
    summary="Consulta RAG en lenguaje natural",
    description=(
        "Realiza una consulta en lenguaje natural sobre temas agrícolas colombianos. "
        "El sistema busca fragmentos relevantes en los documentos indexados y genera "
        "una respuesta contextualizada con citación de fuentes (documento, página, institución). "
        "Si no hay documentos relevantes retorna respuestas predefinidas como fallback."
    ),
)
async def rag_query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """
    Pipeline RAG completo:
    1. Recibe pregunta en español
    2. Busca chunks relevantes en ChromaDB
    3. Genera respuesta con Gemini
    4. Retorna fuentes citadas con documento, página e institución
    5. Fallback si no hay documentos relevantes
    """
    start = time.time()
    client_ip = http_request.client.host if http_request.client else "unknown"

    logger.info(
        "QUERY | ip=%s | pregunta='%s' | top_k=%d | categoria=%s",
        client_ip,
        request.pregunta[:80],
        request.top_k,
        request.filtro_categoria or "todas",
    )

    try:
        service = get_rag_service()
        response = await service.query(request)

        # Si no hay fuentes relevantes, usar fallback
        if not response.fuentes or (response.relevancia_pct and response.relevancia_pct < 20):
            fallback = get_fallback_response(request.pregunta)
            logger.warning(
                "FALLBACK | pregunta='%s' | relevancia=%.1f%%",
                request.pregunta[:80],
                response.relevancia_pct or 0,
            )
            response.respuesta = fallback + "\n\n" + response.respuesta if response.fuentes else fallback

        duration_ms = round((time.time() - start) * 1000, 2)

        logger.info(
            "QUERY OK | ip=%s | fuentes=%d | relevancia=%.1f%% | tokens=%s | ms=%.1f",
            client_ip,
            len(response.fuentes),
            response.relevancia_pct or 0,
            response.tokens_usados or "N/A",
            duration_ms,
        )

        return response

    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 2)
        logger.error(
            "QUERY ERROR | ip=%s | pregunta='%s' | error=%s | ms=%.1f",
            client_ip,
            request.pregunta[:80],
            str(e),
            duration_ms,
        )
        raise HTTPException(status_code=500, detail=f"Error al procesar consulta: {str(e)}")
