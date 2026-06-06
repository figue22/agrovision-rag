"""
HU-050 — Endpoints de feedback y métricas RAG
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException

from src.schemas.rag import FeedbackRequest, FeedbackResponse, MetricasRAG
from src.services.feedback_service import FeedbackService

router = APIRouter(prefix="/feedback", tags=["Feedback & Métricas"])
logger = logging.getLogger("agrovision-rag")

_feedback_service: Optional[FeedbackService] = None


def get_feedback_service() -> FeedbackService:
    global _feedback_service
    if _feedback_service is None:
        _feedback_service = FeedbackService()
    return _feedback_service


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Enviar feedback 1-5 estrellas para una respuesta RAG",
)
async def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Registra el feedback del agricultor sobre una respuesta RAG."""
    if not 1 <= request.rating <= 5:
        raise HTTPException(status_code=400, detail="El rating debe ser entre 1 y 5")
    try:
        service = get_feedback_service()
        feedback_id = service.guardar_feedback(request)
        logger.info(
            "Feedback recibido | consulta=%s | rating=%d",
            request.consulta_id, request.rating,
        )
        return FeedbackResponse(feedback_id=feedback_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error guardando feedback: {str(e)}")


@router.get(
    "/metricas",
    response_model=MetricasRAG,
    summary="Dashboard métricas calidad RAG",
)
async def get_metricas() -> MetricasRAG:
    """
    Retorna métricas del sistema RAG:
    - Satisfacción promedio (1-5 estrellas)
    - Distribución de ratings
    - Consultas más frecuentes
    - Documentos más citados
    - Relevancia y latencia promedio
    """
    try:
        service = get_feedback_service()
        return service.get_metricas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculando métricas: {str(e)}")
