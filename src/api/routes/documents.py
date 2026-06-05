"""
HU-044 — Endpoints de gestión de documentos
"""

import os
import shutil
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.schemas.rag import DocumentUploadResponse, DocumentsListResponse, EstadoIndexacion
from src.services.ingestion_service import IngestionService
from src.config.settings import get_settings

router = APIRouter(prefix="/documents", tags=["Documents"])
settings = get_settings()

_ingestion_service: Optional[IngestionService] = None


def get_ingestion_service() -> IngestionService:
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = IngestionService()
    return _ingestion_service


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Subir e indexar documento PDF o DOCX",
)
async def upload_document(
    file: UploadFile = File(...),
    titulo: str = Form(...),
    categoria: str = Form(default="general"),
    parcela_id: Optional[str] = Form(default=None),
    subido_por_id: Optional[str] = Form(default=None),
    idioma: str = Form(default="es"),
) -> DocumentUploadResponse:
    """Sube un documento PDF o DOCX, lo procesa y lo indexa en ChromaDB."""

    # Validar tipo de archivo
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".docx", ".doc"):
        raise HTTPException(
            status_code=400,
            detail="Solo se aceptan archivos PDF o DOCX",
        )

    # Leer contenido
    content = await file.read()
    tamano_kb = len(content) // 1024

    # Validar tamaño
    if tamano_kb > settings.MAX_FILE_SIZE_MB * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"El archivo supera el límite de {settings.MAX_FILE_SIZE_MB} MB",
        )

    # Guardar archivo
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}{ext}")

    with open(file_path, "wb") as f:
        f.write(content)

    # Pipeline de ingesta
    service = get_ingestion_service()
    result = await service.ingest_document(
        file_path=file_path,
        titulo=titulo,
        categoria=categoria,
        tipo_archivo=ext.lstrip("."),
        tamano_kb=tamano_kb,
        subido_por_id=subido_por_id,
        parcela_id=parcela_id,
        idioma=idioma,
    )

    return DocumentUploadResponse(
        documento_id=result["documento_id"],
        titulo=titulo,
        tipo_archivo=ext.lstrip("."),
        tamano_kb=tamano_kb,
        estado_indexacion=EstadoIndexacion(result["estado"]),
        chunks_generados=result["chunks_generados"],
        mensaje=result["mensaje"],
    )


@router.get(
    "",
    response_model=DocumentsListResponse,
    summary="Listar documentos indexados",
)
async def list_documents() -> DocumentsListResponse:
    from src.api.routes.health import get_rag_service
    service = get_rag_service()
    return service.get_documents_list()


@router.delete(
    "/{documento_id}",
    summary="Eliminar documento",
)
async def delete_document(documento_id: str) -> JSONResponse:
    service = get_ingestion_service()
    success = service.delete_document(documento_id)
    if not success:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return JSONResponse({"mensaje": "Documento eliminado correctamente"})