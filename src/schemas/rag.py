from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum


# ── Enums ──

class EstadoIndexacion(str, Enum):
    PENDIENTE = "pendiente"
    PROCESANDO = "procesando"
    INDEXADO = "indexado"
    FALLIDO = "fallido"
    EXCLUIDO = "excluido"


# ── Query (consulta RAG) ──

class QueryRequest(BaseModel):
    """Request para consulta RAG."""
    pregunta: str = Field(..., description="Pregunta en español")
    top_k: int = Field(default=5, ge=1, le=20)
    filtro_categoria: Optional[str] = Field(default=None)
    # Contexto del agricultor para personalización
    cultivo: Optional[str] = Field(default=None, description="Cultivo del agricultor: cafe, cacao")
    region: Optional[str] = Field(default=None, description="Departamento del agricultor")
    nombre_agricultor: Optional[str] = Field(default=None, description="Nombre del agricultor")
    parcela_nombre: Optional[str] = Field(default=None, description="Nombre de la parcela")


class SourceReference(BaseModel):
    """Referencia a la fuente de una respuesta."""
    documento_id: str | None = None
    titulo: str
    pagina: int | None = None
    institucion: str | None = Field(default=None, description="Institución autora del documento")
    chunk_text: str = Field(..., description="Fragmento relevante del documento")
    score: float = Field(..., description="Score de similitud")


class QueryResponse(BaseModel):
    """Respuesta del sistema RAG con citación de fuentes."""
    respuesta: str = Field(..., description="Respuesta generada por el LLM")
    fuentes: list[SourceReference] = Field(default_factory=list)
    pregunta_original: str
    modelo_usado: str
    tokens_usados: Optional[int] = None
    tiempo_respuesta_ms: float
    relevancia_pct: Optional[float] = Field(default=None, description="Score de relevancia promedio %")
    timestamp: datetime


# ── Documents ──

class DocumentUploadResponse(BaseModel):
    """Respuesta al subir un documento para indexación."""
    documento_id: str
    titulo: str
    tipo_archivo: str
    tamano_kb: int
    estado_indexacion: EstadoIndexacion
    chunks_generados: int = 0
    mensaje: str


class DocumentInfo(BaseModel):
    """Información de un documento indexado."""
    documento_id: str
    titulo: str
    categoria: str
    tipo_archivo: str
    tamano_kb: int
    idioma: str = "es"
    estado_indexacion: EstadoIndexacion
    chunks: int = 0
    fecha_indexacion: Optional[datetime] = None
    creado_en: datetime


class DocumentsListResponse(BaseModel):
    """Lista de documentos indexados."""
    documentos: list[DocumentInfo]
    total: int
    total_chunks: int


# ── Health ──

class HealthResponse(BaseModel):
    """Health check del servicio RAG."""
    status: str = "ok"
    service: str = "agrovision-rag"
    version: str
    environment: str
    chromadb_connected: bool = False
    chromadb_collections: int = 0
    chromadb_total_documents: int = 0
    openai_configured: bool = False
    database_connected: bool = False
    uptime_seconds: float = 0
    timestamp: datetime
