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

# ── Historial ──

class MensajeHistorial(BaseModel):
    """Mensaje del historial de conversación."""
    rol: str = Field(..., description="'usuario' o 'asistente'")
    contenido: str

# ── Query (consulta RAG) ──

class QueryRequest(BaseModel):
    """Request para consulta RAG."""
    pregunta: str = Field(..., description="Pregunta en español")
    top_k: int = Field(default=5, ge=1, le=20)
    filtro_categoria: Optional[str] = Field(default=None)
    cultivo: Optional[str] = Field(default=None, description="Cultivo del agricultor: platano, cacao")
    region: Optional[str] = Field(default=None, description="Departamento del agricultor")
    nombre_agricultor: Optional[str] = Field(default=None, description="Nombre del agricultor")
    parcela_nombre: Optional[str] = Field(default=None, description="Nombre de la parcela")
    historial: Optional[list[MensajeHistorial]] = Field(
        default=None,
        description="Últimos mensajes de la conversación para mantener contexto",
    )

# ── resto del archivo igual ──

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

class DocumentUploadResponse(BaseModel):
    documento_id: str
    titulo: str
    tipo_archivo: str
    tamano_kb: int
    estado_indexacion: EstadoIndexacion
    chunks_generados: int = 0
    mensaje: str

class DocumentInfo(BaseModel):
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
    documentos: list[DocumentInfo]
    total: int
    total_chunks: int

class HealthResponse(BaseModel):
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
    
# ── Feedback y Métricas RAG ──

class FeedbackRequest(BaseModel):
    """Request para enviar feedback de una respuesta RAG."""
    consulta_id: str = Field(..., description="ID único de la consulta")
    pregunta: str = Field(..., description="Pregunta original")
    respuesta: str = Field(..., description="Respuesta recibida")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5 estrellas")
    comentario: Optional[str] = Field(default=None)
    usuario_id: Optional[str] = Field(default=None)
    cultivo: Optional[str] = Field(default=None)
    region: Optional[str] = Field(default=None)
    documentos_citados: Optional[list[str]] = Field(default_factory=list)
    tiempo_respuesta_ms: Optional[float] = Field(default=None)
    relevancia_pct: Optional[float] = Field(default=None)

class FeedbackResponse(BaseModel):
    """Respuesta al enviar feedback."""
    feedback_id: str
    mensaje: str = "Feedback registrado correctamente"

class MetricasRAG(BaseModel):
    """Métricas del sistema RAG."""
    total_consultas: int
    satisfaccion_promedio: float
    consultas_con_feedback: int
    rating_1: int
    rating_2: int
    rating_3: int
    rating_4: int
    rating_5: int
    consultas_frecuentes: list[dict]
    documentos_mas_citados: list[dict]
    relevancia_promedio_pct: float
    tiempo_respuesta_promedio_ms: float