import time
import logging
from datetime import datetime

from src.config.settings import get_settings
from src.vectorstore.chroma_store import ChromaStore
from src.schemas.rag import (
    QueryRequest, QueryResponse, SourceReference, DocumentsListResponse,
)


logger = logging.getLogger("agrovision-rag")


class RAGService:
    """Servicio principal del pipeline RAG.

    En esta fase (HU-013) se implementa la estructura base con ChromaDB.
    La integración con LangChain chains, embeddings de OpenAI y
    generación de respuestas se completa en las HU-044 a HU-050.
    """

    def __init__(self):
        self.settings = get_settings()
        self.chroma = ChromaStore()
        self._start_time = time.time()
        self._initialize()

    def _initialize(self) -> None:
        """Conecta a ChromaDB al iniciar."""
        self.chroma.connect

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    @property
    def openai_configured(self) -> bool:
        return (
            self.settings.OPENAI_API_KEY != "your-openai-api-key"
            and len(self.settings.OPENAI_API_KEY) > 10
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        """Procesa una consulta RAG.

        En esta fase retorna resultados de búsqueda vectorial sin LLM.
        La generación con LangChain se integra en HU-046.
        """
        start = time.time()

        # Buscar chunks relevantes en ChromaDB
        where_filter = None
        if request.filtro_categoria:
            where_filter = {"categoria": request.filtro_categoria}

        results = self.chroma.query(
            query_text=request.pregunta,
            n_results=request.top_k,
            where=where_filter,
        )

        # Construir fuentes
        fuentes = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 0
            score = max(0, 1 - distance)  # Convertir distancia a score de similitud

            fuentes.append(SourceReference(
                documento_id=metadata.get("documento_id"),
                titulo=metadata.get("titulo", "Documento sin título"),
                pagina=metadata.get("pagina"),
                chunk_text=doc_text[:500],
                score=round(score, 4),
            ))

        # Generar respuesta
        if fuentes:
            respuesta = (
                f"Encontré {len(fuentes)} fragmentos relevantes sobre tu consulta. "
                "La generación de respuestas con LLM se habilitará en la HU-046. "
                "Por ahora, revisa los fragmentos fuente a continuación."
            )
        else:
            respuesta = (
                "No encontré documentos relevantes para tu consulta. "
                "Asegúrate de que haya documentos indexados en el sistema."
            )

        duration_ms = round((time.time() - start) * 1000, 2)

        return QueryResponse(
            respuesta=respuesta,
            fuentes=fuentes,
            pregunta_original=request.pregunta,
            modelo_usado="chromadb-similarity" if not self.openai_configured else self.settings.LLM_MODEL,
            tokens_usados=None,
            tiempo_respuesta_ms=duration_ms,
            timestamp=datetime.utcnow(),
        )

    def get_documents_list(self) -> DocumentsListResponse:
        """Lista los documentos indexados en ChromaDB."""
        # En esta fase retorna info básica desde ChromaDB
        # La integración con la tabla `documentos` de PostgreSQL se hace en HU-047
        total_chunks = self.chroma.document_count

        return DocumentsListResponse(
            documentos=[],
            total=0,
            total_chunks=total_chunks,
        )
