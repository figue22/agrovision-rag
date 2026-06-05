import time
import logging
from datetime import datetime

from src.config.settings import get_settings
from src.vectorstore.chroma_store import ChromaStore
from src.embeddings.gemini_embeddings import GeminiEmbeddings
from src.schemas.rag import (
    QueryRequest, QueryResponse, SourceReference, DocumentsListResponse,
)

logger = logging.getLogger("agrovision-rag")


class RAGService:
    """Servicio principal del pipeline RAG."""

    def __init__(self):
        self.settings = get_settings()
        self._chroma = None
        self._embeddings = None
        self._start_time = time.time()

    def _get_chroma(self) -> ChromaStore:
        if self._chroma is None:
            self._chroma = ChromaStore()
            self._chroma.connect()
        return self._chroma

    def _get_embeddings(self) -> GeminiEmbeddings:
        if self._embeddings is None:
            self._embeddings = GeminiEmbeddings()
        return self._embeddings

    @property
    def uptime(self) -> float:
        return time.time() - self._start_time

    @property
    def openai_configured(self) -> bool:
        return (
            self.settings.GOOGLE_API_KEY != "your-google-api-key"
            and len(self.settings.GOOGLE_API_KEY) > 10
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        start = time.time()
        chroma = self._get_chroma()
        embeddings = self._get_embeddings()

        # Generar embedding de la query con Gemini
        logger.info("Generando embedding para query: %s", request.pregunta[:50])
        query_embedding = embeddings.embed_query(request.pregunta)

        where_filter = None
        if request.filtro_categoria:
            where_filter = {"categoria": request.filtro_categoria}

        # Buscar por embedding en ChromaDB
        results = chroma.query(
            query_embedding=query_embedding,
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
            score = max(0, 1 - distance)

            fuentes.append(SourceReference(
                documento_id=metadata.get("documento_id"),
                titulo=metadata.get("titulo", "Documento sin título"),
                pagina=metadata.get("pagina"),
                chunk_text=doc_text[:500],
                score=round(score, 4),
            ))

        if fuentes:
            respuesta = (
                f"Encontré {len(fuentes)} fragmentos relevantes sobre tu consulta. "
                "La generación de respuestas con LLM se habilitará en la próxima HU. "
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
            modelo_usado=self.settings.LLM_MODEL,
            tokens_usados=None,
            tiempo_respuesta_ms=duration_ms,
            timestamp=datetime.utcnow(),
        )

    def get_documents_list(self) -> DocumentsListResponse:
        chroma = self._get_chroma()
        total_chunks = chroma.document_count

        return DocumentsListResponse(
            documentos=[],
            total=0,
            total_chunks=total_chunks,
        )
