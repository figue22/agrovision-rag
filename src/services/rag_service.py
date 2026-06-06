import time
import logging
from datetime import datetime

import google.generativeai as genai

from src.config.settings import get_settings
from src.vectorstore.chroma_store import ChromaStore
from src.embeddings.gemini_embeddings import GeminiEmbeddings
from src.schemas.rag import (
    QueryRequest, QueryResponse, SourceReference, DocumentsListResponse,
)

logger = logging.getLogger("agrovision-rag")

SYSTEM_PROMPT = """Eres AgroVision, un asistente agrícola especializado en cultivos colombianos,
especialmente café y cacao. Tu función es responder preguntas de agricultores colombianos
basándote ÚNICAMENTE en los documentos técnicos proporcionados como contexto.

Reglas:
1. Responde SOLO con información del contexto proporcionado
2. Si la información no está en el contexto, dilo claramente
3. Cita las fuentes usando [Fuente N] al final de cada afirmación relevante
4. Usa lenguaje claro y práctico para agricultores
5. Responde siempre en español
6. Sé conciso pero completo (máximo 3-4 párrafos)"""


class RAGService:
    """Servicio principal del pipeline RAG con generación Gemini."""

    def __init__(self):
        self.settings = get_settings()
        self._chroma = None
        self._embeddings = None
        self._llm = None
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

    def _get_llm(self):
        if self._llm is None:
            genai.configure(api_key=self.settings.GOOGLE_API_KEY)
            self._llm = genai.GenerativeModel(
                model_name=self.settings.LLM_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.settings.LLM_TEMPERATURE,
                    max_output_tokens=self.settings.LLM_MAX_TOKENS,
                ),
            )
        return self._llm

    def _assemble_context(self, fuentes: list[SourceReference]) -> str:
        """Ensambla el contexto a partir de los chunks recuperados."""
        if not fuentes:
            return ""
        partes = []
        for i, f in enumerate(fuentes, 1):
            partes.append(
                f"[Fuente {i}] — {f.titulo} (página {f.pagina or 'N/A'}):\n{f.chunk_text}"
            )
        return "\n\n".join(partes)

    def _inject_citations(self, respuesta: str, fuentes: list[SourceReference]) -> str:
        """Verifica que las citas estén en la respuesta, agrega lista al final."""
        referencias = "\n\n**Referencias:**"
        for i, f in enumerate(fuentes, 1):
            referencias += f"\n[Fuente {i}] {f.titulo}"
            if f.pagina:
                referencias += f", página {f.pagina}"
        return respuesta + referencias

    def _calculate_relevance(self, distances: list) -> float:
        """Calcula score de relevancia promedio (0-100%)."""
        if not distances:
            return 0.0
        scores = [max(0, 1 - d) for d in distances]
        return round(sum(scores) / len(scores) * 100, 1)

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

        # ── PASO 1: Query processing ──
        logger.info("RAG query: '%s'", request.pregunta[:60])

        # ── PASO 2: Embedding de la query ──
        embeddings = self._get_embeddings()
        query_embedding = embeddings.embed_query(request.pregunta)

        # ── PASO 3: Similarity search (cosine, top-k=5) ──
        chroma = self._get_chroma()
        where_filter = None
        if request.filtro_categoria:
            where_filter = {"categoria": request.filtro_categoria}

        results = chroma.query(
            query_embedding=query_embedding,
            n_results=request.top_k,
            where=where_filter,
        )

        # ── PASO 4: Construir fuentes ──
        fuentes = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_text in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0, 1 - distance)

            fuentes.append(SourceReference(
                documento_id=metadata.get("documento_id"),
                titulo=metadata.get("titulo", "Documento sin título"),
                pagina=metadata.get("pagina"),
                chunk_text=doc_text[:500],
                score=round(score, 4),
            ))

        relevancia = self._calculate_relevance(distances)
        logger.info(
            "Chunks recuperados: %d | Relevancia: %.1f%%",
            len(fuentes), relevancia,
        )

        # ── PASO 5: Context assembly ──
        contexto = self._assemble_context(fuentes)

        # ── PASO 6: Generation con Gemini ──
        respuesta = ""
        tokens_usados = None

        if fuentes and self.openai_configured:
            try:
                prompt = f"""{SYSTEM_PROMPT}

Contexto de documentos técnicos agrícolas:

{contexto}

Pregunta del agricultor: {request.pregunta}

Responde basándote únicamente en el contexto anterior. Cita las fuentes usando [Fuente N]."""

                llm = self._get_llm()
                response = llm.generate_content(prompt)
                respuesta = response.text

                # Contar tokens aproximados
                tokens_usados = len(prompt.split()) + len(respuesta.split())

                logger.info("Respuesta generada: %d tokens aprox.", tokens_usados)

            except Exception as e:
                logger.error("Error en generación LLM: %s", e)
                respuesta = (
                    f"Encontré {len(fuentes)} fragmentos relevantes pero hubo un error "
                    f"al generar la respuesta: {str(e)}"
                )
        elif not fuentes:
            respuesta = (
                "No encontré documentos relevantes para tu consulta. "
                "Asegúrate de que haya documentos indexados sobre este tema."
            )
        else:
            respuesta = (
                f"Encontré {len(fuentes)} fragmentos relevantes. "
                "Configura GOOGLE_API_KEY para habilitar respuestas generadas."
            )

        # ── PASO 7: Citation injection ──
        if fuentes and respuesta:
            respuesta = self._inject_citations(respuesta, fuentes)

        duration_ms = round((time.time() - start) * 1000, 2)
        logger.info("Query completado en %.1fms | Relevancia: %.1f%%", duration_ms, relevancia)

        return QueryResponse(
            respuesta=respuesta,
            fuentes=fuentes,
            pregunta_original=request.pregunta,
            modelo_usado=self.settings.LLM_MODEL,
            tokens_usados=tokens_usados,
            tiempo_respuesta_ms=duration_ms,
            relevancia_pct=relevancia,
            timestamp=datetime.utcnow(),
        )

    def get_documents_list(self) -> DocumentsListResponse:
        chroma = self._get_chroma()
        return DocumentsListResponse(
            documentos=[],
            total=0,
            total_chunks=chroma.document_count,
        )
