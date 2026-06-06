import time
import logging
from datetime import datetime
from typing import Optional

import google.generativeai as genai

from src.config.settings import get_settings
from src.vectorstore.chroma_store import ChromaStore
from src.embeddings.gemini_embeddings import GeminiEmbeddings
from src.schemas.rag import (
    QueryRequest, QueryResponse, SourceReference, DocumentsListResponse,
)

logger = logging.getLogger("agrovision-rag")

SYSTEM_PROMPT = """Eres AgroVision, un asistente agrícola especializado en cultivos colombianos.
Tu función es responder preguntas de agricultores colombianos basándote ÚNICAMENTE en los
documentos técnicos proporcionados como contexto.

Reglas:
1. Responde SOLO con información del contexto proporcionado
2. Si la información no está en el contexto, dilo claramente
3. Cita las fuentes usando [Fuente N] al final de cada afirmación relevante
4. Usa lenguaje claro y práctico para agricultores colombianos
5. Responde siempre en español
6. Sé conciso pero completo (máximo 3-4 párrafos)
7. Cuando tengas contexto del agricultor, personaliza la respuesta para su situación específica"""


def build_personalized_prompt(
    pregunta: str,
    contexto: str,
    cultivo: Optional[str] = None,
    region: Optional[str] = None,
    nombre_agricultor: Optional[str] = None,
    parcela_nombre: Optional[str] = None,
) -> str:
    """Construye prompt personalizado según el contexto del agricultor."""

    # Contexto del agricultor
    contexto_agricultor = ""
    if any([nombre_agricultor, cultivo, region, parcela_nombre]):
        partes = []
        if nombre_agricultor:
            partes.append(f"Agricultor: {nombre_agricultor}")
        if cultivo:
            cultivo_nombre = {"cafe": "café", "cacao": "cacao"}.get(cultivo, cultivo)
            partes.append(f"Cultivo principal: {cultivo_nombre}")
        if region:
            partes.append(f"Región/Departamento: {region}")
        if parcela_nombre:
            partes.append(f"Parcela: {parcela_nombre}")
        contexto_agricultor = "**Contexto del agricultor:**\n" + "\n".join(partes) + "\n\n"

    prompt = f"""{SYSTEM_PROMPT}

{contexto_agricultor}**Documentos técnicos de referencia:**

{contexto}

**Pregunta:** {pregunta}

Responde de forma personalizada considerando el contexto del agricultor si está disponible.
Cita las fuentes usando [Fuente N] cuando uses información específica de los documentos."""

    return prompt


class RAGService:
    """Servicio principal del pipeline RAG con personalización por cultivo y región."""

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

    def _build_where_filter(
        self,
        filtro_categoria: Optional[str],
        cultivo: Optional[str],
        region: Optional[str],
    ) -> Optional[dict]:
        """
        Construye filtro de metadata para ChromaDB.
        Filtra por cultivo y/o categoría si están disponibles.
        """
        conditions = []

        if filtro_categoria:
            conditions.append({"categoria": {"$eq": filtro_categoria}})

        if cultivo:
            # Buscar documentos específicos del cultivo o generales
            conditions.append({
                "$or": [
                    {"categoria": {"$eq": cultivo}},
                    {"categoria": {"$eq": "general"}},
                ]
            })

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def _assemble_context(self, fuentes: list) -> str:
        if not fuentes:
            return ""
        partes = []
        for i, f in enumerate(fuentes, 1):
            inst = f" ({f.institucion})" if f.institucion else ""
            partes.append(
                f"[Fuente {i}] — {f.titulo}{inst} (página {f.pagina or 'N/A'}):\n{f.chunk_text}"
            )
        return "\n\n".join(partes)

    def _inject_citations(self, respuesta: str, fuentes: list) -> str:
        referencias = "\n\n**Referencias:**"
        for i, f in enumerate(fuentes, 1):
            inst = f" — {f.institucion}" if f.institucion else ""
            referencias += f"\n[Fuente {i}] {f.titulo}{inst}"
            if f.pagina:
                referencias += f", página {f.pagina}"
        return respuesta + referencias

    def _calculate_relevance(self, distances: list) -> float:
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

        logger.info(
            "RAG query: '%s' | cultivo=%s | region=%s",
            request.pregunta[:60],
            request.cultivo or "N/A",
            request.region or "N/A",
        )

        # ── PASO 1: Embedding de la query ──
        embeddings = self._get_embeddings()
        query_embedding = embeddings.embed_query(request.pregunta)

        # ── PASO 2: Metadata filtering por cultivo y región ──
        chroma = self._get_chroma()
        where_filter = self._build_where_filter(
            filtro_categoria=request.filtro_categoria,
            cultivo=request.cultivo,
            region=request.region,
        )

        logger.info("Filtro ChromaDB: %s", where_filter)

        # Primer intento con filtro
        results = chroma.query(
            query_embedding=query_embedding,
            n_results=request.top_k,
            where=where_filter,
        )

        # Si no hay resultados con filtro, buscar sin filtro
        documents = results.get("documents", [[]])[0]
        if not documents and where_filter:
            logger.info("Sin resultados con filtro, buscando sin filtro...")
            results = chroma.query(
                query_embedding=query_embedding,
                n_results=request.top_k,
                where=None,
            )

        # ── PASO 3: Construir fuentes ──
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
                institucion=metadata.get("institucion"),
                chunk_text=doc_text[:500],
                score=round(score, 4),
            ))

        relevancia = self._calculate_relevance(distances)
        logger.info("Chunks: %d | Relevancia: %.1f%%", len(fuentes), relevancia)

        # ── PASO 4: Context assembly ──
        contexto = self._assemble_context(fuentes)

        # ── PASO 5: Generation personalizada ──
        respuesta = ""
        tokens_usados = None

        if fuentes and self.openai_configured:
            try:
                prompt = build_personalized_prompt(
                    pregunta=request.pregunta,
                    contexto=contexto,
                    cultivo=request.cultivo,
                    region=request.region,
                    nombre_agricultor=request.nombre_agricultor,
                    parcela_nombre=request.parcela_nombre,
                )

                llm = self._get_llm()
                response = llm.generate_content(prompt)
                respuesta = response.text
                tokens_usados = len(prompt.split()) + len(respuesta.split())

                logger.info("Respuesta generada: %d tokens aprox.", tokens_usados)

            except Exception as e:
                logger.error("Error en generación LLM: %s", e)
                respuesta = f"Error al generar respuesta: {str(e)}"
        elif not fuentes:
            respuesta = (
                "No encontré documentos relevantes para tu consulta. "
                "Asegúrate de que haya documentos indexados sobre este tema."
            )
        else:
            respuesta = f"Encontré {len(fuentes)} fragmentos relevantes."

        # ── PASO 6: Citation injection ──
        if fuentes and respuesta:
            respuesta = self._inject_citations(respuesta, fuentes)

        duration_ms = round((time.time() - start) * 1000, 2)
        logger.info("Query completado en %.1fms", duration_ms)

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
