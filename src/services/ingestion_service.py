"""
HU-044 — Servicio de ingesta de documentos
Orquesta: carga -> chunking -> embeddings -> ChromaDB -> BD
"""

import logging
import os
import uuid
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config.settings import get_settings
from src.document_loaders.pipeline import DocumentPipeline
from src.embeddings.gemini_embeddings import GeminiEmbeddings
from src.vectorstore.chroma_store import ChromaStore
from src.schemas.rag import EstadoIndexacion

logger = logging.getLogger("agrovision-rag")


class IngestionService:
    """
    Servicio de ingesta completo.

    Flujo:
    1. Guardar metadata en tabla documentos (PostgreSQL)
    2. Procesar archivo (PDF/DOCX) → chunks
    3. Generar embeddings con Gemini
    4. Indexar en ChromaDB con metadata
    5. Actualizar estado en BD
    """

    def __init__(self):
        self.settings = get_settings()
        self.pipeline = DocumentPipeline()
        self._embeddings = None
        self._chroma = None
        self._engine = None
        self._session = None
        self._init_db()
        os.makedirs(self.settings.UPLOAD_DIR, exist_ok=True)

    @property
    def embeddings(self):
        if self._embeddings is None:
            self._embeddings = GeminiEmbeddings()
        return self._embeddings

    @property
    def chroma(self):
        if self._chroma is None:
            self._chroma = ChromaStore()
            self._chroma.connect()
        return self._chroma

    def _init_db(self):
        """Inicializa conexión a PostgreSQL."""
        try:
            self._engine = create_engine(self.settings.DATABASE_URL)
            Session = sessionmaker(bind=self._engine)
            self._session = Session()
            logger.info("Conexión a PostgreSQL establecida")
        except Exception as e:
            logger.error("Error conectando a PostgreSQL: %s", e)
            self._engine = None
            self._session = None

    def _save_documento_bd(
        self,
        documento_id: str,
        titulo: str,
        categoria: str,
        ruta_archivo: str,
        tipo_archivo: str,
        tamano_kb: int,
        subido_por_id: Optional[str],
        parcela_id: Optional[str],
        idioma: str = "es",
    ) -> bool:
        """Guarda metadata del documento en la tabla documentos."""
        if not self._session:
            return False
        try:
            self._session.execute(text("""
                INSERT INTO documentos (
                    documento_id, parcela_id, subido_por_id, titulo, categoria,
                    ruta_archivo, tipo_archivo, tamano_kb, idioma,
                    estado_indexacion, esta_activo, creado_en, actualizado_en
                ) VALUES (
                    :documento_id, :parcela_id, :subido_por_id, :titulo, :categoria,
                    :ruta_archivo, :tipo_archivo, :tamano_kb, :idioma,
                    :estado_indexacion, true, NOW(), NOW()
                )
            """), {
                "documento_id": documento_id,
                "parcela_id": parcela_id,
                "subido_por_id": subido_por_id,
                "titulo": titulo,
                "categoria": categoria,
                "ruta_archivo": ruta_archivo,
                "tipo_archivo": tipo_archivo,
                "tamano_kb": tamano_kb,
                "idioma": idioma,
                "estado_indexacion": EstadoIndexacion.PROCESANDO.value,
            })
            self._session.commit()
            return True
        except Exception as e:
            logger.error("Error guardando documento en BD: %s", e)
            self._session.rollback()
            return False

    def _update_estado(
        self,
        documento_id: str,
        estado: EstadoIndexacion,
        chunks: int = 0,
    ) -> None:
        """Actualiza el estado de indexación en la BD."""
        if not self._session:
            return
        try:
            self._session.execute(text("""
                UPDATE documentos
                SET estado_indexacion = :estado,
                    chunks_indexados = :chunks,
                    actualizado_en = NOW(),
                    fecha_indexacion = CASE WHEN :estado = 'indexado' THEN NOW() ELSE fecha_indexacion END
                WHERE documento_id = :documento_id
            """), {
                "estado": estado.value,
                "chunks": chunks,
                "documento_id": documento_id,
            })
            self._session.commit()
        except Exception as e:
            logger.error("Error actualizando estado: %s", e)
            self._session.rollback()

    async def ingest_document(
        self,
        file_path: str,
        titulo: str,
        categoria: str,
        tipo_archivo: str,
        tamano_kb: int,
        subido_por_id: Optional[str] = None,
        parcela_id: Optional[str] = None,
        idioma: str = "es",
    ) -> dict:
        documento_id = str(uuid.uuid4())
        logger.info("PASO 1: Iniciando ingesta: %s (%s)", titulo, documento_id)

        self._save_documento_bd(
            documento_id=documento_id,
            titulo=titulo,
            categoria=categoria,
            ruta_archivo=file_path,
            tipo_archivo=tipo_archivo,
            tamano_kb=tamano_kb,
            subido_por_id=subido_por_id,
            parcela_id=parcela_id,
            idioma=idioma,
        )
        logger.info("PASO 2: Metadata guardada en BD")

        try:
            metadata_base = {
                "titulo": titulo,
                "categoria": categoria,
                "tipo_archivo": tipo_archivo,
                "idioma": idioma,
                "documento_id": documento_id,
            }
            if parcela_id:
                metadata_base["parcela_id"] = parcela_id

            logger.info("PASO 3: Iniciando pipeline de chunks")
            chunks = self.pipeline.process_file(file_path, documento_id, metadata_base)
            logger.info("PASO 4: Chunks generados: %d", len(chunks))

            if not chunks:
                self._update_estado(documento_id, EstadoIndexacion.FALLIDO)
                return {
                    "documento_id": documento_id,
                    "chunks_generados": 0,
                    "estado": EstadoIndexacion.FALLIDO.value,
                    "mensaje": "No se pudo extraer texto del documento",
                }

            textos = [c.texto for c in chunks]
            logger.info("PASO 5: Iniciando embeddings para %d textos", len(textos))
            embeddings = self.embeddings.embed_documents(textos)
            logger.info("PASO 6: Embeddings completados: %d", len(embeddings))

            ids = [c.chunk_id for c in chunks]
            metadatas = [c.metadata for c in chunks]

            logger.info("PASO 7: Indexando en ChromaDB")
            self.chroma.delete_by_document_id(documento_id)
            self.chroma.add_documents(
                ids=ids,
                documents=textos,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            logger.info("PASO 8: Indexados %d chunks en ChromaDB", len(chunks))

            self._update_estado(documento_id, EstadoIndexacion.INDEXADO, len(chunks))
            logger.info("PASO 9: Estado actualizado en BD")

            return {
                "documento_id": documento_id,
                "chunks_generados": len(chunks),
                "estado": EstadoIndexacion.INDEXADO.value,
                "mensaje": f"Documento indexado exitosamente con {len(chunks)} chunks",
            }

        except Exception as e:
            logger.error("Error en pipeline de ingesta: %s", e)
            self._update_estado(documento_id, EstadoIndexacion.FALLIDO)
            return {
                "documento_id": documento_id,
                "chunks_generados": 0,
                "estado": EstadoIndexacion.FALLIDO.value,
                "mensaje": f"Error en ingesta: {str(e)}",
            }

    def delete_document(self, documento_id: str) -> bool:
        """Elimina un documento de ChromaDB y BD."""
        try:
            self.chroma.delete_by_document_id(documento_id)
            if self._session:
                self._session.execute(text("""
                    UPDATE documentos
                    SET esta_activo = false, actualizado_en = NOW()
                    WHERE documento_id = :documento_id
                """), {"documento_id": documento_id})
                self._session.commit()
            return True
        except Exception as e:
            logger.error("Error eliminando documento: %s", e)
            return False
