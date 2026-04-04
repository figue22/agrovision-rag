import logging
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config.settings import get_settings

logger = logging.getLogger("agrovision-rag")


class ChromaStore:
    """Gestión de ChromaDB con persistent storage."""

    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    def connect(self) -> None:
        """Inicializa el cliente ChromaDB con almacenamiento persistente."""
        try:
            self._client = chromadb.PersistentClient(
                path=self.settings.CHROMA_PERSIST_DIRECTORY,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.settings.CHROMA_COLLECTION_NAME,
                metadata={"description": "Documentos agrícolas AgroVision"},
            )
            logger.info(
                "ChromaDB conectado — colección '%s' con %d documentos",
                self.settings.CHROMA_COLLECTION_NAME,
                self._collection.count(),
            )
        except Exception as e:
            logger.error("Error conectando ChromaDB: %s", str(e))
            self._client = None
            self._collection = None

    @property
    def is_connected(self) -> bool:
        return self._client is not None and self._collection is not None

    @property
    def collection(self):
        return self._collection

    @property
    def document_count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    @property
    def collection_count(self) -> int:
        if self._client is None:
            return 0
        return len(self._client.list_collections())

    def query(self, query_text: str, n_results: int = 5, where: dict = None) -> dict:
        """Busca chunks similares a la consulta."""
        if not self.is_connected:
            return {"documents": [], "metadatas": [], "distances": []}

        params = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            params["where"] = where

        return self._collection.query(**params)

    def add_documents(
        self,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]] = None,
    ) -> None:
        """Agrega documentos (chunks) a la colección."""
        if not self.is_connected:
            raise RuntimeError("ChromaDB no está conectado")

        params = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if embeddings:
            params["embeddings"] = embeddings

        self._collection.add(**params)

    def delete_by_document_id(self, documento_id: str) -> None:
        """Elimina todos los chunks de un documento."""
        if not self.is_connected:
            return
        self._collection.delete(where={"documento_id": documento_id})
