"""
HU-044 — Pipeline de ingesta: PDF/DOCX -> chunks -> embeddings -> ChromaDB
"""

import logging
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional

import tiktoken

from src.config.settings import get_settings

logger = logging.getLogger("agrovision-rag")


@dataclass
class Chunk:
    """Representa un fragmento de texto con metadata."""
    chunk_id: str
    documento_id: str
    texto: str
    pagina: Optional[int]
    posicion: int
    tokens: int
    metadata: dict


class DocumentPipeline:
    """
    Pipeline completo de ingesta de documentos.

    Flujo: archivo -> texto -> chunks -> embeddings -> ChromaDB
    """

    def __init__(self):
        self.settings = get_settings()
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.tokenizer = None

    def _count_tokens(self, text: str) -> int:
        """Cuenta tokens de un texto."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def load_pdf(self, file_path: str) -> List[dict]:
        """Extrae texto de un PDF usando PyPDF2."""
        try:
            import PyPDF2
            paginas = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    texto = page.extract_text() or ""
                    texto = texto.strip()
                    if texto:
                        paginas.append({"pagina": i + 1, "texto": texto})
            logger.info("PDF cargado: %d páginas con texto", len(paginas))
            return paginas
        except Exception as e:
            logger.error("Error cargando PDF %s: %s", file_path, e)
            raise

    def load_docx(self, file_path: str) -> List[dict]:
        """Extrae texto de un DOCX usando python-docx."""
        try:
            from docx import Document
            doc = Document(file_path)
            paginas = []
            texto_acum = []
            pagina_actual = 1

            for para in doc.paragraphs:
                texto = para.text.strip()
                if texto:
                    texto_acum.append(texto)
                    # Simular salto de página cada 40 párrafos
                    if len(texto_acum) % 40 == 0:
                        paginas.append({
                            "pagina": pagina_actual,
                            "texto": "\n".join(texto_acum[-40:]),
                        })
                        pagina_actual += 1

            # Resto del texto
            if texto_acum:
                texto_final = "\n".join(texto_acum)
                if not paginas or paginas[-1]["texto"] != texto_final:
                    paginas.append({"pagina": pagina_actual, "texto": texto_final})

            logger.info("DOCX cargado: %d secciones", len(paginas))
            return paginas
        except Exception as e:
            logger.error("Error cargando DOCX %s: %s", file_path, e)
            raise

    def chunk_text(
        self,
        paginas: List[dict],
        documento_id: str,
        metadata_base: dict,
    ) -> List[Chunk]:
        chunks = []
        posicion = 0

        # Unir todo el texto
        texto_completo = "\n".join(p["texto"] for p in paginas if p["texto"].strip())
        if not texto_completo.strip():
            return chunks

        palabras = texto_completo.split()
        total = len(palabras)
        paso = 300   # palabras por chunk
        overlap = 50  # palabras de overlap

        logger.info("Total palabras: %d, paso: %d, overlap: %d", total, paso, overlap)

        i = 0
        while i < total:
            fin = min(i + paso, total)
            chunk_texto = " ".join(palabras[i:fin]).strip()

            if chunk_texto:
                chunk_id = f"{documento_id}_chunk_{posicion}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    documento_id=documento_id,
                    texto=chunk_texto,
                    pagina=1,
                    posicion=posicion,
                    tokens=self._count_tokens(chunk_texto),
                    metadata={
                        **metadata_base,
                        "documento_id": documento_id,
                        "pagina": 1,
                        "posicion": posicion,
                        "tokens": self._count_tokens(chunk_texto),
                    },
                ))
                posicion += 1

            # Si llegamos al final salimos
            if fin == total:
                break

            # Avanzar con overlap
            i = fin - overlap

        logger.info("Chunking completado: %d chunks", len(chunks))
        return chunks

    def process_file(
        self,
        file_path: str,
        documento_id: str,
        metadata: dict,
    ) -> List[Chunk]:
        """
        Procesa un archivo (PDF o DOCX) y retorna lista de chunks.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            paginas = self.load_pdf(file_path)
        elif ext in (".docx", ".doc"):
            paginas = self.load_docx(file_path)
        else:
            raise ValueError(f"Tipo de archivo no soportado: {ext}")

        if not paginas:
            raise ValueError("El archivo no contiene texto extraíble")

        return self.chunk_text(paginas, documento_id, metadata)