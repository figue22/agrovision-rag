"""
HU-050 — Servicio de feedback y métricas RAG
"""

import json
import logging
import uuid
from typing import Optional

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.config.settings import get_settings
from src.schemas.rag import FeedbackRequest, MetricasRAG

logger = logging.getLogger("agrovision-rag")


class FeedbackService:
    """Gestiona el feedback de calidad y las métricas del sistema RAG."""

    def __init__(self):
        self.settings = get_settings()
        self._session = None
        self._init_db()
        self._ensure_table()

    def _init_db(self):
        try:
            engine = create_engine(self.settings.DATABASE_URL)
            Session = sessionmaker(bind=engine)
            self._session = Session()
            logger.info("FeedbackService: BD conectada")
        except Exception as e:
            logger.error("FeedbackService: error BD: %s", e)

    def _ensure_table(self):
        """Crea la tabla de feedback si no existe."""
        if not self._session:
            return
        try:
            self._session.execute(text("""
                CREATE TABLE IF NOT EXISTS rag_feedback (
                    feedback_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    consulta_id     VARCHAR(100) NOT NULL,
                    pregunta        TEXT NOT NULL,
                    respuesta       TEXT NOT NULL,
                    rating          INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
                    comentario      TEXT,
                    usuario_id      UUID,
                    cultivo         VARCHAR(50),
                    region          VARCHAR(100),
                    documentos_citados JSONB DEFAULT '[]',
                    tiempo_respuesta_ms FLOAT,
                    relevancia_pct  FLOAT,
                    creado_en       TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """))
            self._session.execute(text("""
                CREATE TABLE IF NOT EXISTS rag_consultas (
                    consulta_id     VARCHAR(100) PRIMARY KEY,
                    pregunta        TEXT NOT NULL,
                    cultivo         VARCHAR(50),
                    region          VARCHAR(100),
                    usuario_id      UUID,
                    documentos_citados JSONB DEFAULT '[]',
                    tiempo_respuesta_ms FLOAT,
                    relevancia_pct  FLOAT,
                    creado_en       TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """))
            self._session.commit()
        except Exception as e:
            logger.error("Error creando tablas feedback: %s", e)
            self._session.rollback()

    def registrar_consulta(
        self,
        consulta_id: str,
        pregunta: str,
        documentos_citados: list,
        tiempo_ms: float,
        relevancia_pct: Optional[float],
        cultivo: Optional[str] = None,
        region: Optional[str] = None,
        usuario_id: Optional[str] = None,
    ) -> None:
        """Registra una consulta para métricas."""
        logger.info(
            "Registrando consulta: %s | session=%s",
            consulta_id, self._session is not None,
        )
        if not self._session:
            logger.error("NO HAY SESIÓN DE BD — consulta no registrada")
            return
        try:
            self._session.execute(text("""
                INSERT INTO rag_consultas (
                    consulta_id, pregunta, cultivo, region, usuario_id,
                    documentos_citados, tiempo_respuesta_ms, relevancia_pct, creado_en
                ) VALUES (
                    :consulta_id, :pregunta, :cultivo, :region, :usuario_id,
                    :docs, :tiempo_ms, :relevancia_pct, NOW()
                ) ON CONFLICT (consulta_id) DO NOTHING
            """), {
                "consulta_id": consulta_id,
                "pregunta": pregunta,
                "cultivo": cultivo,
                "region": region,
                "usuario_id": usuario_id,
                "docs": json.dumps(documentos_citados),
                "tiempo_ms": tiempo_ms,
                "relevancia_pct": relevancia_pct,
            })
            self._session.commit()
            logger.info("Consulta registrada exitosamente: %s", consulta_id)
        except Exception as e:
            logger.error("Error registrando consulta: %s", e)
            self._session.rollback()

    def guardar_feedback(self, req: FeedbackRequest) -> str:
        """Guarda el feedback del usuario."""
        if not self._session:
            return str(uuid.uuid4())
        try:
            feedback_id = str(uuid.uuid4())
            self._session.execute(text("""
                INSERT INTO rag_feedback (
                    feedback_id, consulta_id, pregunta, respuesta, rating,
                    comentario, usuario_id, cultivo, region,
                    documentos_citados, tiempo_respuesta_ms, relevancia_pct, creado_en
                ) VALUES (
                    :feedback_id, :consulta_id, :pregunta, :respuesta, :rating,
                    :comentario, :usuario_id, :cultivo, :region,
                    :docs, :tiempo_ms, :relevancia_pct, NOW()
                )
            """), {
                "feedback_id": feedback_id,
                "consulta_id": req.consulta_id,
                "pregunta": req.pregunta,
                "respuesta": req.respuesta[:1000],
                "rating": req.rating,
                "comentario": req.comentario,
                "usuario_id": req.usuario_id,
                "cultivo": req.cultivo,
                "region": req.region,
                "docs": json.dumps(req.documentos_citados or []),
                "tiempo_ms": req.tiempo_respuesta_ms,
                "relevancia_pct": req.relevancia_pct,
            })
            self._session.commit()
            logger.info("Feedback guardado: %s | rating=%d", feedback_id, req.rating)
            return feedback_id
        except Exception as e:
            logger.error("Error guardando feedback: %s", e)
            self._session.rollback()
            return str(uuid.uuid4())

    def get_metricas(self) -> MetricasRAG:
        """Calcula métricas del sistema RAG."""
        if not self._session:
            return self._empty_metricas()
        try:
            # Totales
            total = self._session.execute(
                text("SELECT COUNT(*) FROM rag_consultas")
            ).scalar() or 0

            # Satisfacción
            sat = self._session.execute(text("""
                SELECT
                    COUNT(*) as total,
                    COALESCE(AVG(rating), 0) as promedio,
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as r1,
                    SUM(CASE WHEN rating = 2 THEN 1 ELSE 0 END) as r2,
                    SUM(CASE WHEN rating = 3 THEN 1 ELSE 0 END) as r3,
                    SUM(CASE WHEN rating = 4 THEN 1 ELSE 0 END) as r4,
                    SUM(CASE WHEN rating = 5 THEN 1 ELSE 0 END) as r5
                FROM rag_feedback
            """)).fetchone()

            # Consultas frecuentes (top 5)
            frecuentes = self._session.execute(text("""
                SELECT pregunta, COUNT(*) as veces
                FROM rag_consultas
                GROUP BY pregunta
                ORDER BY veces DESC
                LIMIT 5
            """)).fetchall()

            # Documentos más citados
            docs_citados = self._session.execute(text("""
                SELECT
                    doc_id,
                    COUNT(*) as veces_citado
                FROM rag_consultas,
                    jsonb_array_elements_text(documentos_citados) as doc_id
                GROUP BY doc_id
                ORDER BY veces_citado DESC
                LIMIT 5
            """)).fetchall()

            # Promedios
            promedios = self._session.execute(text("""
                SELECT
                    COALESCE(AVG(relevancia_pct), 0) as rel_prom,
                    COALESCE(AVG(tiempo_respuesta_ms), 0) as tiempo_prom
                FROM rag_consultas
                WHERE relevancia_pct IS NOT NULL
            """)).fetchone()

            return MetricasRAG(
                total_consultas=int(total),
                satisfaccion_promedio=round(float(sat.promedio or 0), 2),
                consultas_con_feedback=int(sat.total or 0),
                rating_1=int(sat.r1 or 0),
                rating_2=int(sat.r2 or 0),
                rating_3=int(sat.r3 or 0),
                rating_4=int(sat.r4 or 0),
                rating_5=int(sat.r5 or 0),
                consultas_frecuentes=[
                    {"pregunta": r.pregunta[:100], "veces": int(r.veces)}
                    for r in frecuentes
                ],
                documentos_mas_citados=[
                    {"documento_id": r.doc_id, "veces_citado": int(r.veces_citado)}
                    for r in docs_citados
                ],
                relevancia_promedio_pct=round(float(promedios.rel_prom or 0), 2),
                tiempo_respuesta_promedio_ms=round(float(promedios.tiempo_prom or 0), 2),
            )
        except Exception as e:
            logger.error("Error calculando métricas: %s", e)
            return self._empty_metricas()

    def _empty_metricas(self) -> MetricasRAG:
        return MetricasRAG(
            total_consultas=0, satisfaccion_promedio=0.0, consultas_con_feedback=0,
            rating_1=0, rating_2=0, rating_3=0, rating_4=0, rating_5=0,
            consultas_frecuentes=[], documentos_mas_citados=[],
            relevancia_promedio_pct=0.0, tiempo_respuesta_promedio_ms=0.0,
        )
