import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import get_settings
from src.api.routes import health_router, query_router, documents_router
from src.api.middleware.logging_middleware import LoggingMiddleware

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agrovision-rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Iniciando %s v%s [%s]", settings.APP_NAME, settings.APP_VERSION, settings.APP_ENV)
    logger.info("ChromaDB path: %s", settings.CHROMA_PERSIST_DIRECTORY)
    logger.info("Swagger UI disponible en: http://localhost:%s/docs", settings.APP_PORT)

    yield

    logger.info("Apagando %s", settings.APP_NAME)


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Servicio de Retrieval-Augmented Generation para consultas agrícolas. "
            "Usa ChromaDB como vector store, LangChain para orquestación y OpenAI para embeddings/generación. "
            "Indexa documentos técnicos de FAO, ICA y AGROSAVIA. "
            "Parte del sistema AgroVision Predictor & RAG-Support."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:4000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(LoggingMiddleware)

    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(documents_router)

    return app


app = create_app()
