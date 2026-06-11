# 📚 AgroVision RAG - Sistema de Consultas Agrícolas

> Servicio de Retrieval-Augmented Generation con LangChain y ChromaDB para consultas agrícolas en lenguaje natural.

## Descripción

Servicio independiente que implementa un sistema RAG para responder consultas agrícolas basándose en documentación técnica oficial de FAO, ICA y AGROSAVIA. Procesa documentos PDF/DOCX, genera embeddings vectoriales, realiza búsqueda semántica y genera respuestas contextualizadas con citación de fuentes. Los metadatos de documentos se persisten en la tabla `documentos` y el estado de indexación en `indice_rag_documentos`.

## Stack Tecnológico

| Tecnología | Versión | Propósito |
|---|---|---|
| Python | 3.11 | Lenguaje de programación |
| FastAPI | 0.109.0 | Framework API async |
| LangChain | 0.1.0 | Orquestación pipeline RAG |
| ChromaDB | 0.4.22 | Base de datos vectorial |
| OpenAI | 1.10.0 | Embeddings y generación de respuestas |
| PyPDF2 | 3.0.1 | Procesamiento de PDFs |
| python-docx | 1.1.0 | Procesamiento de Word |
| tiktoken | 0.5.2 | Tokenización |
| BeautifulSoup4 | 4.12.0 | Procesamiento de HTML |
| Pydantic | 2.5.3 | Validación de datos |

## Estructura del Proyecto

```
src/
├── api/
│   ├── routes/                 # Endpoints FastAPI
│   │   ├── query.py            # POST /query - Consultas RAG
│   │   ├── documents.py        # CRUD de documentos indexados
│   │   └── health.py           # GET /health
│   └── middleware/              # Auth, logging, error handling
├── chains/                     # LangChain RAG chains
│   ├── qa_chain.py             # Chain principal de pregunta-respuesta
│   └── citation_chain.py       # Chain con citación de fuentes
├── document_loaders/           # Carga y procesamiento de documentos
│   ├── pdf_loader.py           # PyPDF2 para PDFs
│   ├── docx_loader.py          # python-docx para Word
│   └── html_loader.py          # BeautifulSoup para HTML
├── embeddings/                 # Generación de embeddings
│   └── openai_embeddings.py    # OpenAI text-embedding-3-large (3072 dim)
├── retriever/                  # Búsqueda semántica
│   └── similarity_retriever.py # Top-k=5, cosine distance
├── vectorstore/                # Gestión de ChromaDB
│   └── chroma_store.py         # Persistent storage, metadata filtering
├── services/
│   ├── rag_service.py          # Orquestación del pipeline completo
│   ├── ingestion_service.py    # Ingesta y chunking de documentos
│   └── citation_service.py     # Tracking de fuentes y páginas
├── schemas/                    # Pydantic models
├── config/                     # Settings y configuración
└── utils/                      # Text splitter, helpers

data/
├── documents/                  # PDFs/DOCX originales (FAO, ICA, AGROSAVIA)
└── chroma/                     # ChromaDB persistent vector storage

scripts/                        # Scripts de ingesta masiva de documentos
tests/
├── unit/
└── integration/
docs/
```

## Interacción con Base de Datos (ER v3)

El servicio RAG lee y escribe en las siguientes tablas de PostgreSQL:

| Tabla | Uso |
|---|---|
| `documentos` | Lectura/Escritura: metadatos de documentos (título, categoría, ruta S3, tipo archivo, tamaño, idioma, estado indexación) |
| `indice_rag_documentos` | Escritura: estado técnico en ChromaDB (colección, chunks, modelo embedding, dimensiones, vector IDs JSONB, duración indexación) |
| `recomendaciones` | Lectura: las recomendaciones pueden referenciar un `documento_fuente_id` para trazar la fuente RAG |

### Flujo de Indexación → BD

1. Se sube un documento → se crea registro en `documentos` con `estado_indexacion='pendiente'`
2. El pipeline procesa el documento → actualiza `estado_indexacion='procesando'`
3. Se generan chunks y embeddings → se insertan vectores en ChromaDB
4. Se crea registro en `indice_rag_documentos` con: `nombre_coleccion`, `cantidad_chunks`, `tamano_chunk_tokens`, `overlap_tokens`, `modelo_embedding`, `dimensiones_embedding`, `ids_vectores` (JSONB array)
5. Se actualiza `documentos.estado_indexacion='indexado'`

### Categorías de Documentos

Definidas en `documentos.categoria`: `manual_fao`, `guia_ica`, `agrosavia`, `ficha_tecnica`, `boletin_climatico`

## Endpoints API

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/query` | Consulta en lenguaje natural al sistema RAG |
| `POST` | `/documents` | Ingestar nuevo documento (PDF/DOCX/HTML) |
| `GET` | `/documents` | Listar documentos indexados |
| `GET` | `/documents/{id}` | Detalle de documento indexado + estado en ChromaDB |
| `DELETE` | `/documents/{id}` | Eliminar documento del índice vectorial (usa `ids_vectores` para borrado selectivo) |
| `POST` | `/documents/{id}/reindex` | Re-indexar documento (cuando cambia modelo de embeddings) |
| `GET` | `/health` | Health check del servicio |

## Pipeline RAG

```
Consulta del usuario
    ↓
1. Query Processing (limpieza, normalización)
    ↓
2. Embedding Generation (OpenAI text-embedding-3-large, 3072 dim)
    ↓
3. Similarity Search (ChromaDB, top-k=5, cosine distance)
    ↓
4. Context Assembly (chunks relevantes + metadata de documentos)
    ↓
5. Response Generation (GPT-4, temperature=0.3, max_tokens=500)
    ↓
6. Citation Injection (documento fuente, página, sección, categoria)
    ↓
Respuesta con fuentes citadas
```

## Configuración de Ingesta

- **Text Splitter**: RecursiveCharacterTextSplitter
- **Chunk size**: 512 tokens (almacenado en `indice_rag_documentos.tamano_chunk_tokens`)
- **Chunk overlap**: 50 tokens (almacenado en `indice_rag_documentos.overlap_tokens`)
- **Embeddings**: OpenAI text-embedding-3-large, 3072 dimensiones
- **Metadata**: título, autor, institución, fecha, categoría, páginas

## Fuentes Documentales

| Institución | Tipo de Documentos | Categoría BD |
|---|---|---|
| **FAO** | Manuales de buenas prácticas agrícolas, guías de cultivos | `manual_fao` |
| **ICA** | Protocolos fitosanitarios, manejo integrado de plagas | `guia_ica` |
| **AGROSAVIA** | Publicaciones de investigación, boletines técnicos | `agrosavia` |

Meta: 100+ documentos indexados en fase inicial.

## Variables de Entorno

```env
# General
APP_ENV=development
APP_PORT=8001

# Base de datos (para persistir estado de documentos e índice)
DATABASE_URL=postgresql://user:password@localhost:5432/agrovision

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# ChromaDB
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=agrovision_docs

# LLM Generation
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=500

# Retrieval
RETRIEVER_TOP_K=5
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Fallback LLM (opcional)
FALLBACK_LLM_MODEL=claude-3-5-sonnet
ANTHROPIC_API_KEY=your-anthropic-key-optional
```

## Instalación y Ejecución

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env

# Ingestar documentos iniciales
python scripts/ingest_documents.py --dir ./data/documents

# Desarrollo
uvicorn src.main:app --reload --port 8001

# Producción
uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 2

# Tests
pytest tests/
pytest tests/ --cov=src --cov-report=html
```

## Documentación API

Documentación automática disponible en:
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## Comunicación con Otros Servicios

| Origen | Protocolo | Descripción |
|---|---|---|
| NestJS Backend | HTTP REST | Recibe consultas del chatbot module y gestión de documentos |
| OpenAI API | HTTP (SDK) | Embeddings y generación de respuestas |

## Métricas de Calidad

| Métrica | Objetivo |
|---|---|
| Relevancia de respuestas | > 85% satisfacción |
| Latencia de consulta (p95) | < 3s |
| Citación correcta de fuentes | > 95% |
| Documentos indexados | 100+ en fase inicial |

## Contribución

1. Crear branch desde `develop`: `git checkout -b feature/nombre-feature`
2. Commits con convención: `feat:`, `fix:`, `docs:`, `refactor:`
3. Pull Request hacia `develop`

## Licencia

Proyecto privado - AgroVision © 2026
