import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.mark.asyncio
async def test_health_check():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "agrovision-rag"
    assert "version" in data
    assert "chromadb_connected" in data
    assert "openai_configured" in data


@pytest.mark.asyncio
async def test_documents_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/documents")

    assert response.status_code == 200
    data = response.json()
    assert "documentos" in data
    assert "total" in data
    assert "total_chunks" in data


@pytest.mark.asyncio
async def test_query_rag():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={
            "pregunta": "¿Cuál es la temperatura óptima para el cacao?",
            "top_k": 3,
        })

    assert response.status_code == 200
    data = response.json()
    assert "respuesta" in data
    assert "fuentes" in data
    assert "pregunta_original" in data
    assert data["pregunta_original"] == "¿Cuál es la temperatura óptima para el cacao?"
    assert "tiempo_respuesta_ms" in data


@pytest.mark.asyncio
async def test_query_validation_error():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/query", json={
            "pregunta": "",
        })

    assert response.status_code == 422
