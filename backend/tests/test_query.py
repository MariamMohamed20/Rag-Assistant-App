"""
Tests mock retrieval/generation so they run without a live Ollama process or a
built vector store - useful for CI. Run a manual test against Swagger UI too
once the real store + Ollama are up (see README).
"""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_query_happy_path():
    fake_chunks = [
        {"text": "A bar chart is used for ranking.", "source": "lecture_2.txt", "distance": 0.1}
    ]
    with patch("app.api.routes.query.retrieval.is_ready", return_value=True), \
         patch("app.api.routes.query.retrieval.retrieve", return_value=fake_chunks), \
         patch("app.api.routes.query.generation.generate_answer", return_value="Use a bar chart for ranking (see [1])."):
        response = client.post("/query", json={"question": "When should I use a bar chart?"})

    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["sources"] == ["lecture_2.txt"]


def test_query_invalid_input_returns_422():
    # Missing the required "question" field
    response = client.post("/query", json={})
    assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
