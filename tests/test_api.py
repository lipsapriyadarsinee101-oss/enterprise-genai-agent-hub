from fastapi.testclient import TestClient

from app.api import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_query_validation():
    with TestClient(app) as client:
        response = client.post("/v1/query", json={"question": "x"})
    assert response.status_code == 422
