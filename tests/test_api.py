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


def test_response_exposes_limits_without_confidence(monkeypatch):
    from app.config import Settings
    monkeypatch.setattr('app.api.get_settings', lambda: Settings(demo_mode=True, knowledge_dir='data/knowledge'))
    with TestClient(app) as client:
        response = client.post('/v1/query', json={'question': 'What are pre-deployment controls?'})
    assert response.status_code == 200
    data = response.json()
    assert data['confidence'] is None and data['grounded'] is None
    assert data['quality']['support_status'] == 'exact_extract'
    assert data['usage']['cost_usd'] is None
    assert not data['abstained']
