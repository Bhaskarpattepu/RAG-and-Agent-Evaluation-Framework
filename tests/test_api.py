from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_question_validation():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422

def test_ingest_rejects_path_traversal():
    response = client.post("/ingest", json={"filename": "../../.env"})
    assert response.status_code == 400

def test_ask_contract(monkeypatch):
    monkeypatch.setattr("app.main.answer", lambda q: {"answer": "Demo", "sources": [], "used_tool": False})
    response = client.post("/ask", json={"question": "Hello"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Demo"
