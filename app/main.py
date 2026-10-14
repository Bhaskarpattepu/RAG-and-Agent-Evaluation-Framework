from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from app.agent import answer
from app.rag import ingest

app = FastAPI(title="RAG + Agent Evaluation Framework", version="1.0.0")

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

class IngestRequest(BaseModel):
    filename: str = Field(min_length=1)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
def ingest_file(request: IngestRequest):
    # Deliberately restrict files to a local, operator-managed directory.
    root = Path("data/documents").resolve()
    path = (root / request.filename).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise HTTPException(status_code=400, detail="File must exist under data/documents")
    try:
        return {"chunks": ingest(path), "filename": path.name}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/ask")
def ask(request: AskRequest):
    return answer(request.question)
