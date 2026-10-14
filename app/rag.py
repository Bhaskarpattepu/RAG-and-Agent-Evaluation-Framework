from pathlib import Path
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from app.config import settings
from dotenv import load_dotenv

load_dotenv()
COLLECTION = "knowledge"

def store():
    return Chroma(collection_name=COLLECTION,
                  embedding_function=OpenAIEmbeddings(model=settings.embedding_model),
                  persist_directory=settings.chroma_path)

def load_document(path: Path):
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return [Document(page_content=page.extract_text() or "",
                         metadata={"source": path.name, "page": i + 1})
                for i, page in enumerate(reader.pages)]
    if path.suffix.lower() in (".txt", ".md"):
        return [Document(page_content=path.read_text(encoding="utf-8"),
                         metadata={"source": path.name})]
    raise ValueError("Supported formats: .txt, .md, .pdf")

def ingest(path: Path):
    documents = [d for d in load_document(path) if d.page_content.strip()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120).split_documents(documents)
    if not chunks:
        raise ValueError("No extractable text found")
    store().add_documents(chunks)
    return len(chunks)

def retrieve(query: str, k: int = 4):
    docs = store().similarity_search(query, k=k)
    return [{"text": d.page_content, "source": d.metadata.get("source", "unknown"),
             "page": d.metadata.get("page")} for d in docs]
