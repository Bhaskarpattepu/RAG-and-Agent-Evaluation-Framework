from pathlib import Path
from app.rag import ingest

if __name__ == "__main__":
    count = ingest(Path("data/documents/refund_policy.txt"))
    print(f"Indexed {count} chunks")
