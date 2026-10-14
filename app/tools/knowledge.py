
import json

from langchain_core.tools import tool
from app.rag import retrieve


@tool
def search_knowledge_base(query: str) -> str:
    """
    Search company policy documents.

    Use this tool for questions about refunds,
    shipping, cancellation, and company policies.
    """
    documents = retrieve(query, k=4)

    return json.dumps(
        documents,
        ensure_ascii=False
    )
