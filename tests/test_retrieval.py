from app import rag

def test_retrieval_shape(monkeypatch):
    class FakeDoc:
        page_content = "Refunds are available within 30 days"
        metadata = {"source": "policy.txt", "page": 1}
    class FakeStore:
        def similarity_search(self, query, k):
            assert query == "refund"
            return [FakeDoc()]
    monkeypatch.setattr(rag, "store", lambda: FakeStore())
    results = rag.retrieve("refund")
    assert results[0]["source"] == "policy.txt"
    assert "30 days" in results[0]["text"]
