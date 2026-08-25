from app.retrieval import KnowledgeBase


def test_search_returns_relevant_source():
    kb = KnowledgeBase("data/knowledge")
    results = kb.search("monitor retrieval relevance and citation coverage", top_k=2)
    assert results
    assert results[0].source == "rag-operations.md"


def test_documents_are_loaded():
    kb = KnowledgeBase("data/knowledge")
    assert len(kb.documents) == 3
