from app.config import Settings
from app.vector_store import VectorKnowledgeBase

if __name__ == '__main__':
    settings = Settings()
    if settings.retrieval_backend != 'opensearch':
        raise SystemExit('Set RETRIEVAL_BACKEND=opensearch before ingesting')
    kb = VectorKnowledgeBase(settings)
    kb.ingest()
    print(f'Indexed {len(kb.chunks)} chunks')
