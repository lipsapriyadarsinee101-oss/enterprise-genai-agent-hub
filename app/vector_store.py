"""Dense retrieval. OpenSearch indexes must be ingested explicitly, never on API startup."""
import hashlib

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.retrieval import KnowledgeBase, SearchResult


class VectorKnowledgeBase:
    def __init__(self, settings, embeddings=None, client=None):
        self.settings = settings
        self.chunks = KnowledgeBase(settings.knowledge_dir).chunks
        self.embeddings = embeddings or OpenAIEmbeddings(
            model=settings.embedding_model, dimensions=settings.embedding_dimensions,
            api_key=settings.openai_api_key)
        self.store = None
        if settings.retrieval_backend == 'faiss':
            from langchain_community.vectorstores import FAISS
            if self.chunks:
                self.store = FAISS.from_documents(
                    [Document(page_content=c.text, metadata={'source': c.source}) for c in self.chunks],
                    self.embeddings)
        else:
            from opensearchpy import OpenSearch
            self.client = client or OpenSearch(settings.opensearch_url, timeout=30, verify_certs=True)

    @property
    def documents(self):
        return sorted({c.source for c in self.chunks})

    def ingest(self):
        """Use a new index name when changing embedding model/dimensions or corpus."""
        if self.settings.retrieval_backend != 'opensearch':
            raise ValueError('ingest requires the OpenSearch backend')
        index = self.settings.opensearch_index
        if self.client.indices.exists(index=index):
            raise ValueError('Index already exists; choose a new OPENSEARCH_INDEX to avoid stale chunks')
        vectors = self.embeddings.embed_documents([c.text for c in self.chunks]) if self.chunks else []
        if any(len(v) != self.settings.embedding_dimensions for v in vectors):
            raise ValueError('Embedding dimensions do not match index')
        self.client.indices.create(index=index, body={
            'settings': {'index': {'knn': True}},
            'mappings': {'properties': {
                'vector': {'type': 'knn_vector', 'dimension': self.settings.embedding_dimensions,
                           'method': {'name': 'hnsw', 'engine': 'lucene', 'space_type': 'cosinesimil'}},
                'text': {'type': 'text'}, 'source': {'type': 'keyword'}}}})
        for c, vector in zip(self.chunks, vectors):
            key = hashlib.sha256((c.source + c.text).encode()).hexdigest()
            self.client.index(index=index, id=key, body={'source': c.source, 'text': c.text, 'vector': vector})
        self.client.indices.refresh(index=index)

    def search(self, query, top_k=3):
        if not query.strip():
            return []
        if self.settings.retrieval_backend == 'faiss':
            if self.store is None:
                return []
            # FAISS returns squared L2 distance, not calibrated confidence.
            return [SearchResult(d.metadata['source'], d.page_content, 1/(1+max(0,float(distance))))
                    for d, distance in self.store.similarity_search_with_score(query, k=top_k)]
        vector = self.embeddings.embed_query(query)
        result = self.client.search(index=self.settings.opensearch_index, body={
            'size': top_k, 'query': {'knn': {'vector': {'vector': vector, 'k': top_k}}}})
        return [SearchResult(h['_source']['source'], h['_source']['text'], float(h['_score']))
                for h in result['hits']['hits']]
