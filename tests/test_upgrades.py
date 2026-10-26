from unittest.mock import MagicMock
import pytest
from pydantic import ValidationError
from langchain_core.embeddings import Embeddings
from app.config import Settings
from app.judge import JudgeScore, judge_answer
from app.vector_store import VectorKnowledgeBase


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [float('monitor' in text.lower()), float('privacy' in text.lower()), 1.0]


def test_faiss_retrieves_with_real_index(tmp_path):
    (tmp_path / 'monitor.md').write_text('Monitor retrieval latency and monitor response quality across the entire pipeline.')
    (tmp_path / 'privacy.md').write_text('Privacy protection requires a privacy review before processing personal information.')
    kb = VectorKnowledgeBase(Settings(knowledge_dir=str(tmp_path), retrieval_backend='faiss'), TinyEmbeddings())
    assert kb.search('monitor', 1)[0].source == 'monitor.md'
    assert kb.search('') == []


def test_opensearch_ingestion_and_query_contract():
    client = MagicMock()
    client.indices.exists.return_value = False
    settings = Settings(retrieval_backend='opensearch', embedding_dimensions=3)
    kb = VectorKnowledgeBase(settings, TinyEmbeddings(), client)
    kb.ingest()
    mapping = client.indices.create.call_args.kwargs['body']['mappings']
    assert mapping['properties']['vector']['dimension'] == 3
    assert client.index.call_count == len(kb.chunks)
    client.search.return_value = {'hits': {'hits': [
        {'_source': {'source': 'policy.md', 'text': 'Evidence'}, '_score': .9}]}}
    assert kb.search('monitor', 1)[0].source == 'policy.md'
    assert client.search.call_args.kwargs['body']['query']['knn']['vector']['k'] == 1


def test_opensearch_existing_index_is_not_overwritten():
    client = MagicMock()
    client.indices.exists.return_value = True
    kb = VectorKnowledgeBase(Settings(retrieval_backend='opensearch'), TinyEmbeddings(), client)
    with pytest.raises(ValueError, match='already exists'):
        kb.ingest()
    client.index.assert_not_called()


def test_judge_structured_output_and_bounds():
    from langchain_core.runnables import RunnableLambda
    captured = []
    class Model:
        def with_structured_output(self, schema):
            def answer(prompt):
                captured.append(prompt.to_messages())
                return schema(correctness=.8, grounding=.9, explanation='Supported')
            return RunnableLambda(answer)
    score = judge_answer(Model(), 'question', 'answer', ['context'], 'reference')
    assert score.grounding == .9
    assert 'reference' in captured[0][1].content
    with pytest.raises(ValidationError):
        JudgeScore(correctness=2, grounding=.5, explanation='Invalid')


def test_empty_faiss_corpus(tmp_path):
    kb = VectorKnowledgeBase(Settings(knowledge_dir=str(tmp_path), retrieval_backend='faiss'), TinyEmbeddings())
    assert kb.search('anything') == []


def test_bad_embedding_dimension_does_not_create_index():
    client = MagicMock()
    client.indices.exists.return_value = False
    kb = VectorKnowledgeBase(Settings(retrieval_backend='opensearch', embedding_dimensions=5), TinyEmbeddings(), client)
    with pytest.raises(ValueError, match='dimensions'):
        kb.ingest()
    client.indices.create.assert_not_called()


def test_langchain_synthesis_with_fake_model(monkeypatch):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.agents import SynthesisAgent
    from app.retrieval import SearchResult
    monkeypatch.setattr('langchain_openai.ChatOpenAI', lambda **kw: FakeListChatModel(responses=['Supported [policy.md]']))
    answer, _, mode = SynthesisAgent('test-model', 'fake-key', False).run(
        'question', 'policy', [SearchResult('policy.md', 'Evidence', .8)])
    assert answer == 'Supported [policy.md]'
    assert mode == 'openai'
