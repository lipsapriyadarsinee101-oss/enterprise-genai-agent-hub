import os
import time

from app.agents import QualityAgent, RouterAgent, SynthesisAgent
from app.config import Settings
from app.models import Citation, QueryResponse
from app.retrieval import KnowledgeBase
from app.workflow import build_graph


class AgentHub:
    def __init__(self, settings: Settings):
        self.settings = settings
        if settings.retrieval_backend == 'tfidf':
            self.kb = KnowledgeBase(settings.knowledge_dir)
        else:
            from app.vector_store import VectorKnowledgeBase
            self.kb = VectorKnowledgeBase(settings)
        self.router = RouterAgent()
        self.synthesizer = SynthesisAgent(settings.openai_model, settings.openai_api_key, settings.demo_mode)
        self.quality = QualityAgent()
        self.graph = build_graph(self)
        self.callbacks = []
        if settings.langfuse_enabled:
            if not all(os.getenv(k) for k in ('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY')):
                raise ValueError('Langfuse enabled but credentials are missing')
            from langfuse.langchain import CallbackHandler
            self.callbacks.append(CallbackHandler())

    def query(self, question: str, top_k: int = 3) -> QueryResponse:
        if not question.strip() or not 1 <= top_k <= 8:
            raise ValueError('Provide a question and top_k between 1 and 8')
        started = time.perf_counter()
        state = self.graph.invoke({'question': question, 'top_k': top_k},
                                  config={'callbacks': self.callbacks})
        return QueryResponse(
            answer=state['answer'], route=state['route'],
            citations=[Citation(source=e.source, excerpt=e.text[:280], score=e.score)
                       for e in state['evidence']],
            trace=state['trace'], confidence=state['confidence'], grounded=state['grounded'],
            latency_ms=round((time.perf_counter()-started)*1000), mode=state['mode'])
