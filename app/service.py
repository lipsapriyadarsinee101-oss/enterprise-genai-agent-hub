import time

from app.agents import QualityAgent, RouterAgent, SynthesisAgent
from app.config import Settings
from app.models import AgentStep, Citation, QueryResponse
from app.retrieval import KnowledgeBase


class AgentHub:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.kb = KnowledgeBase(settings.knowledge_dir)
        self.router = RouterAgent()
        self.synthesizer = SynthesisAgent(settings.openai_model, settings.openai_api_key, settings.demo_mode)
        self.quality = QualityAgent()

    def query(self, question: str, top_k: int = 3) -> QueryResponse:
        started = time.perf_counter()
        trace: list[AgentStep] = []
        route, step = self.router.run(question)
        trace.append(step)
        evidence = self.kb.search(question, top_k)
        trace.append(AgentStep(agent=f"{route}_specialist", action="retrieve", detail=f"Retrieved {len(evidence)} evidence chunks"))
        answer, step, mode = self.synthesizer.run(question, route, evidence)
        trace.append(step)
        confidence, grounded, step = self.quality.run(answer, evidence)
        trace.append(step)
        citations = [Citation(source=e.source, excerpt=e.text[:280], score=e.score) for e in evidence]
        return QueryResponse(
            answer=answer,
            route=route,
            citations=citations,
            trace=trace,
            confidence=confidence,
            grounded=grounded,
            latency_ms=round((time.perf_counter() - started) * 1000),
            mode=mode,
        )
